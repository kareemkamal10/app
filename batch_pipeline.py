#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the ten batches sequentially on one machine.

For each batch:
  download -> detect -> package originals -> upload package + face report
  -> verify -> delete local batch.

Only after all ten batches succeed:
  upload master download report and master face summary.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run(cmd, env=None):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    p = subprocess.run(cmd, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {p.returncode}: {cmd}")


def upload_master_report(bucket: str, report: Path, token: str, remote_name: str):
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(report),
        path_in_repo=remote_name,
        repo_id=bucket,
        repo_type="dataset",
    )
    api.get_paths_info(repo_id=bucket, repo_type="dataset", paths=[remote_name])


def fetch_remote_master(bucket: str, token: str) -> dict | None:
    """Download the checkpoint master report from the bucket, if one exists.

    Kaggle sessions do not keep local disk between restarts, so resume state
    must live on the remote bucket rather than in ./pipeline_work/state.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    try:
        path = hf_hub_download(
            repo_id=bucket,
            repo_type="dataset",
            filename="final/download_master_report.json",
            token=token,
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None
    except Exception as exc:
        print(f"Could not check remote checkpoint (continuing without it): {exc}")
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def remote_batch_complete(bucket: str, token: str, batch: int) -> bool:
    """Check the bucket directly for a batch's package + face report."""
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    prefix = f"batches/batch_{batch:02d}"
    package_remote = f"{prefix}/batch_{batch:02d}.parquet"
    report_remote = f"{prefix}/batch_{batch:02d}_face_detection_report.json"
    try:
        infos = list(api.get_paths_info(
            repo_id=bucket, repo_type="dataset",
            paths=[package_remote, report_remote],
        ))
    except Exception:
        return False
    found = {x.path for x in infos}
    return package_remote in found and report_remote in found


def checkpoint_master_report(master: dict, reports: Path, bucket: str, token: str):
    """Write + upload the master report after every batch, so a restart can
    resume from the bucket instead of starting over."""
    master_path = reports / "download_master_report.json"
    master_md = reports / "download_master_report.md"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master, f, ensure_ascii=False, indent=2)

    lines = [
        "# Master download / batch report",
        f"- Batches: **{master['batch_count']}**",
        "",
        "| Batch | Images | Download OK | Download failed | Faces | No face |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for b in master["batches"]:
        lines.append(
            f"| {b['batch_index']} | {b['listed_images']} | "
            f"{b['download_success']} | {b['download_failed']} | "
            f"{b['face_detected']} | {b['face_not_detected']} |"
        )
    with open(master_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    upload_master_report(bucket, master_path, token, "final/download_master_report.json")
    from huggingface_hub import HfApi
    HfApi(token=token).upload_file(
        path_or_fileobj=str(master_md),
        path_in_repo="final/download_master_report.md",
        repo_id=bucket,
        repo_type="dataset",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-source", required=True)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--batch-count", type=int, default=10)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--devices", default="0,1",
                     help="Comma-separated GPU ids passed through to 02_detect_faces.py, e.g. '0,1' for T4 x2.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--work-dir", default="./pipeline_work")
    args = ap.parse_args()

    required = ["HF_TOKEN"]
    missing = [x for x in required if not os.environ.get(x)]
    if missing:
        raise RuntimeError("Missing required environment variables: " + ", ".join(missing))

    root = Path(args.work_dir)
    downloads = root / "downloads"
    reports = root / "reports"
    package = root / "package"
    state = root / "state"
    for p in (downloads, reports, package, state):
        p.mkdir(parents=True, exist_ok=True)

    token = os.environ["HF_TOKEN"]

    remote_master = fetch_remote_master(args.bucket, token)
    if remote_master and remote_master.get("json_source") == args.json_source:
        print("Found existing checkpoint on the bucket — resuming from it.")
        master = remote_master
        master.setdefault("batches", [])
    else:
        if remote_master:
            print("Remote checkpoint is for a different json-source; starting fresh.")
        master = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "batch_count": args.batch_count,
            "json_source": args.json_source,
            "bucket": args.bucket,
            "batches": [],
        }

    # Only treat a checkpoint as complete when it points to the current
    # plaintext Parquet format. Older checkpoints may contain .parquet.enc
    # from the previous encrypted version and must be rebuilt.
    done_batches = {
        b["batch_index"]
        for b in master["batches"]
        if str(b.get("parquet", "")).endswith(".parquet")
    }

    for batch in range(1, args.batch_count + 1):
        marker = state / f"batch_{batch:02d}.uploaded.ok"
        if batch in done_batches or marker.exists() or remote_batch_complete(args.bucket, token, batch):
            print(f"Batch {batch:02d} already uploaded (local or remote checkpoint); skipping.")
            marker.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
            continue

        print(f"\n========== BATCH {batch:02d}/{args.batch_count} ==========")

        run([
            sys.executable, "01_download_images.py",
            "--json-source", args.json_source,
            "--batch-index", str(batch),
            "--batch-count", str(args.batch_count),
            "--output-dir", str(downloads),
            "--report-dir", str(reports),
            "--workers", str(args.workers),
        ])

        download_report = reports / f"batch_{batch:02d}_download_report.json"

        run([
            sys.executable, "02_detect_faces.py",
            "--download-report", str(download_report),
            "--report-dir", str(reports),
            "--batch-index", str(batch),
            "--devices", args.devices,
            "--batch-size", str(args.batch_size),
        ])

        face_report = reports / f"batch_{batch:02d}_face_detection_report.json"

        run([
            sys.executable, "03_package_batch.py",
            "--download-report", str(download_report),
            "--output-dir", str(package),
            "--batch-index", str(batch),
        ])

        parquet = package / f"batch_{batch:02d}.parquet"

        run([
            sys.executable, "04_upload_batch.py",
            "--bucket", args.bucket,
            "--package", str(parquet),
            "--face-report", str(face_report),
            "--batch-index", str(batch),
        ])

        # Only now is deletion allowed.
        if downloads.exists():
            shutil.rmtree(downloads)
            downloads.mkdir(parents=True, exist_ok=True)
        if parquet.exists():
            parquet.unlink()

        marker.write_text(
            datetime.now(timezone.utc).isoformat(), encoding="utf-8"
        )

        with open(download_report, "r", encoding="utf-8") as f:
            dr = json.load(f)
        with open(face_report, "r", encoding="utf-8") as f:
            fr = json.load(f)

        master["batches"] = [b for b in master["batches"] if b["batch_index"] != batch]
        master["batches"].append({
            "batch_index": batch,
            "listed_images": dr["listed_images_in_batch"],
            "download_success": dr["download_success"],
            "download_failed": dr["download_failed"],
            "face_detected": fr["face_detected"],
            "face_not_detected": fr["face_not_detected"],
            "download_report": download_report.name,
            "face_report": face_report.name,
            "parquet": f"batches/batch_{batch:02d}/batch_{batch:02d}.parquet",
        })
        master["batches"].sort(key=lambda b: b["batch_index"])

        # Checkpoint after every batch: this is the file a restarted Kaggle
        # session downloads at the top of main() to know what's already done.
        checkpoint_master_report(master, reports, args.bucket, token)
        print(f"Checkpoint uploaded after batch {batch:02d}.")

    master["completed_at"] = datetime.now(timezone.utc).isoformat()
    checkpoint_master_report(master, reports, args.bucket, token)

    print("\nALL BATCHES COMPLETED AND MASTER REPORT UPLOADED.")


if __name__ == "__main__":
    main()
