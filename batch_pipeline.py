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
import tempfile


def run(cmd, env=None):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    p = subprocess.run(cmd, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {p.returncode}: {cmd}")


def upload_master_report(bucket: str, report: Path, token: str, remote_name: str):
    """Upload a small checkpoint file to the Hugging Face Bucket."""
    from huggingface_hub import batch_bucket_files, get_bucket_paths_info

    batch_bucket_files(bucket, add=[(report, remote_name)], token=token)
    infos = list(get_bucket_paths_info(bucket, [remote_name], token=token))
    found = {x.path: x for x in infos}
    if remote_name not in found:
        raise RuntimeError(f"Remote checkpoint upload failed: {remote_name} not found.")
    if found[remote_name].size != report.stat().st_size:
        raise RuntimeError(
            f"Remote checkpoint size mismatch for {remote_name}: "
            f"local={report.stat().st_size}, remote={found[remote_name].size}"
        )


def fetch_remote_master(bucket: str, token: str) -> dict | None:
    """Download the checkpoint master report from the Hugging Face Bucket."""
    from huggingface_hub import HfApi

    remote_name = "final/download_master_report.json"
    api = HfApi(token=token)
    try:
        infos = list(api.get_bucket_paths_info(bucket, [remote_name], token=token))
        if not any(info.path == remote_name for info in infos):
            return None
        tmp = Path(tempfile.gettempdir()) / "performer_pipeline_master_checkpoint.json"
        api.download_bucket_files(bucket, [(remote_name, tmp)], token=token, raise_on_missing_files=True)
        with open(tmp, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"Could not check/download remote checkpoint (continuing without it): {exc}")
        return None
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except UnboundLocalError:
            pass


def remote_batch_complete(bucket: str, token: str, batch: int) -> bool:
    """Check the Hugging Face Bucket directly for a production batch."""
    from huggingface_hub import get_bucket_paths_info

    prefix = f"batches/batch_{batch:02d}"
    paths = [
        f"{prefix}/batch_{batch:02d}.parquet",
        f"{prefix}/batch_{batch:02d}_face_detection_report.json",
    ]
    try:
        found = {x.path for x in get_bucket_paths_info(bucket, paths, token=token)}
        return all(path in found for path in paths)
    except Exception:
        return False


def recover_remote_batch(bucket: str, token: str, batch: int, reports: Path) -> dict | None:
    """Recover metadata for an uploaded production batch when its checkpoint is missing."""
    from huggingface_hub import HfApi

    prefix = f"batches/batch_{batch:02d}"
    package_remote = f"{prefix}/batch_{batch:02d}.parquet"
    face_remote = f"{prefix}/batch_{batch:02d}_face_detection_report.json"
    api = HfApi(token=token)

    try:
        found = {x.path for x in api.get_bucket_paths_info(
            bucket, [package_remote, face_remote], token=token
        )}
        if package_remote not in found or face_remote not in found:
            return None

        local_download = reports / f"batch_{batch:02d}_download_report.json"
        local_face = reports / f"batch_{batch:02d}_face_detection_report.json"
        if local_download.exists() and local_face.exists():
            with open(local_download, "r", encoding="utf-8") as f:
                dr = json.load(f)
            with open(local_face, "r", encoding="utf-8") as f:
                fr = json.load(f)
            return {
                "batch_index": batch,
                "listed_images": dr["listed_images_in_batch"],
                "download_success": dr["download_success"],
                "download_failed": dr["download_failed"],
                "face_detected": fr["face_detected"],
                "face_not_detected": fr["face_not_detected"],
                "download_report": local_download.name,
                "face_report": local_face.name,
                "parquet": package_remote,
            }

        tmp = Path(tempfile.gettempdir()) / f"batch_{batch:02d}_face_report.json"
        api.download_bucket_files(bucket, [(face_remote, tmp)], token=token, raise_on_missing_files=True)
        with open(tmp, "r", encoding="utf-8") as f:
            fr = json.load(f)
        tmp.unlink(missing_ok=True)
        results = fr.get("all_results", [])
        failed = sum(1 for row in results if row.get("pass_used") == "missing_file")
        return {
            "batch_index": batch,
            "listed_images": fr.get("total_checked", len(results)),
            "download_success": len(results) - failed,
            "download_failed": failed,
            "face_detected": fr["face_detected"],
            "face_not_detected": fr["face_not_detected"],
            "download_report": f"batch_{batch:02d}_download_report.json",
            "face_report": Path(face_remote).name,
            "parquet": package_remote,
        }
    except Exception as exc:
        print(f"Could not recover remote batch {batch:02d}: {exc}")
        return None


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
    upload_master_report(bucket, master_md, token, "final/download_master_report.md")


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

    # Recover any already-uploaded production batches missing from the checkpoint.
    known_batches = {
        b.get("batch_index")
        for b in master.get("batches", [])
        if str(b.get("parquet", "")).endswith(".parquet")
    }
    for existing_batch in range(1, args.batch_count + 1):
        if existing_batch in known_batches:
            continue
        recovered = recover_remote_batch(args.bucket, token, existing_batch, reports)
        if recovered is not None:
            master["batches"].append(recovered)
    master["batches"].sort(key=lambda b: b["batch_index"])
    if master["batches"]:
        print("Recognized production batches already on the Bucket: " + ", ".join(
            f"{b['batch_index']:02d}" for b in master["batches"]
        ))

    # Only treat a checkpoint as complete when it points to the current
    # plaintext Parquet format only.
    done_batches = {
        b["batch_index"]
        for b in master["batches"]
        if str(b.get("parquet", "")).endswith(".parquet")
    }

    for batch in range(1, args.batch_count + 1):
        marker = state / f"batch_{batch:02d}.uploaded.ok"
        remote_done = remote_batch_complete(args.bucket, token, batch)
        if batch in done_batches or remote_done or (marker.exists() and remote_done):
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
        marker.write_text(
            datetime.now(timezone.utc).isoformat(), encoding="utf-8"
        )
        print(f"Checkpoint uploaded after batch {batch:02d}.")

    master["completed_at"] = datetime.now(timezone.utc).isoformat()
    checkpoint_master_report(master, reports, args.bucket, token)

    print("\nALL BATCHES COMPLETED AND MASTER REPORT UPLOADED.")


if __name__ == "__main__":
    main()
