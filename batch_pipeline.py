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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-source", required=True)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--batch-count", type=int, default=10)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--work-dir", default="./pipeline_work")
    args = ap.parse_args()

    required = ["HF_TOKEN", "PARQUET_PASSWORD"]
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

    master = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "batch_count": args.batch_count,
        "json_source": args.json_source,
        "bucket": args.bucket,
        "batches": [],
    }

    for batch in range(1, args.batch_count + 1):
        marker = state / f"batch_{batch:02d}.uploaded.ok"
        if marker.exists():
            print(f"Batch {batch:02d} already marked uploaded; skipping.")
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
            "--device", args.device,
            "--batch-size", str(args.batch_size),
        ])

        face_report = reports / f"batch_{batch:02d}_face_detection_report.json"

        run([
            sys.executable, "03_package_batch.py",
            "--download-report", str(download_report),
            "--output-dir", str(package),
            "--batch-index", str(batch),
        ])

        encrypted = package / f"batch_{batch:02d}.parquet.enc"

        run([
            sys.executable, "04_upload_batch.py",
            "--bucket", args.bucket,
            "--package", str(encrypted),
            "--face-report", str(face_report),
            "--batch-index", str(batch),
        ])

        # Only now is deletion allowed.
        if downloads.exists():
            shutil.rmtree(downloads)
            downloads.mkdir(parents=True, exist_ok=True)
        if encrypted.exists():
            encrypted.unlink()

        marker.write_text(
            datetime.now(timezone.utc).isoformat(), encoding="utf-8"
        )

        with open(download_report, "r", encoding="utf-8") as f:
            dr = json.load(f)
        with open(face_report, "r", encoding="utf-8") as f:
            fr = json.load(f)

        master["batches"].append({
            "batch_index": batch,
            "listed_images": dr["listed_images_in_batch"],
            "download_success": dr["download_success"],
            "download_failed": dr["download_failed"],
            "face_detected": fr["face_detected"],
            "face_not_detected": fr["face_not_detected"],
            "download_report": download_report.name,
            "face_report": face_report.name,
            "parquet": f"batches/batch_{batch:02d}/batch_{batch:02d}.parquet.enc",
        })

    master_path = reports / "download_master_report.json"
    master_md = reports / "download_master_report.md"
    master["completed_at"] = datetime.now(timezone.utc).isoformat()
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master, f, ensure_ascii=False, indent=2)

    lines = [
        "# Master download / batch report",
        f"- Batches: **{args.batch_count}**",
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

    upload_master_report(
        args.bucket, master_path, os.environ["HF_TOKEN"],
        "final/download_master_report.json"
    )
    # Upload markdown counterpart as well.
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.upload_file(
        path_or_fileobj=str(master_md),
        path_in_repo="final/download_master_report.md",
        repo_id=args.bucket,
        repo_type="dataset",
    )
    api.get_paths_info(
        repo_id=args.bucket,
        repo_type="dataset",
        paths=["final/download_master_report.json", "final/download_master_report.md"],
    )

    print("\nALL BATCHES COMPLETED AND MASTER REPORT UPLOADED.")


if __name__ == "__main__":
    main()
