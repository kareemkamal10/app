#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the ten performer/user batches sequentially.

For each batch:
  select performers -> download all image_urls -> face detection
  -> package originals into normal Parquet -> upload Parquet + face report
  -> verify -> update checkpoint -> delete local batch.

The production batches are 01..N. Batch 00 is reserved for the smoke test.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import (
    batch_bucket_files,
    download_bucket_files,
    get_bucket_paths_info,
)


def run(cmd, env=None):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    p = subprocess.run(cmd, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {p.returncode}: {cmd}")


def retry_call(label, fn, attempts=4, base_delay=5):
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt == attempts:
                break
            delay = base_delay * attempt
            print(f"{label} failed (attempt {attempt}/{attempts}): {exc}; retrying in {delay}s...", flush=True)
            time.sleep(delay)
    raise RuntimeError(f"{label} failed after {attempts} attempts") from last_exc


def upload_bucket_file(bucket: str, local_path: Path, remote_path: str, token: str):
    def op():
        batch_bucket_files(
            bucket,
            add=[(local_path, remote_path)],
            token=token,
        )
        infos = list(get_bucket_paths_info(bucket, [remote_path], token=token))
        found = {x.path: x for x in infos}
        info = found.get(remote_path)
        if info is None:
            raise RuntimeError(f"Bucket verification failed: {remote_path} was not found.")
        if info.size != local_path.stat().st_size:
            raise RuntimeError(
                f"Bucket size mismatch for {remote_path}: "
                f"remote={info.size}, local={local_path.stat().st_size}"
            )

    retry_call(f"Upload {remote_path}", op)


def upload_master_report(bucket: str, report: Path, token: str, remote_name: str):
    upload_bucket_file(bucket, report, remote_name, token)


def fetch_remote_master(bucket: str, token: str, work_dir: Path):
    local = work_dir / "remote_download_master_report.json"
    try:
        def op():
            if local.exists():
                local.unlink()
            download_bucket_files(
                bucket,
                files=[("final/download_master_report.json", local)],
                token=token,
            )
            return local

        retry_call("Download master checkpoint", op, attempts=3, base_delay=3)
    except Exception:
        return None

    try:
        with open(local, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception:
        return None


def remote_batch_complete(bucket: str, token: str, batch: int) -> bool:
    prefix = f"batches/batch_{batch:02d}"
    package_remote = f"{prefix}/batch_{batch:02d}.parquet"
    report_remote = f"{prefix}/batch_{batch:02d}_face_detection_report.json"
    try:
        infos = list(
            get_bucket_paths_info(
                bucket,
                [package_remote, report_remote],
                token=token,
            )
        )
    except Exception:
        return False

    found = {x.path for x in infos}
    return package_remote in found and report_remote in found


def load_json_localized(json_source: str, work_dir: Path, token: str) -> Path:
    """
    Fetch the performer JSON once per pipeline run and keep a local copy.
    This prevents every batch from making another remote JSON request.
    """
    cached = work_dir / "performers_source.json"
    if cached.exists() and cached.stat().st_size > 0:
        return cached

    if json_source.startswith(("http://", "https://")):
        import requests

        headers = {}
        if "huggingface.co/" in json_source and token:
            headers["Authorization"] = f"Bearer {token}"

        def op():
            with requests.get(json_source, headers=headers, timeout=60) as r:
                r.raise_for_status()
                cached.write_bytes(r.content)

        retry_call("Download performer JSON", op, attempts=5, base_delay=3)
    else:
        shutil.copy2(json_source, cached)

    with open(cached, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("performers", "data", "items", "results"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError("JSON must contain a list of performer objects.")

    return cached


def build_master_from_remote_reports(bucket: str, token: str, batch_count: int):
    """
    Reconstruct minimal completed-batch metadata when a checkpoint upload
    succeeded partially or is missing but batch artifacts are already present.
    """
    batches = []
    for batch in range(1, batch_count + 1):
        if not remote_batch_complete(bucket, token, batch):
            continue

        # Only existence is authoritative here. Detailed report values are
        # recovered later when that local report is available.
        batches.append({
            "batch_index": batch,
            "remote_verified": True,
            "parquet": f"batches/batch_{batch:02d}/batch_{batch:02d}.parquet",
        })
    return batches


def checkpoint_master_report(master: dict, reports: Path, bucket: str, token: str):
    reports.mkdir(parents=True, exist_ok=True)

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
            f"| {b.get('batch_index', '')} | {b.get('listed_images', '')} | "
            f"{b.get('download_success', '')} | {b.get('download_failed', '')} | "
            f"{b.get('face_detected', '')} | {b.get('face_not_detected', '')} |"
        )

    master_md.write_text("\n".join(lines), encoding="utf-8")

    upload_master_report(bucket, master_path, token, "final/download_master_report.json")
    upload_master_report(bucket, master_md, token, "final/download_master_report.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-source", required=True)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--batch-count", type=int, default=10)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument(
        "--devices",
        default="0,1",
        help="Comma-separated GPU ids passed through to 02_detect_faces.py.",
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--work-dir", default="./pipeline_work")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Missing required environment variable: HF_TOKEN")

    root = Path(args.work_dir)
    downloads = root / "downloads"
    reports = root / "reports"
    package = root / "package"
    state = root / "state"
    for p in (downloads, reports, package, state):
        p.mkdir(parents=True, exist_ok=True)

    # Download performer JSON once. 01_download_images.py receives the local
    # cached path, so later batches do not depend on another remote JSON fetch.
    cached_json = load_json_localized(args.json_source, root, token)

    remote_master = fetch_remote_master(args.bucket, token, root)
    if remote_master and remote_master.get("json_source") == args.json_source:
        print("Found existing master checkpoint on the bucket — resuming.")
        master = remote_master
        master.setdefault("batches", [])
    else:
        master = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "batch_count": args.batch_count,
            "json_source": args.json_source,
            "bucket": args.bucket,
            "batches": [],
        }

    # Never include the smoke-test batch 00 in production state.
    master["batches"] = [
        b for b in master.get("batches", [])
        if int(b.get("batch_index", -1)) >= 1
    ]

    known = {int(b["batch_index"]) for b in master["batches"] if "batch_index" in b}
    remote_completed = build_master_from_remote_reports(args.bucket, token, args.batch_count)

    for row in remote_completed:
        batch = int(row["batch_index"])
        if batch not in known:
            master["batches"].append(row)
            known.add(batch)

    master["batches"].sort(key=lambda b: int(b.get("batch_index", 0)))

    # Save a checkpoint immediately if remote artifacts proved that prior
    # production batches exist but the checkpoint itself was absent/incomplete.
    if remote_completed and not remote_master:
        checkpoint_master_report(master, reports, args.bucket, token)

    done_batches = {int(b["batch_index"]) for b in master["batches"] if "batch_index" in b}

    for batch in range(1, args.batch_count + 1):
        marker = state / f"batch_{batch:02d}.uploaded.ok"

        if batch in done_batches or remote_batch_complete(args.bucket, token, batch):
            print(
                f"Batch {batch:02d} already uploaded on the bucket/checkpoint; skipping."
            )
            marker.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
            continue

        print(f"\n========== BATCH {batch:02d}/{args.batch_count} ==========")

        run([
            sys.executable,
            "01_download_images.py",
            "--json-source", str(cached_json),
            "--batch-index", str(batch),
            "--batch-count", str(args.batch_count),
            "--output-dir", str(downloads),
            "--report-dir", str(reports),
            "--workers", str(args.workers),
        ])

        download_report = reports / f"batch_{batch:02d}_download_report.json"

        run([
            sys.executable,
            "02_detect_faces.py",
            "--download-report", str(download_report),
            "--report-dir", str(reports),
            "--batch-index", str(batch),
            "--devices", args.devices,
            "--batch-size", str(args.batch_size),
        ])

        face_report = reports / f"batch_{batch:02d}_face_detection_report.json"

        run([
            sys.executable,
            "03_package_batch.py",
            "--download-report", str(download_report),
            "--output-dir", str(package),
            "--batch-index", str(batch),
        ])

        parquet = package / f"batch_{batch:02d}.parquet"

        run([
            sys.executable,
            "04_upload_batch.py",
            "--bucket", args.bucket,
            "--package", str(parquet),
            "--face-report", str(face_report),
            "--batch-index", str(batch),
        ])

        with open(download_report, "r", encoding="utf-8") as f:
            dr = json.load(f)
        with open(face_report, "r", encoding="utf-8") as f:
            fr = json.load(f)

        master["batches"] = [
            b for b in master["batches"]
            if int(b.get("batch_index", -1)) != batch
        ]
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
        master["batches"].sort(key=lambda b: int(b["batch_index"]))

        # Checkpoint must succeed before local files are deleted.
        checkpoint_master_report(master, reports, args.bucket, token)

        if downloads.exists():
            shutil.rmtree(downloads)
            downloads.mkdir(parents=True, exist_ok=True)
        if parquet.exists():
            parquet.unlink()
        marker.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")

        done_batches.add(batch)
        print(f"Checkpoint uploaded after batch {batch:02d}.")

    master["completed_at"] = datetime.now(timezone.utc).isoformat()
    checkpoint_master_report(master, reports, args.bucket, token)
    print("\nALL BATCHES COMPLETED AND MASTER REPORT UPLOADED.")


if __name__ == "__main__":
    main()
