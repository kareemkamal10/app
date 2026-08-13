#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smoke test: run the FULL pipeline (download -> detect -> package/encrypt ->
upload -> verify) on a small slice of images, end to end, in a completely
isolated location.

This is throwaway. It uses batch-index 0, which the real 10-batch pipeline
(batch_pipeline.py) never uses and never looks at:
  - Remote files land under batches/batch_00/... on the bucket.
  - Nothing is written to final/download_master_report.json (the real
    resume checkpoint) -- only batch_pipeline.py touches that file, and
    this script never calls it.
  - batch_pipeline.py's remote_batch_complete() only ever checks
    batches/batch_01 .. batches/batch_<batch-count>, so batch_00 is
    invisible to it.

So: run this, confirm it works, then start the real 10-batch run exactly as
before -- nothing here affects it.

Uses the same 01/02/03/04 scripts as subprocesses so it's testing the exact
same code path the real pipeline uses, just on --image-count images instead
of a full batch.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(cmd):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {p.returncode}: {cmd}")


def load_performers(json_source: str) -> list:
    import importlib.util
    spec = importlib.util.spec_from_file_location("dl01", HERE / "01_download_images.py")
    dl01 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl01)
    return dl01.load_json(json_source, timeout=25), dl01


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-source", required=True)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--image-count", type=int, default=100)
    ap.add_argument("--work-dir", default="/kaggle/temp/pipeline_smoke_test")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    root = Path(args.work_dir)
    if root.exists():
        shutil.rmtree(root)
    downloads = root / "downloads"
    reports = root / "reports"
    package = root / "package"
    for p in (downloads, reports, package):
        p.mkdir(parents=True, exist_ok=True)

    print(f"Loading performer list from {args.json_source} ...")
    performers, dl01 = load_performers(args.json_source)

    # Flatten to a fixed-order list of (performer, url, index-within-performer)
    # and take the first N images, deterministically.
    flat = []
    for p in performers:
        urls = p.get("image_urls") or []
        for idx, url in enumerate(urls, 1):
            flat.append((p, url, idx))
            if len(flat) >= args.image_count:
                break
        if len(flat) >= args.image_count:
            break

    if not flat:
        raise RuntimeError("No images found in json-source to build a smoke test batch.")
    print(f"Smoke test batch: {len(flat)} images from {len({id(p) for p, _, _ in flat})} performers.")

    tasks = []
    for p, url, idx in flat:
        pid = str(p.get("id"))
        name = p.get("name") or "(no name)"
        dest_dir = downloads / pid
        tasks.append(dl01.ImageTask(pid, name, url, idx, len((p.get("image_urls") or [])), dest_dir))

    results = []
    with dl01.build_session(3, 0.7) as session:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(dl01.download_one, session, t, 25, 512) for t in tasks]
            for fut in as_completed(futures):
                results.append(fut.result())

    ok = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    print(f"Downloaded {len(ok)}/{len(results)} images ({len(failed)} failed).")

    download_report = reports / "batch_00_download_report.json"
    report = {
        "batch_index": 0,
        "listed_images_in_batch": len(results),
        "download_success": len(ok),
        "download_failed": len(failed),
        "successful_images": [
            {
                "performer_id": r.performer_id, "performer_name": r.performer_name,
                "url": r.url, "index": r.index, "path": r.path,
            }
            for r in ok
        ],
        "failed_images": [
            {"performer_id": r.performer_id, "performer_name": r.performer_name,
             "url": r.url, "index": r.index, "error": r.error}
            for r in failed
        ],
    }
    with open(download_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if not ok:
        raise RuntimeError("All smoke-test downloads failed -- nothing to detect/package.")

    run([
        sys.executable, str(HERE / "02_detect_faces.py"),
        "--download-report", str(download_report),
        "--report-dir", str(reports),
        "--batch-index", "0",
        "--device", args.device,
        "--batch-size", str(args.batch_size),
    ])

    face_report = reports / "batch_00_face_detection_report.json"
    with open(face_report, encoding="utf-8") as f:
        fr = json.load(f)
    print(f"Faces detected: {fr['face_detected']}/{fr['total_checked']}")

    run([
        sys.executable, str(HERE / "03_package_batch.py"),
        "--download-report", str(download_report),
        "--output-dir", str(package),
        "--batch-index", "0",
    ])

    encrypted = package / "batch_00.parquet.enc"

    run([
        sys.executable, str(HERE / "04_upload_batch.py"),
        "--bucket", args.bucket,
        "--package", str(encrypted),
        "--face-report", str(face_report),
        "--batch-index", "0",
    ])

    print(
        "\nSMOKE TEST PASSED. Uploaded to batches/batch_00/ on the bucket "
        "-- this is isolated test data, harmless to leave there, and does "
        "NOT affect the real 10-batch run's resume state."
    )
    print(f"Cleaning up local smoke-test files at {root} ...")
    shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
