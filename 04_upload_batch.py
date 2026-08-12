#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Upload one encrypted batch and its face report to a Hugging Face Bucket."""

import argparse
import os
from pathlib import Path

from huggingface_hub import batch_bucket_files, get_bucket_paths_info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True, help="e.g. abdelwahabnabil500/faces")
    ap.add_argument("--package", required=True)
    ap.add_argument("--face-report", required=True)
    ap.add_argument("--batch-index", type=int, required=True)
    ap.add_argument("--token-env", default="HF_TOKEN")
    args = ap.parse_args()

    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"Missing environment variable: {args.token_env}")

    package = Path(args.package)
    report = Path(args.face_report)
    if not package.exists() or not report.exists():
        raise FileNotFoundError("Batch package or face report is missing.")

    prefix = f"batches/batch_{args.batch_index:02d}"
    package_remote = f"{prefix}/{package.name}"
    report_remote = f"{prefix}/{report.name}"

    batch_bucket_files(
        args.bucket,
        add=[
            (package, package_remote),
            (report, report_remote),
        ],
        token=token,
    )

    infos = list(get_bucket_paths_info(
        args.bucket,
        [package_remote, report_remote],
        token=token,
    ))
    found = {x.path: x for x in infos}

    if package_remote not in found or report_remote not in found:
        raise RuntimeError("Remote verification failed: one or more uploaded files were not found.")

    if found[package_remote].size != package.stat().st_size:
        raise RuntimeError("Remote package size does not match local package size.")
    if found[report_remote].size != report.stat().st_size:
        raise RuntimeError("Remote face-report size does not match local report size.")

    print(f"Verified remote upload for batch {args.batch_index:02d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
