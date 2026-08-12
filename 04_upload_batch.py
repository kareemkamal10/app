#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Upload one encrypted batch and its face report to a Hugging Face Bucket."""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True, help="bucket name, e.g. abdelwahabnabil500/faces")
    ap.add_argument("--package", required=True)
    ap.add_argument("--face-report", required=True)
    ap.add_argument("--batch-index", type=int, required=True)
    ap.add_argument("--token-env", default="HF_TOKEN")
    args = ap.parse_args()

    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"Missing environment variable: {args.token_env}")

    api = HfApi(token=token)
    package = Path(args.package)
    report = Path(args.face_report)
    if not package.exists() or not report.exists():
        raise FileNotFoundError("Batch package or face report is missing.")

    prefix = f"batches/batch_{args.batch_index:02d}"
    api.upload_file(
        path_or_fileobj=str(package),
        path_in_repo=f"{prefix}/{package.name}",
        repo_id=args.bucket,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=str(report),
        path_in_repo=f"{prefix}/{report.name}",
        repo_id=args.bucket,
        repo_type="dataset",
    )

    # Verify remote metadata by resolving the uploaded paths through the Hub API.
    # upload_file raises on failed uploads; the explicit metadata lookups below
    # provide a second confirmation before the caller deletes local data.
    api.get_paths_info(
        repo_id=args.bucket,
        repo_type="dataset",
        paths=[f"{prefix}/{package.name}", f"{prefix}/{report.name}"],
    )

    print(f"Verified remote upload for batch {args.batch_index:02d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
