#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Package the ORIGINAL downloaded images of one batch into a Parquet file.

The output is a normal, unencrypted Parquet file:
  batch_XX.parquet

After successful remote verification, the pipeline deletes the local
images and Parquet file.
"""

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def collect_images(download_report: Path) -> list[dict]:
    with open(download_report, "r", encoding="utf-8") as f:
        report = json.load(f)
    rows = []
    for item in report.get("successful_images", []):
        path = Path(item["path"])
        if not path.exists():
            continue
        with open(path, "rb") as img:
            data = img.read()
        rows.append({
            "performer_id": str(item["performer_id"]),
            "performer_name": item["performer_name"],
            "image_index": int(item["index"]),
            "image_url": item["url"],
            "filename": path.name,
            "mime_type": "image/" + path.suffix.lower().lstrip("."),
            "image_bytes": data,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-report", required=True)
    ap.add_argument("--output-dir", default="./package")
    ap.add_argument("--batch-index", type=int, required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_images(Path(args.download_report))
    if not rows:
        raise RuntimeError("No successfully downloaded original images found.")

    parquet = out_dir / f"batch_{args.batch_index:02d}.parquet"

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, parquet, compression="zstd", use_dictionary=True)

    # Basic integrity check of the plain Parquet output.
    if parquet.stat().st_size <= 20:
        raise RuntimeError("Parquet package is unexpectedly small.")

    print(json.dumps({
        "batch_index": args.batch_index,
        "rows": len(rows),
        "parquet_file": str(parquet),
        "parquet_bytes": parquet.stat().st_size,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
