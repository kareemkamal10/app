#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Package the ORIGINAL downloaded images of one batch into a Parquet file,
then encrypt the resulting Parquet with AES-256-GCM using a key derived
from PARQUET_PASSWORD.

The output is a single encrypted container:
  batch_XX.parquet.enc

After successful remote verification, the pipeline deletes the local
images and encrypted file. The plaintext parquet is temporary and is
deleted immediately after encryption.
"""

import argparse
import base64
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def derive_key(password: str, batch_index: int) -> bytes:
    salt = f"performer-parquet-batch-{batch_index}".encode("utf-8")
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 300_000, dklen=32)


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


def encrypt_file(src: Path, dst: Path, password: str, batch_index: int):
    key = derive_key(password, batch_index)
    nonce = os.urandom(12)
    aad = f"performer-batch:{batch_index}:parquet-v1".encode("utf-8")

    with open(src, "rb") as f:
        plaintext = f.read()

    ciphertext = AESGCM(key).encrypt(nonce, plaintext, aad)
    header = {
        "format": "encrypted-parquet-envelope-v1",
        "batch_index": batch_index,
        "cipher": "AES-256-GCM",
        "kdf": "PBKDF2-HMAC-SHA256",
        "iterations": 300000,
        "nonce_b64": base64.b64encode(nonce).decode("ascii"),
        "aad": aad.decode("utf-8"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(dst, "wb") as f:
        f.write(b"EPQ1\n")
        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        f.write(len(header_bytes).to_bytes(4, "big"))
        f.write(header_bytes)
        f.write(ciphertext)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-report", required=True)
    ap.add_argument("--output-dir", default="./package")
    ap.add_argument("--batch-index", type=int, required=True)
    ap.add_argument("--password-env", default="PARQUET_PASSWORD")
    args = ap.parse_args()

    password = os.environ.get(args.password_env)
    if not password:
        raise RuntimeError(f"Missing environment variable: {args.password_env}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_images(Path(args.download_report))
    if not rows:
        raise RuntimeError("No successfully downloaded original images found.")

    plain = out_dir / f"batch_{args.batch_index:02d}.parquet"
    encrypted = out_dir / f"batch_{args.batch_index:02d}.parquet.enc"

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, plain, compression="zstd", use_dictionary=True)

    encrypt_file(plain, encrypted, password, args.batch_index)
    plain.unlink()

    # Integrity check of encrypted output.
    if encrypted.stat().st_size <= 20:
        raise RuntimeError("Encrypted package is unexpectedly small.")

    print(json.dumps({
        "batch_index": args.batch_index,
        "rows": len(rows),
        "encrypted_file": str(encrypted),
        "encrypted_bytes": encrypted.stat().st_size,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
