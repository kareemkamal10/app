#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run YOLOv11l-face against the images successfully downloaded in one batch.

Supports splitting the batch across multiple GPUs (e.g. Kaggle's T4 x2):
each device gets its own subprocess with its own model instance, and both
run pass 1 + the fallback pass independently on their shard of the images.
"""

import argparse
import json
import logging
import multiprocessing as mp
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import requests

LOG = logging.getLogger("detect_faces")

# Primary + fallback mirror. GitHub release assets (served off
# objects.githubusercontent.com) sometimes stall indefinitely from Kaggle
# with no error — a plain urlretrieve() has no timeout and will hang
# forever in that case, so we use requests with an explicit timeout,
# retries, and a second source to fall back to.
WEIGHTS_URLS = [
    "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11l-face.pt",
    "https://huggingface.co/deepghs/yolo-face/resolve/main/yolov11l-face/model.pt",
]


@dataclass
class Outcome:
    performer_id: str
    performer_name: str
    path: str
    face_found: bool
    num_faces: int
    best_confidence: float | None
    pass_used: str


def _download_with_timeout(url: str, dest: Path, connect_timeout=15, read_timeout=30, retries=3) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=(connect_timeout, read_timeout)) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                downloaded = 0
                last_log = time.monotonic()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        now = time.monotonic()
                        if now - last_log > 5:
                            pct = f" ({downloaded * 100 // total}%)" if total else ""
                            LOG.info("  ...%d MB%s", downloaded // (1 << 20), pct)
                            last_log = now
            if total and tmp.stat().st_size != total:
                raise IOError(f"Incomplete download: got {tmp.stat().st_size} of {total} bytes")
            tmp.rename(dest)
            return
        except Exception as exc:
            last_exc = exc
            LOG.warning("Attempt %d/%d from %s failed: %s", attempt, retries, url, exc)
            tmp.unlink(missing_ok=True)
            time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to download from {url} after {retries} attempts") from last_exc


def ensure_weights(path: Path) -> Path:
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in WEIGHTS_URLS:
        LOG.info("Downloading YOLO weights from %s ...", url)
        try:
            _download_with_timeout(url, path)
            LOG.info("Downloaded weights: %d MB", path.stat().st_size // (1 << 20))
            return path
        except Exception as exc:
            LOG.warning("Source failed (%s), trying next mirror if any...", exc)
            errors.append(str(exc))
    raise RuntimeError("Could not download YOLO weights from any source: " + "; ".join(errors))


def load_download_report(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)
    return report.get("successful_images", [])


def predict_pass(model, entries, imgsz, conf, batch_size, device, augment, quantize=16, log=LOG):
    outcomes = []
    total = len(entries)
    log.info(
        "Running detection on %d images (mini-batch=%d, augment=%s, quantize=%s)...",
        total, batch_size, augment, quantize
    )
    start_time = time.monotonic()
    for start in range(0, total, batch_size):
        batch = entries[start:start + batch_size]
        paths = [e["path"] for e in batch]
        results = model.predict(
            paths, imgsz=imgsz, conf=conf, device=device,
            augment=augment, quantize=quantize, verbose=False, stream=True
        )
        for e, res in zip(batch, results):
            n = len(res.boxes)
            best = float(res.boxes.conf.max()) if n else None
            outcomes.append((e, n, best))
        done = start + len(batch)
        elapsed = time.monotonic() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        log.info(
            "  %d/%d images (%.1f%%) | %.1f img/s | ETA %.0fs",
            done, total, 100 * done / total, rate, eta
        )
    return outcomes


def process_shard(entries, device, args_dict, log_prefix):
    """Run pass 1 + fallback for one device's shard of images. Runs inside
    its own subprocess (one per GPU) so each has an isolated CUDA context."""
    from ultralytics import YOLO

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s | %(levelname)s | [{log_prefix}] %(message)s",
        force=True,
    )
    log = logging.getLogger(f"detect_faces.{log_prefix}")

    quantize = 16 if args_dict["half"] else None
    model = YOLO(str(ensure_weights(Path(args_dict["weights"]))))
    p1 = predict_pass(
        model, entries, args_dict["imgsz"], args_dict["conf"], args_dict["batch_size"],
        device, augment=False, quantize=quantize, log=log
    )

    outcomes = []
    fallback = []
    for e, n, best in p1:
        if n:
            outcomes.append(Outcome(str(e["performer_id"]), e["performer_name"], e["path"], True, n, best, "pass1"))
        else:
            fallback.append(e)

    if fallback:
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        log.info("Fallback pass: %d images", len(fallback))
        p2 = predict_pass(
            model, fallback, args_dict["fallback_imgsz"], args_dict["fallback_conf"],
            args_dict["fallback_batch_size"], device, augment=True, quantize=quantize, log=log
        )
        for e, n, best in p2:
            outcomes.append(Outcome(
                str(e["performer_id"]), e["performer_name"], e["path"],
                bool(n), n, best, "pass2_fallback" if n else "none"
            ))

    return [asdict(o) for o in outcomes]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-report", required=True)
    ap.add_argument("--report-dir", default="./reports")
    ap.add_argument("--weights", default="./weights/yolov11l-face.pt")
    ap.add_argument("--devices", default="0",
                     help="Comma-separated GPU ids to split the batch across, e.g. '0,1' for Kaggle's T4 x2. "
                          "Each device runs its own subprocess with its own model instance.")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--fallback-imgsz", type=int, default=1280)
    ap.add_argument("--fallback-conf", type=float, default=0.10)
    ap.add_argument("--batch-size", type=int, default=64,
                     help="Mini-batch for the main (no-augment) pass. T4s handle large batches fine here.")
    ap.add_argument("--fallback-batch-size", type=int, default=8,
                     help="Smaller than --batch-size: the fallback pass runs augment=True (TTA), "
                          "which multiplies per-image GPU memory use, and it also uses a larger imgsz.")
    ap.add_argument("--no-half", action="store_true",
                     help="Disable FP16 inference. Only useful if you hit numerical issues; "
                          "T4 supports FP16 natively via Tensor Cores and it roughly doubles throughput.")
    ap.add_argument("--batch-index", type=int, required=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    entries = load_download_report(Path(args.download_report))
    valid = [e for e in entries if e.get("path") and Path(e["path"]).exists()]
    missing = [e for e in entries if not (e.get("path") and Path(e["path"]).exists())]

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]

    # Pre-download weights once here, before any workers start -- otherwise
    # multiple GPU subprocesses could race to write the same file at once.
    ensure_weights(Path(args.weights))

    args_dict = {
        "weights": args.weights, "imgsz": args.imgsz, "conf": args.conf,
        "fallback_imgsz": args.fallback_imgsz, "fallback_conf": args.fallback_conf,
        "batch_size": args.batch_size, "fallback_batch_size": args.fallback_batch_size,
        "half": not args.no_half,
    }

    # Split contiguously (not round-robin) so each shard's file-open pattern
    # stays sequential on disk -- doesn't matter much on Kaggle's local SSD,
    # but avoids surprises with networked storage.
    shards = [[] for _ in devices]
    n = len(devices)
    chunk = (len(valid) + n - 1) // n if n else len(valid)
    for i, dev in enumerate(devices):
        shards[i] = valid[i * chunk:(i + 1) * chunk]

    LOG.info(
        "Splitting %d images across %d device(s): %s",
        len(valid), n, ", ".join(f"gpu{d}={len(s)}" for d, s in zip(devices, shards))
    )

    if n <= 1:
        raw = process_shard(valid, devices[0] if devices else "0", args_dict, "gpu" + (devices[0] if devices else "0"))
    else:
        ctx = mp.get_context("spawn")  # required: CUDA + fork don't mix
        with ctx.Pool(processes=n) as pool:
            results = pool.starmap(
                process_shard,
                [(shards[i], devices[i], args_dict, f"gpu{devices[i]}") for i in range(n)],
            )
        raw = [row for shard_result in results for row in shard_result]

    outcomes = [Outcome(**row) for row in raw]

    for e in missing:
        outcomes.append(Outcome(
            str(e["performer_id"]), e["performer_name"], e.get("path") or "",
            False, 0, None, "missing_file"
        ))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "batch_index": args.batch_index,
        "model": "yolov11l-face",
        "total_checked": len(outcomes),
        "face_detected": sum(o.face_found for o in outcomes),
        "face_not_detected": sum(not o.face_found for o in outcomes),
        "not_detected_list": [
            {
                "id": o.performer_id,
                "name": o.performer_name,
                "path": o.path,
                "reason": (
                    "missing_file" if o.pass_used == "missing_file"
                    else "no_face_after_fallback"
                ),
            }
            for o in outcomes if not o.face_found
        ],
        "all_results": [asdict(o) for o in outcomes],
    }

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"batch_{args.batch_index:02d}_face_detection_report.json"
    md_path = report_dir / f"batch_{args.batch_index:02d}_face_detection_report.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md = [
        f"# Batch {args.batch_index:02d} face detection report",
        f"- Checked: **{report['total_checked']}**",
        f"- Face detected: **{report['face_detected']}**",
        f"- Face not detected: **{report['face_not_detected']}**",
        "",
    ]
    if report["not_detected_list"]:
        md += ["## No-face / missing files", "", "| ID | Name | Path | Reason |", "|---|---|---|---|"]
        for x in report["not_detected_list"]:
            md.append(f"| {x['id']} | {x['name']} | {x['path']} | {x['reason']} |")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
