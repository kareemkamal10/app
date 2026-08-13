#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download one deterministic batch of performer images.

Each batch is selected by performer records, not by individual images.
Every performer belongs to exactly one batch, and all of that performer's
image_urls are downloaded together.

Output:
  reports/batch_XX_download_report.json
  reports/batch_XX_download_report.md
  reports/batch_XX_download_results.jsonl
  downloads/<performer_id>/<index>.<ext>
"""

import argparse
import json
import logging
import mimetypes
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

LOG = logging.getLogger("batch_download")


@dataclass
class ImageTask:
    performer_id: str
    performer_name: str
    url: str
    index: int
    total_for_performer: int
    dest_dir: Path


@dataclass
class ImageResult:
    performer_id: str
    performer_name: str
    url: str
    index: int
    success: bool
    path: Optional[str] = None
    error: Optional[str] = None
    bytes_downloaded: int = 0


def build_session(retries: int, backoff: float) -> requests.Session:
    s = requests.Session()
    retry_cfg = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_maxsize=64, pool_connections=64)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
        )
    })
    return s


def guess_extension(url: str, content_type: str) -> str:
    suffix = Path(url.split("?", 1)[0]).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}:
        return ".jpg" if suffix == ".jpeg" else suffix
    ext = mimetypes.guess_extension((content_type or "").split(";", 1)[0].strip())
    if ext:
        return ".jpg" if ext == ".jpe" else ext
    return ".jpg"


def load_json(source: str, timeout: int) -> list:
    if source.startswith(("http://", "https://")):
        with build_session(3, 0.7) as s:
            r = s.get(source, timeout=timeout)
            r.raise_for_status()
            data = r.json()
    else:
        with open(source, "r", encoding="utf-8") as f:
            data = json.load(f)
    if isinstance(data, dict):
        for key in ("performers", "data", "items", "results"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError("JSON must contain a list of performer objects.")
    return data


def select_batch(performers: list, batch_index: int, batch_count: int) -> tuple[list, int, int, int, int]:
    """Split the JSON records into deterministic, balanced performer batches.

    The split is based ONLY on performer position in the JSON, so a performer
    can never be split across batches. Every image_url belonging to a selected
    performer is therefore processed in that same batch.
    """
    total_performers = len(performers)
    base, remainder = divmod(total_performers, batch_count)

    # First `remainder` batches get one extra performer. For 108,335 / 10,
    # batches 1-5 contain 10,834 performers and batches 6-10 contain 10,833.
    before = (batch_index - 1) * base + min(batch_index - 1, remainder)
    size = base + (1 if batch_index <= remainder else 0)
    after = before + size

    selected = performers[before:after]
    total_images_global = sum(len(p.get("image_urls") or []) for p in performers)
    images_in_batch = sum(len(p.get("image_urls") or []) for p in selected)
    return selected, total_performers, size, total_images_global, images_in_batch


def existing_results(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                out[(d["performer_id"], d["index"])] = d
            except Exception:
                continue
    return out


def download_one(session: requests.Session, task: ImageTask, timeout: int, min_bytes: int) -> ImageResult:
    task.dest_dir.mkdir(parents=True, exist_ok=True)

    for existing in task.dest_dir.glob(f"{task.index:03d}.*"):
        if existing.is_file() and existing.stat().st_size >= min_bytes:
            return ImageResult(
                task.performer_id, task.performer_name, task.url, task.index,
                True, str(existing), None, existing.stat().st_size
            )

    try:
        with session.get(task.url, timeout=timeout, stream=True) as r:
            r.raise_for_status()
            ctype = r.headers.get("Content-Type", "")
            if ctype and not ctype.lower().startswith("image"):
                return ImageResult(
                    task.performer_id, task.performer_name, task.url, task.index,
                    False, None, f"Unexpected Content-Type: {ctype}", 0
                )

            ext = guess_extension(task.url, ctype)
            dest = task.dest_dir / f"{task.index:03d}{ext}"
            tmp = dest.with_suffix(dest.suffix + ".part")
            total = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(65536):
                    if chunk:
                        f.write(chunk)
                        total += len(chunk)

            if total < min_bytes:
                tmp.unlink(missing_ok=True)
                return ImageResult(
                    task.performer_id, task.performer_name, task.url, task.index,
                    False, None, f"Too small: {total} bytes", total
                )

            for old in task.dest_dir.glob(f"{task.index:03d}.*"):
                if old != tmp:
                    old.unlink(missing_ok=True)
            tmp.rename(dest)

            return ImageResult(
                task.performer_id, task.performer_name, task.url, task.index,
                True, str(dest), None, total
            )
    except Exception as exc:
        return ImageResult(
            task.performer_id, task.performer_name, task.url, task.index,
            False, None, f"{type(exc).__name__}: {exc}", 0
        )


def write_reports(batch: int, report_dir: Path, results: list, selected: list,
                  total_performers_global: int, total_images_global: int,
                  images_in_batch: int, batch_count: int) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    ok = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "batch_index": batch,
        "batch_count": batch_count,
        "total_performers_global": total_performers_global,
        "performers_in_batch": len(selected),
        "total_images_global": total_images_global,
        "listed_images_in_batch": images_in_batch,
        "attempted_images": len(results),
        "download_success": len(ok),
        "download_failed": len(failed),
        "bytes_downloaded": sum(r.bytes_downloaded for r in results),
        "successful_images": [asdict(r) for r in ok],
        "failed_images": [asdict(r) for r in failed],
        "performer_ids": [str(p.get("id")) for p in selected],
    }
    json_path = report_dir / f"batch_{batch:02d}_download_report.json"
    md_path = report_dir / f"batch_{batch:02d}_download_report.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md = [
        f"# Batch {batch:02d} download report",
        f"- Global performers: **{total_performers_global}**",
        f"- Performers in batch: **{len(selected)}**",
        f"- Global listed images: **{total_images_global}**",
        f"- Listed images in batch: **{images_in_batch}**",
        f"- Successful: **{len(ok)}**",
        f"- Failed: **{len(failed)}**",
        f"- Bytes downloaded: **{report['bytes_downloaded']}**",
        "",
    ]
    if failed:
        md += ["## Failed images", "", "| Performer | Index | URL | Error |", "|---|---:|---|---|"]
        for r in failed:
            md.append(f"| {r.performer_name} | {r.index} | {r.url} | {r.error} |")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-source", required=True)
    ap.add_argument("--batch-index", type=int, required=True)
    ap.add_argument("--batch-count", type=int, default=10)
    ap.add_argument("--output-dir", default="./downloads")
    ap.add_argument("--report-dir", default="./reports")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=25)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--min-bytes", type=int, default=512)
    args = ap.parse_args()

    if not 1 <= args.batch_index <= args.batch_count:
        ap.error("--batch-index must be between 1 and --batch-count")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    performers = load_json(args.json_source, args.timeout)
    (selected, total_performers_global, performers_in_batch,
     total_images_global, images_in_batch) = select_batch(
        performers, args.batch_index, args.batch_count
    )
    LOG.info(
        "Batch %d/%d: %d performers | %d listed images (global %d)",
        args.batch_index, args.batch_count, performers_in_batch,
        images_in_batch, total_images_global
    )

    results_path = Path(args.report_dir) / f"batch_{args.batch_index:02d}_download_results.jsonl"
    previous = existing_results(results_path)
    tasks = []
    for p in selected:
        pid = str(p.get("id"))
        name = p.get("name") or "(no name)"
        urls = p.get("image_urls") or []
        dest_dir = Path(args.output_dir) / pid
        for idx, url in enumerate(urls, 1):
            if (pid, idx) in previous and previous[(pid, idx)].get("success"):
                try:
                    if Path(previous[(pid, idx)]["path"]).exists():
                        continue
                except Exception:
                    pass
            tasks.append(ImageTask(pid, name, url, idx, len(urls), dest_dir))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.report_dir).mkdir(parents=True, exist_ok=True)

    results = []
    with build_session(args.retries, 0.7) as session:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(download_one, session, t, args.timeout, args.min_bytes)
                for t in tasks
            ]
            iterator = as_completed(futures)
            if tqdm:
                iterator = tqdm(iterator, total=len(futures), desc=f"Batch {args.batch_index:02d}")
            with open(results_path, "a", encoding="utf-8") as jf:
                for fut in iterator:
                    r = fut.result()
                    results.append(r)
                    jf.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
                    jf.flush()

    # Include successful previous entries in the report.
    merged = list(previous.values()) + [asdict(r) for r in results]
    merged_map = {(x["performer_id"], x["index"]): x for x in merged}
    merged_results = [ImageResult(
        x["performer_id"], x["performer_name"], x["url"], x["index"],
        bool(x["success"]), x.get("path"), x.get("error"), int(x.get("bytes_downloaded", 0))
    ) for x in merged_map.values()]
    write_reports(
        args.batch_index, Path(args.report_dir), merged_results, selected,
        total_performers_global, total_images_global, images_in_batch, args.batch_count
    )

    failed = sum(1 for r in merged_results if not r.success)
    LOG.info("Batch %d complete: %d failed.", args.batch_index, failed)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
