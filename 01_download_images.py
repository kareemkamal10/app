#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
المرحلة 1: تحميل صور الأشخاص من ملف JSON (بصيغة stashdb performers) وتنظيمها
في مجلدات باسم id الخاص بكل شخص، ثم إعداد تقرير تحميل مفصّل.

الاستخدام:
    python 01_download_images.py \
        --json-source "https://huggingface.co/datasets/.../stashdb_performers_full.json" \
        --output-dir ./downloads \
        --report-dir ./reports \
        --workers 16

يدعم:
    - إعادة المحاولة (retries) مع backoff عند فشل التحميل.
    - الاستئناف (resume): لو الصورة اتحملت قبل كده بنجاح، بيتم تخطيها.
    - كشف امتداد الصورة تلقائياً من Content-Type لو الرابط مالوش امتداد واضح.
    - تقرير JSON + تقرير Markdown بالعربي يوضح بالتحديد:
        * عدد الأشخاص اللي عندهم صورة واحدة فقط في image_urls
        * كام واحد منهم فشل تحميل صورته
        * أسماء وIDs هؤلاء الأشخاص
      بالإضافة لإحصائيات عامة عن باقي الحالات (0 صور / أكتر من صورة).
    - كتابة تدريجية (streaming) لنتيجة كل صورة في ملف
      reports/download_results.jsonl أول ما تخلص (مش بعد ما كل حاجة
      تخلص). ده معناه لو السكربت اتقفل فجأة (كهرباء/كراش) في نص تحميل
      300 ألف صورة، النتائج اللي خلصت فعلاً متسجلة على القرص، وتشغيل
      السكربت تاني هيتخطى تلقائياً أي صورة اتسجلت قبل كده كـ "نجحت"
      (فوق الاستئناف الموجود أصلاً اللي بيعتمد على وجود الملف نفسه).
"""

import argparse
import json
import logging
import mimetypes
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from tqdm import tqdm
except ImportError:  # tqdm اختياري، لو مش موجود هنكمل من غيره
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []


LOG = logging.getLogger("download_images")

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}


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


def build_session(retries: int = 3, backoff: float = 0.7) -> requests.Session:
    session = requests.Session()
    retry_cfg = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_cfg, pool_maxsize=64, pool_connections=64)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(DEFAULT_HEADERS)
    return session


def guess_extension(url: str, content_type: Optional[str]) -> str:
    # 1) جرب من امتداد الرابط نفسه
    url_path = url.split("?")[0]
    suffix = Path(url_path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}:
        return ".jpg" if suffix == ".jpeg" else suffix
    # 2) جرب من الـ Content-Type
    if content_type:
        ext = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if ext:
            return ".jpg" if ext == ".jpe" else ext
    # 3) افتراضي
    return ".jpg"


def load_performers(json_source: str, timeout: int) -> list:
    LOG.info("تحميل ملف الـ JSON من: %s", json_source)
    if json_source.startswith("http://") or json_source.startswith("https://"):
        session = build_session()
        resp = session.get(json_source, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    else:
        with open(json_source, "r", encoding="utf-8") as f:
            data = json.load(f)

    if isinstance(data, dict):
        # بعض الملفات بتكون {"performers": [...]}
        for key in ("performers", "data", "items", "results"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError("شكل ملف الـ JSON غير متوقع؛ متوقع list من الأشخاص.")
    LOG.info("تم تحميل %d عنصر.", len(data))
    return data


def download_one(session: requests.Session, task: ImageTask, timeout: int,
                  min_bytes: int, force: bool) -> ImageResult:
    task.dest_dir.mkdir(parents=True, exist_ok=True)
    # لو الملف موجود بالفعل بحجم معقول ومش --force، اعتبره نجح (استئناف)
    existing = list(task.dest_dir.glob(f"{task.index:03d}.*"))
    if existing and not force and existing[0].stat().st_size >= min_bytes:
        return ImageResult(
            performer_id=task.performer_id,
            performer_name=task.performer_name,
            url=task.url,
            index=task.index,
            success=True,
            path=str(existing[0]),
            bytes_downloaded=existing[0].stat().st_size,
        )

    try:
        resp = session.get(task.url, timeout=timeout, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if content_type and not content_type.lower().startswith("image"):
            return ImageResult(
                performer_id=task.performer_id,
                performer_name=task.performer_name,
                url=task.url,
                index=task.index,
                success=False,
                error=f"Content-Type غير متوقع: {content_type}",
            )

        ext = guess_extension(task.url, content_type)
        dest_path = task.dest_dir / f"{task.index:03d}{ext}"
        tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

        total_bytes = 0
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    total_bytes += len(chunk)

        if total_bytes < min_bytes:
            tmp_path.unlink(missing_ok=True)
            return ImageResult(
                performer_id=task.performer_id,
                performer_name=task.performer_name,
                url=task.url,
                index=task.index,
                success=False,
                error=f"حجم الملف صغير جداً ({total_bytes} bytes) - على الأرجح فشل/فارغ",
            )

        # امسح أي نسخة قديمة بامتداد مختلف
        for old in task.dest_dir.glob(f"{task.index:03d}.*"):
            if old != tmp_path:
                old.unlink(missing_ok=True)
        tmp_path.rename(dest_path)

        return ImageResult(
            performer_id=task.performer_id,
            performer_name=task.performer_name,
            url=task.url,
            index=task.index,
            success=True,
            path=str(dest_path),
            bytes_downloaded=total_bytes,
        )
    except Exception as exc:  # noqa: BLE001
        return ImageResult(
            performer_id=task.performer_id,
            performer_name=task.performer_name,
            url=task.url,
            index=task.index,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def load_existing_results(jsonl_path: Path) -> dict:
    """تحميل نتائج تشغيلة سابقة (لو موجودة) من ملف JSONL.

    ده اللي بيدي الحماية الحقيقية ضد الكراش/قطع الكهرباء في نص تحميل
    300 ألف صورة: بدل ما التقرير يتبني من الرام بس (وده بيضيع لو
    السكربت اتقفل فجأة)، بنقرأ كل سطر اتسجل فعلاً على القرص من التشغيلة
    اللي فاتت، ونستخدمه كنقطة بداية.
    """
    existing: dict = {}
    if not jsonl_path.exists():
        return existing
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue  # سطر ناقص/متقطع (ممكن يحصل لو الكراش وقع أثناء الكتابة)
            existing[(d["performer_id"], d["index"])] = d
    LOG.info("تم تحميل %d نتيجة من تشغيلة سابقة (%s) للاستئناف الآمن.",
              len(existing), jsonl_path.name)
    return existing


def main():
    parser = argparse.ArgumentParser(description="تحميل صور الأشخاص من ملف JSON")
    parser.add_argument("--json-source", required=True,
                         help="رابط أو مسار ملف JSON")
    parser.add_argument("--output-dir", default="./downloads",
                         help="مجلد حفظ الصور (هيتعمل جواه مجلد باسم كل id)")
    parser.add_argument("--report-dir", default="./reports",
                         help="مجلد حفظ التقارير")
    parser.add_argument("--workers", type=int, default=16,
                         help="عدد الخيوط المتوازية للتحميل")
    parser.add_argument("--timeout", type=int, default=25,
                         help="مهلة الاتصال بالثواني لكل صورة")
    parser.add_argument("--retries", type=int, default=3,
                         help="عدد إعادة المحاولات لكل صورة عند فشل الشبكة")
    parser.add_argument("--min-bytes", type=int, default=512,
                         help="أقل حجم مقبول للصورة عشان تُعتبر نجحت (بايت)")
    parser.add_argument("--force", action="store_true",
                         help="إعادة تحميل الصور حتى لو موجودة بالفعل")
    parser.add_argument("--limit", type=int, default=None,
                         help="لو عايز تجرب على أول N شخص فقط (اختياري)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    performers = load_performers(args.json_source, timeout=args.timeout)
    if args.limit:
        performers = performers[: args.limit]

    results_jsonl_path = report_dir / "download_results.jsonl"
    existing_results = {} if args.force else load_existing_results(results_jsonl_path)

    # توزيع عدد الصور لكل شخص
    image_count_distribution = {}
    tasks: list[ImageTask] = []
    performers_index = {}  # id -> {"name":.., "image_urls":[...]}
    total_images_listed = 0
    skipped_already_done = 0

    for p in performers:
        pid = p.get("id")
        name = p.get("name") or "(بدون اسم)"
        urls = p.get("image_urls") or []
        n = len(urls)
        image_count_distribution[n] = image_count_distribution.get(n, 0) + 1
        performers_index[pid] = {"name": name, "image_urls": urls}

        dest_dir = output_dir / str(pid)
        for i, url in enumerate(urls, start=1):
            total_images_listed += 1
            prev = existing_results.get((pid, i))
            # لو الصورة دي نجحت في تشغيلة سابقة وملفها لسه موجود فعلاً على
            # القرص، منعيدش تحميلها تاني (استئناف آمن بعد كراش/قطع اتصال)
            if (prev and prev.get("success") and prev.get("path")
                    and Path(prev["path"]).exists()):
                skipped_already_done += 1
                continue
            tasks.append(ImageTask(
                performer_id=pid,
                performer_name=name,
                url=url,
                index=i,
                total_for_performer=n,
                dest_dir=dest_dir,
            ))

    if skipped_already_done:
        LOG.info("تخطي %d صورة نجح تحميلها في تشغيلة سابقة (استئناف).",
                  skipped_already_done)
    LOG.info("إجمالي الأشخاص: %d | إجمالي الصور المطلوب تحميلها: %d | "
              "المتبقي فعلياً في التشغيلة دي: %d",
              len(performers), total_images_listed, len(tasks))

    session = build_session(retries=args.retries)

    # الكتابة على ملف JSONL أول بأول (append، أو من الصفر لو --force) عشان
    # النتائج تفضل محفوظة على القرص حتى لو السكربت اتقفل فجأة قبل ما
    # التقرير النهائي يتكتب.
    jsonl_mode = "w" if args.force else "a"
    all_results_map: dict = {} if args.force else dict(existing_results)

    with open(results_jsonl_path, jsonl_mode, encoding="utf-8") as jsonl_file:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(download_one, session, t, args.timeout,
                                 args.min_bytes, args.force): t
                for t in tasks
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="تحميل الصور"):
                r = fut.result()
                r_dict = asdict(r)
                jsonl_file.write(json.dumps(r_dict, ensure_ascii=False) + "\n")
                jsonl_file.flush()
                all_results_map[(r.performer_id, r.index)] = r_dict

    # ---- بناء التقرير (من كل النتائج المعروفة: القديمة + الجديدة) ----
    results_by_performer: dict[str, list[dict]] = {}
    for d in all_results_map.values():
        results_by_performer.setdefault(d["performer_id"], []).append(d)

    total_images_success = sum(1 for d in all_results_map.values() if d["success"])
    total_images_failed = total_images_listed - total_images_success

    single_image_total = 0
    single_image_success = 0
    single_image_failed = 0
    single_image_failed_list = []
    single_image_success_list = []

    for pid, info in performers_index.items():
        urls = info["image_urls"]
        if len(urls) != 1:
            continue
        single_image_total += 1
        perf_results = results_by_performer.get(pid, [])
        ok_result = next((r for r in perf_results if r["success"]), None)
        if ok_result:
            single_image_success += 1
            single_image_success_list.append({
                "id": pid,
                "name": info["name"],
                "url": urls[0],
                "path": ok_result["path"],
            })
        else:
            single_image_failed += 1
            err = perf_results[0]["error"] if perf_results else "لم تتم محاولة التحميل"
            single_image_failed_list.append({
                "id": pid,
                "name": info["name"],
                "url": urls[0],
                "error": err,
            })

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "json_source": args.json_source,
        "total_performers": len(performers),
        "image_count_distribution": {
            str(k): v for k, v in sorted(image_count_distribution.items())
        },
        "overall_download_stats": {
            "total_images_listed": total_images_listed,
            "total_images_downloaded_success": total_images_success,
            "total_images_downloaded_failed": total_images_failed,
        },
        "single_image_performers": {
            "total": single_image_total,
            "downloaded_success": single_image_success,
            "downloaded_failed": single_image_failed,
            "failed_list": single_image_failed_list,
            "success_list": single_image_success_list,
        },
    }

    report_json_path = report_dir / "download_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # تقرير Markdown بالعربي
    md_lines = []
    md_lines.append("# تقرير تحميل الصور\n")
    md_lines.append(f"- تاريخ التقرير: {report['generated_at']}\n")
    md_lines.append(f"- إجمالي عدد الأشخاص في الملف: **{report['total_performers']}**\n")
    md_lines.append("\n## توزيع عدد الصور لكل شخص\n")
    for k, v in report["image_count_distribution"].items():
        md_lines.append(f"- عدد الأشخاص اللي عندهم **{k}** صورة: {v}\n")

    md_lines.append("\n## الأشخاص أصحاب صورة واحدة فقط (image_urls بها عنصر واحد)\n")
    s = report["single_image_performers"]
    md_lines.append(f"- إجمالي عددهم: **{s['total']}**\n")
    md_lines.append(f"- تم تحميل صورتهم بنجاح: **{s['downloaded_success']}**\n")
    md_lines.append(f"- **فشل تحميل صورتهم: {s['downloaded_failed']}**\n")
    if s["failed_list"]:
        md_lines.append("\n### أسماء وIDs الأشخاص اللي فشل تحميل صورتهم (صورة واحدة فقط)\n")
        md_lines.append("| # | الاسم | ID | سبب الفشل |\n")
        md_lines.append("|---|-------|----|-----------|\n")
        for i, item in enumerate(s["failed_list"], start=1):
            md_lines.append(f"| {i} | {item['name']} | {item['id']} | {item['error']} |\n")

    md_lines.append("\n## إحصائيات عامة عن التحميل\n")
    o = report["overall_download_stats"]
    md_lines.append(f"- إجمالي الصور المطلوب تحميلها (كل الأشخاص): {o['total_images_listed']}\n")
    md_lines.append(f"- نجح تحميلها: {o['total_images_downloaded_success']}\n")
    md_lines.append(f"- فشل تحميلها: {o['total_images_downloaded_failed']}\n")

    report_md_path = report_dir / "download_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.writelines(md_lines)

    LOG.info("تم الانتهاء. التقرير JSON: %s | التقرير Markdown: %s",
             report_json_path, report_md_path)
    LOG.info("أصحاب الصورة الواحدة: %d | نجح: %d | فشل: %d",
             s["total"], s["downloaded_success"], s["downloaded_failed"])


if __name__ == "__main__":
    main()
