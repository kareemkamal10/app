#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
المرحلة 2: فحص وجود وجه في صور الأشخاص (مصمم للعمل على Kaggle GPU).

يعتمد على تقرير المرحلة الأولى (download_report.json) للحصول على قائمة
الأشخاص أصحاب "صورة واحدة" اللي نجح تحميلها فقط (success_list) — لأن
دول بالتحديد المطلوب فحصهم.

الموديل: YOLOv11l-face (akanametov/yolo-face) — يتحمّل تلقائياً من GitHub
لو مش موجود محلياً. اختير هذا الموديل لأنه:
    - Fine-tuned خصيصاً لكشف الوجوه (مش عام) على WIDERFace.
    - نسخة "l" (large) لأعلى دقة ممكنة، خصوصاً في الحالات الصعبة
      (وجه جزئي، زاوية غريبة، إضاءة سيئة).
    - مبني على معمارية Ultralytics القياسية، فهو خفيف الاستخدام
      ومناسب لأي GPU متاح على Kaggle.
    - إخراجه (bounding boxes) كافٍ لمرحلة "هل يوجد وجه أم لا"، وممكن
      لاحقاً تستخدم نفس الـ bbox كنقطة بداية لقص الوجه قبل تمريره
      لـ GhostFaceNetV2 لاستخراج المتجهات (embeddings).

ملاحظة مهمة للمستقبل (مرحلة الـ embeddings):
    GhostFaceNetV2 (زي أغلب شبكات ArcFace-family) بتديك أفضل دقة لو
    الوجه اتقص ومحاذي (aligned) باستخدام 5 نقاط دالة (عينين/أنف/فم)،
    مش مجرد bounding box. YOLO11-face هنا بيديك bbox بس بدون landmarks.
    لما توصل لمرحلة بناء المتجهات، فكّر تستخدم موديل زي RetinaFace أو
    SCRFD (من مكتبة insightface) للمحاذاة الدقيقة، أو دربّ/استخدم نسخة
    من YOLO-face بها pose/landmarks. الفحص الحالي (وجود وجه أم لا) لا
    يحتاج لمحاذاة، فـ YOLO11l-face مناسب تماماً للمرحلة دي.

الاستخدام على Kaggle (GPU):
    !pip install -q ultralytics
    !python 02_detect_faces.py \
        --download-report /kaggle/working/reports/download_report.json \
        --images-dir /kaggle/working/downloads \
        --report-dir /kaggle/working/reports \
        --device 0
"""

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

LOG = logging.getLogger("detect_faces")

WEIGHTS_URL = (
    "https://github.com/akanametov/yolo-face/releases/download/"
    "1.0.0/yolov11l-face.pt"
)


@dataclass
class DetectionOutcome:
    performer_id: str
    performer_name: str
    path: str
    face_found: bool
    num_faces: int
    best_confidence: Optional[float]
    pass_used: str  # "pass1" أو "pass2_fallback" أو "none"


def ensure_weights(weights_path: Path) -> Path:
    if weights_path.exists() and weights_path.stat().st_size > 1_000_000:
        LOG.info("استخدام الموديل الموجود بالفعل: %s", weights_path)
        return weights_path
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("تحميل موديل YOLOv11l-face من: %s", WEIGHTS_URL)
    urlretrieve(WEIGHTS_URL, str(weights_path))
    LOG.info("تم تحميل الموديل (%.1f MB)", weights_path.stat().st_size / 1e6)
    return weights_path


def load_single_image_success_list(download_report_path: Path) -> list:
    with open(download_report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    entries = report.get("single_image_performers", {}).get("success_list", [])
    if not entries:
        LOG.warning(
            "لا توجد عناصر في single_image_performers.success_list — "
            "تأكد إنك مشغّل تقرير المرحلة الأولى المحدّث."
        )
    return entries


def run_detection(model, entries: list, imgsz: int, conf: float,
                   fallback_imgsz: int, fallback_conf: float,
                   batch_size: int, device: str) -> list:
    outcomes: list[DetectionOutcome] = []

    # تجهيز قائمة المسارات الصالحة فقط (اللي فعلاً موجودة على القرص)
    valid_entries = []
    for e in entries:
        p = Path(e["path"]) if e.get("path") else None
        if p and p.exists():
            valid_entries.append(e)
        else:
            outcomes.append(DetectionOutcome(
                performer_id=e["id"], performer_name=e["name"],
                path=e.get("path") or "", face_found=False, num_faces=0,
                best_confidence=None, pass_used="missing_file",
            ))

    # --- الفحص الأساسي (Pass 1) على دفعات (batches) لسرعة أعلى على GPU ---
    # stream=True مهم مع آلاف/مئات الآلاف الصور: من غيرها ultralytics
    # بيحاول يجمّع نتايج الـ batch كله في الذاكرة قبل ما يرجعها، وده بيكبر
    # استهلاك الرام مع الوقت. مع stream=True النتايج بترجع generator
    # بيتفرّغ صورة-صورة أول بأول، فاستهلاك الرام بيفضل ثابت تقريباً بغض
    # النظر عن حجم الداتا الكلي.
    LOG.info("Pass 1: فحص %d صورة بـ imgsz=%d, conf=%.2f",
              len(valid_entries), imgsz, conf)
    needs_fallback = []
    for start in range(0, len(valid_entries), batch_size):
        batch = valid_entries[start:start + batch_size]
        paths = [e["path"] for e in batch]
        results = model.predict(
            paths, imgsz=imgsz, conf=conf, device=device,
            augment=True, verbose=False, stream=True,
        )
        for e, res in zip(batch, results):
            n = len(res.boxes)
            if n > 0:
                best_conf = float(res.boxes.conf.max())
                outcomes.append(DetectionOutcome(
                    performer_id=e["id"], performer_name=e["name"],
                    path=e["path"], face_found=True, num_faces=n,
                    best_confidence=best_conf, pass_used="pass1",
                ))
            else:
                needs_fallback.append(e)

    # --- محاولة ثانية أكثر تساهلاً للحالات الصعبة (وجه جزئي/زاوية صعبة) ---
    # اتحولت لدفعات (batches) زي Pass 1 بدل صورة-صورة: لو نسبة الحالات
    # الصعبة كبيرة (وارد جداً مع 300 ألف صورة)، المعالجة صورة-صورة كانت
    # هتبقى أبطأ بكتير من استغلال الـ GPU على دفعات.
    if needs_fallback:
        LOG.info("Pass 2 (fallback): إعادة محاولة %d صورة بـ imgsz=%d, conf=%.2f",
                  len(needs_fallback), fallback_imgsz, fallback_conf)
        for start in range(0, len(needs_fallback), batch_size):
            batch = needs_fallback[start:start + batch_size]
            paths = [e["path"] for e in batch]
            results = model.predict(
                paths, imgsz=fallback_imgsz, conf=fallback_conf,
                device=device, augment=True, verbose=False, stream=True,
            )
            for e, res in zip(batch, results):
                n = len(res.boxes)
                if n > 0:
                    best_conf = float(res.boxes.conf.max())
                    outcomes.append(DetectionOutcome(
                        performer_id=e["id"], performer_name=e["name"],
                        path=e["path"], face_found=True, num_faces=n,
                        best_confidence=best_conf, pass_used="pass2_fallback",
                    ))
                else:
                    outcomes.append(DetectionOutcome(
                        performer_id=e["id"], performer_name=e["name"],
                        path=e["path"], face_found=False, num_faces=0,
                        best_confidence=None, pass_used="none",
                    ))

    return outcomes


def main():
    parser = argparse.ArgumentParser(description="فحص وجود وجه في صور الأشخاص")
    parser.add_argument("--download-report", required=True,
                         help="مسار download_report.json الناتج من المرحلة الأولى")
    parser.add_argument("--images-dir", default=None,
                         help="غير مستخدم مباشرة (المسارات موجودة بالفعل في "
                              "التقرير) — موجود للتوافق فقط")
    parser.add_argument("--weights", default="./weights/yolov11l-face.pt",
                         help="مسار موديل YOLOv11l-face (هيتحمل تلقائياً لو مش موجود)")
    parser.add_argument("--report-dir", default="./reports")
    parser.add_argument("--device", default="cpu",
                         help="'0' لأول GPU على Kaggle، أو 'cpu'")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--fallback-imgsz", type=int, default=1600,
                         help="حجم أكبر للمحاولة الثانية (حالات صعبة)")
    parser.add_argument("--fallback-conf", type=float, default=0.10,
                         help="عتبة ثقة أقل للمحاولة الثانية (حالات صعبة)")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s | %(levelname)s | %(message)s")

    from ultralytics import YOLO  # استيراد هنا عشان --help يشتغل من غير الحزمة

    weights_path = ensure_weights(Path(args.weights))
    model = YOLO(str(weights_path))

    entries = load_single_image_success_list(Path(args.download_report))
    LOG.info("عدد الأشخاص أصحاب الصورة الواحدة (نجح تحميلهم) للفحص: %d",
              len(entries))

    outcomes = run_detection(
        model, entries,
        imgsz=args.imgsz, conf=args.conf,
        fallback_imgsz=args.fallback_imgsz, fallback_conf=args.fallback_conf,
        batch_size=args.batch_size, device=args.device,
    )

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    detected = [o for o in outcomes if o.face_found]
    not_detected = [o for o in outcomes if not o.face_found]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "yolov11l-face (akanametov/yolo-face)",
        "total_checked": len(outcomes),
        "face_detected": len(detected),
        "face_not_detected": len(not_detected),
        "not_detected_list": [
            {"id": o.performer_id, "name": o.performer_name, "path": o.path,
             "reason": "ملف الصورة غير موجود" if o.pass_used == "missing_file"
                       else "لم يتم اكتشاف وجه حتى بعد محاولة ثانية بعتبة أقل"}
            for o in not_detected
        ],
        "all_results": [
            {"id": o.performer_id, "name": o.performer_name, "path": o.path,
             "face_found": o.face_found, "num_faces": o.num_faces,
             "best_confidence": o.best_confidence, "pass_used": o.pass_used}
            for o in outcomes
        ],
    }

    report_json_path = report_dir / "face_detection_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md_lines = []
    md_lines.append("# تقرير فحص الوجوه (للأشخاص أصحاب الصورة الواحدة)\n")
    md_lines.append(f"- تاريخ التقرير: {report['generated_at']}\n")
    md_lines.append(f"- الموديل المستخدم: {report['model']}\n")
    md_lines.append(f"- إجمالي عدد الصور المفحوصة: **{report['total_checked']}**\n")
    md_lines.append(f"- تم اكتشاف وجه فيها: **{report['face_detected']}**\n")
    md_lines.append(f"- **لم يتم اكتشاف وجه فيها: {report['face_not_detected']}**\n")

    if report["not_detected_list"]:
        md_lines.append("\n## أسماء وIDs الأشخاص اللي لم يتم اكتشاف وجه في صورتهم\n")
        md_lines.append("| # | الاسم | ID | السبب |\n")
        md_lines.append("|---|-------|----|-------|\n")
        for i, item in enumerate(report["not_detected_list"], start=1):
            md_lines.append(f"| {i} | {item['name']} | {item['id']} | {item['reason']} |\n")

    report_md_path = report_dir / "face_detection_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.writelines(md_lines)

    LOG.info("تم الانتهاء. تم اكتشاف وجه في %d من %d | لم يُكتشف في %d",
              report["face_detected"], report["total_checked"],
              report["face_not_detected"])
    LOG.info("التقرير JSON: %s | التقرير Markdown: %s",
              report_json_path, report_md_path)


if __name__ == "__main__":
    main()
