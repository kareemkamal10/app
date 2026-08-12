#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Submit the complete ten-batch pipeline as one Lightning L4 Job."""

import argparse
import sys
import time

from lightning_sdk import Job, Machine, Studio


def stream_job(job):
    last = 0
    print(f"Job: {job.name}")
    print(f"Link: {job.link}")
    while True:
        try:
            logs = job.logs or ""
            if len(logs) > last:
                print(logs[last:], end="", flush=True)
                last = len(logs)
        except Exception:
            pass

        status = str(job.status)
        if status in {"Status.Completed", "Status.Failed", "Status.Stopped"}:
            print(f"\nFinal status: {status}")
            return status

        time.sleep(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teamspace", required=True)
    ap.add_argument("--org", required=True)
    ap.add_argument("--studio-name", default="performers-pipeline")
    ap.add_argument("--json-source", required=True)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--batch-count", type=int, default=10)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    studio = Studio(
        name=args.studio_name,
        teamspace=args.teamspace,
        org=args.org,
        create_ok=True,
    )
    # Start the Studio on its lowest default machine only to provide the
    # compute environment for the Job; the actual work is the L4 Job below.
    studio.start()

    studio.run_with_exit_code(
        "git pull --ff-only 2>/dev/null || true"
    )

    studio.run_with_exit_code(
        "python -m pip install -r requirements.txt"
    )

    command = (
        "python batch_pipeline.py "
        f"--json-source {args.json_source!r} "
        f"--bucket {args.bucket!r} "
        f"--batch-count {args.batch_count} "
        f"--workers {args.workers} "
        f"--batch-size {args.batch_size}"
    )

    # HF_TOKEN and PARQUET_PASSWORD are Teamspace secrets, so the Job gets
    # them automatically as environment variables at runtime.
    job = Job.run(
        command=command,
        name="performers-batched-pipeline",
        machine=Machine.L4,
        studio=studio,
    )

    status = stream_job(job)
    if status != "Status.Completed":
        sys.exit(1)


if __name__ == "__main__":
    main()
