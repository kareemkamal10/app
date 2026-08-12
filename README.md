# Batched performer image pipeline (Kaggle)

This version processes the dataset in 10 deterministic batches, meant to run inside a Kaggle notebook (`kaggle_notebook.ipynb`).

For every batch:

1. Download original images.
2. Write a batch download report.
3. Run YOLOv11l-face.
4. Write a batch face-detection report.
5. Package the ORIGINAL downloaded image bytes into a Parquet file.
6. Encrypt the complete Parquet file with AES-256-GCM using a key derived from `PARQUET_PASSWORD`.
7. Upload the encrypted package and the batch face report to the Hugging Face Bucket.
8. Verify the remote files.
9. Update and re-upload the master checkpoint report.
10. Delete local batch images and the temporary encrypted package.
11. Continue to the next batch.

At the end, the master report (covering all batches) is left on the bucket at `final/download_master_report.json` / `.md`.

## Secrets

Set these as Kaggle Secrets (Add-ons -> Secrets) so they aren't hardcoded in the notebook:

- `HF_TOKEN`
- `PARQUET_PASSWORD`

Do not commit either secret to GitHub.

## Run

Open `kaggle_notebook.ipynb` in a Kaggle Notebook with GPU and Internet enabled, fill in the config cell (repo URL, JSON source, bucket name, batch/worker counts), and run all cells. `batch_pipeline.py` does the actual orchestration:

```bash
python batch_pipeline.py \
  --json-source "https://huggingface.co/datasets/abdelwahabnabil500/datafile/resolve/main/stashdb_performers_full.json" \
  --bucket "abdelwahabnabil500/faces" \
  --batch-count 10 \
  --workers 16 \
  --batch-size 32 \
  --work-dir /kaggle/temp/pipeline_work
```

## Why `/kaggle/temp`

The full dataset is close to 80GB, which doesn't fit in `/kaggle/working` (~20GB or less). `/kaggle/temp` gives ~50GB and doesn't count against the persistent output quota, which is fine here because each batch's local files are deleted immediately after it's uploaded and verified — only one batch's worth of images needs to be on disk at a time.

## Important encryption note

The uploaded file has the suffix `.parquet.enc`. It contains a valid Parquet file encrypted as an AES-256-GCM envelope. The plaintext Parquet exists only temporarily on the worker and is deleted immediately after encryption.

The password is the same logical password for all batches, but each batch derives a distinct encryption key using PBKDF2-HMAC-SHA256 with a batch-specific salt.

## Resume behavior

Kaggle sessions don't keep local disk between restarts, so resume state lives on the bucket, not on disk:

- After every batch, the pipeline uploads an updated `final/download_master_report.json` to the bucket.
- On startup, it downloads that file (if present) and skips any batch already recorded in it, or already found on the bucket under `batches/batch_XX/`.
- If the pipeline (or the whole notebook) is interrupted and restarted, re-running `batch_pipeline.py` with the same `--json-source` picks up right after the last successfully uploaded batch.
