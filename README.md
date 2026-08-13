# Batched performer image pipeline (Kaggle)

This version processes the performer dataset in **10 deterministic batches by performer/user record**, not by image count.

There are 108,335 performer records in the JSON source. The batch split keeps each performer intact: all of a performer's `image_urls` belong to the same batch, even when a performer has many images.

For every batch:

1. Select that batch's performer records from the JSON source.
2. Download **all `image_urls` for those performers**.
3. Store each performer's downloaded images under a directory named with that performer's `id`.
4. Write a batch download report.
5. Run YOLOv11l-face on the downloaded images.
6. Write a batch face-detection report.
7. Package the ORIGINAL downloaded image bytes into a normal Parquet file.
8. Upload the Parquet package and the face report to the Hugging Face Bucket.
9. Verify the remote files.
10. Update and re-upload the master checkpoint report.
11. Delete the local batch images and temporary Parquet package.
12. Continue to the next batch.

The Parquet files are **not encrypted**.

At the end, the master report (covering all batches) is left on the bucket at:
`final/download_master_report.json` / `final/download_master_report.md`.

## Secrets

Only one Kaggle Secret is required:

- `HF_TOKEN`

Do not commit the token to GitHub.

There is **no `PARQUET_PASSWORD`** and no encryption key.

## Run

Open `kaggle_notebook.ipynb` in a Kaggle Notebook with GPU and Internet enabled, fill in the config cell (repo URL, JSON source, bucket name, batch/worker counts), and run all cells.

`batch_pipeline.py` does the actual orchestration:

```bash
python batch_pipeline.py \
  --json-source "https://huggingface.co/datasets/abdelwahabnabil500/datafile/resolve/main/stashdb_performers_full.json" \
  --bucket "abdelwahabnabil500/faces" \
  --batch-count 10 \
  --workers 16 \
  --devices "0,1" \
  --batch-size 32 \
  --work-dir /kaggle/temp/pipeline_work
```

## Batch behavior

The split is performed by **performer records**, in their original JSON order.

With 108,335 performers and 10 batches:

- Batches 1-5 contain 10,834 performers each.
- Batches 6-10 contain 10,833 performers each.

A performer is never split across batches. If a performer has 1 image or 20 images, all of those images are processed in the same batch.

## Directory layout during processing

A batch is downloaded approximately like this:

```text
downloads/
├── <performer-id-1>/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
├── <performer-id-2>/
│   ├── 001.jpg
│   └── ...
└── ...
```

The Parquet rows contain the performer metadata needed by the packaging step together with the original image bytes.

## Why `/kaggle/temp`

The complete image collection is too large for `/kaggle/working`. The pipeline therefore uses `/kaggle/temp` as temporary storage and deletes each batch locally after successful upload and verification, so only one batch needs to exist on local disk at a time.

## Resume behavior

Kaggle sessions do not keep local disk between restarts, so resume state lives on the Hugging Face Bucket:

- After every completed batch, the pipeline uploads an updated `final/download_master_report.json`.
- On startup, that report is downloaded when available.
- A batch is skipped only when the checkpoint and/or the remote batch files confirm that it was completed.
- The remote completion check expects the final, **unencrypted** filename:
  `batches/batch_XX/batch_XX.parquet`.

If a previous checkpoint from the old encrypted version contains `.parquet.enc`, it should not be treated as a completed batch for the new unencrypted format.
