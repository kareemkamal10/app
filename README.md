# Batched performer image pipeline

This version processes the dataset in 10 deterministic batches.

For every batch:

1. Download original images.
2. Write a batch download report.
3. Run YOLOv11l-face.
4. Write a batch face-detection report.
5. Package the ORIGINAL downloaded image bytes into a Parquet file.
6. Encrypt the complete Parquet file with AES-256-GCM using a key derived from `PARQUET_PASSWORD`.
7. Upload the encrypted package and the batch face report to the Hugging Face Bucket.
8. Verify the remote files.
9. Delete local batch images and the temporary encrypted package.
10. Continue to the next batch.

At the end, upload the master download report.

## Secrets

Create these Teamspace secrets in Lightning:

- `HF_TOKEN`
- `PARQUET_PASSWORD`

Do not put either secret in GitHub.

## Run

From a local PowerShell terminal:

```powershell
$env:LIGHTNING_USER_ID="YOUR_USER_ID"
$env:LIGHTNING_API_KEY="YOUR_API_KEY"

python run_pipeline.py `
  --teamspace "deploy-model-project" `
  --org "kareemkamal500-org" `
  --json-source "https://huggingface.co/datasets/abdelwahabnabil500/datafile/resolve/main/stashdb_performers_full.json" `
  --bucket "abdelwahabnabil500/faces" `
  --studio-name "performers-pipeline" `
  --batch-count 10 `
  --workers 16 `
  --batch-size 32
```

The Studio provides the environment for the L4 Job. The work itself runs in the L4 Job.

## Important encryption note

The uploaded file has the suffix `.parquet.enc`. It contains a valid Parquet file encrypted as an AES-256-GCM envelope. The plaintext Parquet exists only temporarily on the worker and is deleted immediately after encryption.

The password is the same logical password for all batches, but each batch derives a distinct encryption key using PBKDF2-HMAC-SHA256 with a batch-specific salt.

## Resume behavior

Each successfully uploaded batch creates:

`pipeline_work/state/batch_XX.uploaded.ok`

If the Job is restarted in the same persistent Studio, completed batches are skipped.

A failed upload must complete successfully before its local images are deleted.
