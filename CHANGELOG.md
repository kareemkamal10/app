# Changelog

All notable changes to this project are documented in this file.

## [2026-08-13]

### Changed

- Changed batch partitioning from **image-based batches** to **performer/user-based batches**.
- The JSON dataset is now divided into **10 deterministic batches by performer records**.
- Each performer remains completely inside one batch; all of that performer's `image_urls` are processed together.
- The batch downloader therefore supports performers with one image or many images without splitting their images across batches.
- For the current dataset of **108,335 performers**:
  - Batches 1-5 contain 10,834 performers each.
  - Batches 6-10 contain 10,833 performers each.
- Kept the performer directory structure based on the performer `id`, with all downloaded images stored inside that performer's directory.

### Removed

- Removed Parquet encryption completely.
- Removed the AES-256-GCM encryption layer.
- Removed the `encrypt_file` function.
- Removed the PBKDF2 key derivation used for encrypted Parquet files.
- Removed the `PARQUET_PASSWORD` requirement.
- Removed the `cryptography` dependency from `requirements.txt`.
- Replaced the encrypted `.parquet.enc` output with normal `.parquet` files.

### Packaging and Upload

- The pipeline now writes the downloaded original image bytes directly to a normal Parquet file.
- The resulting Parquet file is uploaded without encryption.
- Remote batch completion checks now use the `.parquet` filename.
- Resume handling was aligned with the new unencrypted package format so old `.parquet.enc` checkpoint entries are not treated as completed new-format batches.

### Pipeline / Execution

- Updated the pipeline documentation to describe the new performer-based batching model.
- Updated the Kaggle notebook configuration to require only `HF_TOKEN`.
- Updated `run_pipeline.py` to accept a configurable `--devices` argument.
- `run_pipeline.py` now passes the selected GPU device IDs through to `batch_pipeline.py`.
- Default GPU configuration is `--devices "0,1"`.

### Documentation

- Updated the README to document:
  - Performer-based batching.
  - The 108,335-performer dataset split.
  - Per-performer image directories.
  - Unencrypted Parquet output.
  - The single required `HF_TOKEN` secret.
  - Updated resume behavior.
  - GPU device configuration.

## Notes

The intended final processing flow is:

JSON performer records
→ split performers into 10 batches
→ download all `image_urls` for each performer
→ run YOLOv11l-face
→ create normal Parquet
→ upload Parquet + face report
→ verify remote upload
→ update checkpoint
→ delete local batch
→ continue to the next batch

### Long-run reliability hardening

- Fixed HTTP retry configuration so `Retry(...)` is actually attached to the `HTTPAdapter`.
- Added per-image download retries with exponential backoff.
- A small number of permanently failed image URLs no longer aborts an otherwise successful batch; failures remain recorded and are retried automatically if the same batch is run again.
- Cache the performer JSON locally once per pipeline run so later batches do not depend on repeated Hugging Face downloads.
- Changed Parquet packaging to chunked `ParquetWriter` output to avoid loading an entire multi-GB batch into RAM.
- Added atomic temporary Parquet output so a failed package cannot leave a misleading partial `.parquet` file.
- Added retry and size-aware verification to Bucket uploads; already matching remote files are not re-uploaded.
- Made master checkpoint uploads use the Hugging Face Bucket API with retries and verification.
- Master checkpoint download/resume now uses the Bucket API instead of the Dataset API.
- The local batch-completion marker is written only after the master checkpoint has been uploaded successfully.
- Added recovery for a remotely completed batch whose checkpoint record was not written before a prior process stopped.
