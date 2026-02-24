# storage.py
try:
    from google.cloud import storage
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency 'google-cloud-storage'. "
        "Install it with: python3 -m pip install --user google-cloud-storage"
    ) from exc
from pathlib import Path
import argparse
from config import RAW_DATA_DIR, GCS_BUCKET, GCS_RAW_PREFIX

def upload_raw_to_gcs(local_dir=RAW_DATA_DIR, bucket_name=GCS_BUCKET, prefix=GCS_RAW_PREFIX):
    """Upload all raw parquet files to GCS Archive."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    local_path = Path(local_dir)
    for parquet_file in local_path.glob("**/*.parquet"):
        # Determine blob name: prefix + year-month/filename
        relative = parquet_file.relative_to(local_path)
        blob_name = f"{prefix}{relative}"
        blob = bucket.blob(blob_name)
        
        # Set storage class to ARCHIVE
        blob.storage_class = "ARCHIVE"
        
        print(f"Uploading {parquet_file} -> gs://{bucket_name}/{blob_name}")
        blob.upload_from_filename(str(parquet_file))
        
        # Optional: delete local after successful upload to save space
        # parquet_file.unlink()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", default=RAW_DATA_DIR)
    parser.add_argument("--bucket", default=GCS_BUCKET)
    parser.add_argument("--prefix", default=GCS_RAW_PREFIX)
    args = parser.parse_args()
    
    upload_raw_to_gcs(args.local_dir, args.bucket, args.prefix)
