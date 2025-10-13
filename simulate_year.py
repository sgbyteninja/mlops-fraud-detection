import boto3
import time
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-central-1")
BUCKET_NAME = os.getenv("BUCKET_NAME")
MONTHS_PREFIX = os.getenv("MONTHS_PREFIX", "dataset/months")

# S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION
)

def upload_month_file(local_file, month_number):
    """Upload a month CSV to S3."""
    s3_key = f"{MONTHS_PREFIX}/creditcard_month{month_number}.csv"
    s3_client.upload_file(local_file, BUCKET_NAME, s3_key)
    print(f"Uploaded {local_file} → s3://{BUCKET_NAME}/{s3_key}")

# Simulation
NUM_MONTHS = 12
for month in range(1, NUM_MONTHS + 1):
    local_file = f"dataset/months/creditcard_month{month}.csv"
    if os.path.exists(local_file):
        upload_month_file(local_file, month)
        time.sleep(420) #to make sure, drift_watchdog has enough time for retraining
    else:
        print(f"File not found: {local_file}")

print("\n--- Year Simulation Completed ---")