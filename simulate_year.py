import boto3
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-central-1")
BUCKET_NAME = os.getenv("BUCKET_NAME")
WEEKS_PREFIX = os.getenv("WEEKS_PREFIX", "weekly_data")

# S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION
)

def upload_week_file(local_file, week_number):
    """Upload a week's CSV to S3."""
    s3_key = f"{WEEKS_PREFIX}/week_{week_number}.csv"
    s3_client.upload_file(local_file, BUCKET_NAME, s3_key)
    print(f"Uploaded {local_file} → s3://{BUCKET_NAME}/{s3_key}")

# Simulation of weekly uploads
NUM_WEEKS = 52
DATA_FOLDER = "dataset/weeks"

for week in range(1, NUM_WEEKS + 1):
    local_file = os.path.join(DATA_FOLDER, f"week_{week}.csv")
    if os.path.exists(local_file):
        upload_week_file(local_file, week)
        # wait some time for drift_watchdog to pick it up (for testing)
        time.sleep(60)  # 60s for demo; in real test could be longer
    else:
        print(f"File not found: {local_file}")

print("\n--- Yearly Weekly Data Simulation Completed ---")