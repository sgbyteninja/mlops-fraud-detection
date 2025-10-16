import time
import subprocess
from datetime import datetime
import boto3
import os
from dotenv import load_dotenv
from config import BUCKET_NAME, WEEKS_PREFIX, CHECK_INTERVAL

# Load environment variables
load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-central-1")

# boto3 S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION
)

RETRAIN_SCRIPT = "retrain.py"
DRIFT_CHECK_SCRIPT = "drift_check.py"

def list_weeks():
    """Return list of weekly CSV keys in S3 for drift checking."""
    resp = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=WEEKS_PREFIX)
    return [obj['Key'] for obj in resp.get('Contents', []) if obj['Key'].endswith(".csv")]

def run_drift_check():
    """Run the drift check script and return list of drifted features."""
    result = subprocess.run(["python", DRIFT_CHECK_SCRIPT], capture_output=True, text=True)
    output = result.stdout.strip()
    if "Drift detected" in output:
        features_str = output.split("Drift detected in features: ")[1]
        features = eval(features_str)
        return features
    return []

def run_retraining(reason):
    """Trigger retraining when drift is detected, logs to MLflow."""
    print(f"Starting retraining due to: {reason}")
    # MLflow setup to match train/retrain scripts
    import mlflow
    mlflow_path = os.path.join(os.getcwd(), "mlruns")
    os.makedirs(mlflow_path, exist_ok=True)
    MLFLOW_TRACKING_URI = f"file:{mlflow_path}"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("fraud_detection")

    # Run the retraining script
    subprocess.run(["python", RETRAIN_SCRIPT], capture_output=False, text=True)

if __name__ == "__main__":
    last_seen = set()

    while True:
        # Check for new weekly files
        current_files = set(list_weeks())
        new_files = current_files - last_seen

        if new_files:
            print(f"New weekly files detected: {new_files}")
            drift_features = run_drift_check()
            if drift_features:
                run_retraining("drift detected")
            else:
                print("No drift detected. Skipping retraining.")

        last_seen = current_files

        # Wait before checking again
        time.sleep(CHECK_INTERVAL)
