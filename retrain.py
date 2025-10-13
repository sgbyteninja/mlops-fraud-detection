import time
import subprocess
from datetime import datetime
import boto3
import os
from dotenv import load_dotenv
from config import BUCKET_NAME, MONTHS_PREFIX, CHECK_INTERVAL, LATEST_MODEL_PATH
import mlflow
import mlflow.sklearn
import joblib
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from io import BytesIO
import json

# Load environment variables
load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-central-1")

# Set MLflow tracking URI from env or fallback to local ./mlruns
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if not mlflow_tracking_uri or mlflow_tracking_uri.startswith("s3://"):
    mlflow_tracking_uri = "file://" + os.path.abspath("./mlruns")
mlflow.set_tracking_uri(mlflow_tracking_uri)
print("MLflow tracking URI:", mlflow.get_tracking_uri())

# boto3 S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION
)

DRIFT_CHECK_SCRIPT = "drift_check.py"

def list_months():
    """List all CSV files under the months prefix in S3"""
    resp = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=MONTHS_PREFIX)
    return [obj['Key'] for obj in resp.get('Contents', []) if obj['Key'].endswith(".csv")]

def load_all_months():
    """Load all monthly CSVs from S3 and concatenate into a single DataFrame"""
    dfs = []
    for key in list_months():
        obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
        df = pd.read_csv(BytesIO(obj['Body'].read()))
        dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

def run_drift_check():
    """Run the drift_check.py script and return list of features with drift"""
    result = subprocess.run(["python", DRIFT_CHECK_SCRIPT], capture_output=True, text=True)
    output = result.stdout.strip()
    if "Drift detected" in output:
        features_str = output.split("Drift detected in features: ")[1]
        features = eval(features_str)
        return features
    return []

def upload_model_to_s3(model_file, metrics_file, input_example_file):
    """Upload model, metrics, and input example to S3 under a timestamped folder"""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M")
    s3_folder = f"model_backups/{timestamp}-Model"

    s3_client.upload_file(model_file, BUCKET_NAME, f"{s3_folder}/{os.path.basename(model_file)}")
    s3_client.upload_file(metrics_file, BUCKET_NAME, f"{s3_folder}/{os.path.basename(metrics_file)}")
    s3_client.upload_file(input_example_file, BUCKET_NAME, f"{s3_folder}/{os.path.basename(input_example_file)}")

    print(f"Model, metrics, and input example uploaded to s3://{BUCKET_NAME}/{s3_folder}")

def run_retraining(reason):
    """Retrain the model using all available months and upload model + metrics to S3"""
    print(f"Starting retraining due to: {reason}")
    
    df = load_all_months()
    if df.empty:
        print("No data available. Skipping retraining.")
        return
    
    X = df.drop("Class", axis=1)
    y = df["Class"]
    
    model = RandomForestClassifier(n_estimators=25, max_depth=8, min_samples_split=5, min_samples_leaf=3, max_features='sqrt', bootstrap=True, n_jobs=-1, random_state=42)


    # MLflow experiment
    mlflow.set_experiment("fraud_detection")
    with mlflow.start_run(run_name="retrain_local"):
        model.fit(X, y)

        # Save model locally
        os.makedirs(os.path.dirname(LATEST_MODEL_PATH), exist_ok=True)
        joblib.dump(model, LATEST_MODEL_PATH)

        # Log model and input example in MLflow
        mlflow.sklearn.log_model(model, "model", input_example=X.head(1))

        # Log training metrics
        train_accuracy = model.score(X, y)
        mlflow.log_metric("train_accuracy", train_accuracy)
        print(f"Training accuracy: {train_accuracy}")

        # Save metrics and input example for S3 upload
        metrics_file = "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump({"train_accuracy": train_accuracy}, f)
        input_example_file = "input_example.csv"
        X.head(1).to_csv(input_example_file, index=False)

        upload_model_to_s3(LATEST_MODEL_PATH, metrics_file, input_example_file)

    print("Retraining completed.")

if __name__ == "__main__":
    last_seen = set()

    while True:
        current_files = set(list_months())
        new_files = current_files - last_seen

        if new_files:
            print(f"New files detected: {new_files}")
            drift_features = run_drift_check()
            if drift_features:
                run_retraining("drift detected")
            else:
                print("No drift detected. No action.")
        else:
            print("No new files detected. No action.")

        last_seen = current_files
        time.sleep(CHECK_INTERVAL)
