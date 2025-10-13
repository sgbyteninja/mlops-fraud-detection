import pandas as pd
from io import BytesIO
import boto3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import mlflow
import mlflow.sklearn
from config import BUCKET_NAME, MONTHS_PREFIX, LATEST_MODEL_PATH
from dotenv import load_dotenv
from datetime import datetime
import json

# Load environment variables
load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-central-1")

# Set MLflow tracking URI
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
if mlflow_tracking_uri.startswith("file://"):
    mlflow_tracking_uri = "file:" + mlflow_tracking_uri[7:]

# Make sure the folder exists
if mlflow_tracking_uri.startswith("file:"):
    local_path = mlflow_tracking_uri.replace("file:", "")
    os.makedirs(local_path, exist_ok=True)

mlflow.set_tracking_uri(mlflow_tracking_uri)
print("MLflow tracking URI:", mlflow_tracking_uri)

# boto3 S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION
)

def load_all_months():
    """Load all CSV files from S3 and concatenate into one DataFrame."""
    resp = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=MONTHS_PREFIX)
    dfs = []
    for obj in resp.get("Contents", []):
        if obj['Key'].endswith(".csv"):
            df = pd.read_csv(BytesIO(s3_client.get_object(Bucket=BUCKET_NAME, Key=obj['Key'])['Body'].read()))
            dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

def save_model_to_s3(model_file, metrics_file=None, input_example_file=None):
    """Upload model, metrics, and input example to S3 under timestamped folder."""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M")
    s3_folder = f"model_backups/{timestamp}-Model"

    s3_client.upload_file(model_file, BUCKET_NAME, f"{s3_folder}/{os.path.basename(model_file)}")
    if metrics_file:
        s3_client.upload_file(metrics_file, BUCKET_NAME, f"{s3_folder}/{os.path.basename(metrics_file)}")
    if input_example_file:
        s3_client.upload_file(input_example_file, BUCKET_NAME, f"{s3_folder}/{os.path.basename(input_example_file)}")

    print(f"Model, metrics, and input example uploaded to s3://{BUCKET_NAME}/{s3_folder}")

def main():
    df = load_all_months()
    if df.empty:
        print("No data available in S3. Exiting.")
        return
    print("Data loaded:", df.shape)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest
    model = RandomForestClassifier(n_estimators=25, max_depth=8, min_samples_split=5, min_samples_leaf=3, max_features='sqrt', bootstrap=True, n_jobs=-1, random_state=42)
    
    # MLflow logging
    mlflow.set_experiment("fraud_detection")
    with mlflow.start_run(run_name="train_local"):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Test Accuracy: {acc:.4f}")

        mlflow.log_metric("accuracy", acc)

        os.makedirs(os.path.dirname(LATEST_MODEL_PATH), exist_ok=True)
        joblib.dump(model, LATEST_MODEL_PATH)
        mlflow.sklearn.log_model(model, "model", input_example=X_train.head(1))

        metrics_file = "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump({"accuracy": acc}, f)
        input_example_file = "input_example.csv"
        X_train.head(1).to_csv(input_example_file, index=False)

        save_model_to_s3(LATEST_MODEL_PATH, metrics_file, input_example_file)

    print(f"Model saved locally at {LATEST_MODEL_PATH}, logged to MLflow, and backed up to S3.")

if __name__ == "__main__":
    main()
