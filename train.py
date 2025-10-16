import os
import pandas as pd
from io import BytesIO
import boto3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from datetime import datetime
import json
from config import BUCKET_NAME, WEEKS_PREFIX, LATEST_MODEL_PATH, NUM_WEEKS_FOR_TRAINING

# Load environment variables
load_dotenv()

s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION", "eu-central-1")
)

EXPERIMENT_NAME = "fraud_detection"
mlflow.set_experiment(EXPERIMENT_NAME)

def load_last_weeks():
    resp = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=WEEKS_PREFIX)
    files = sorted([obj['Key'] for obj in resp.get("Contents", []) if obj['Key'].endswith(".csv")])
    if not files:
        return pd.DataFrame()
    files_to_load = files[-NUM_WEEKS_FOR_TRAINING:]
    dfs = [pd.read_csv(BytesIO(s3_client.get_object(Bucket=BUCKET_NAME, Key=key)['Body'].read())) for key in files_to_load]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def save_model_to_s3(model_file, metrics_file=None, input_example_file=None):
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M")
    s3_folder = f"model_backups/{timestamp}-Model"
    for file in [model_file, metrics_file, input_example_file]:
        if file and os.path.exists(file):
            s3_client.upload_file(file, BUCKET_NAME, f"{s3_folder}/{os.path.basename(file)}")
    print(f"Model, metrics, and input example uploaded to s3://{BUCKET_NAME}/{s3_folder}")

def main():
    df = load_last_weeks()
    if df.empty:
        print(f"No data available in the last {NUM_WEEKS_FOR_TRAINING} weeks. Exiting.")
        return
    print(f"Data loaded from last {NUM_WEEKS_FOR_TRAINING} weeks: {df.shape}")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(
        n_estimators=25, max_depth=8, min_samples_split=5,
        min_samples_leaf=3, max_features='sqrt', bootstrap=True,
        n_jobs=-1, random_state=42
    )

    with mlflow.start_run(run_name="train_local"):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Compute metrics
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        print(f"Test Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Save model locally
        os.makedirs(os.path.dirname(LATEST_MODEL_PATH), exist_ok=True)
        joblib.dump(model, LATEST_MODEL_PATH)

        mlflow.sklearn.log_model(model, "model", input_example=X_train.head(1))

        metrics_file = "metrics.json"
        input_example_file = "input_example.csv"
        with open(metrics_file, "w") as f:
            json.dump({"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}, f)
        X_train.head(1).to_csv(input_example_file, index=False)

        save_model_to_s3(LATEST_MODEL_PATH, metrics_file, input_example_file)

    print(f"Model saved locally at {LATEST_MODEL_PATH}, logged to MLflow, and backed up to S3.")

if __name__ == "__main__":
    main()
