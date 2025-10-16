import os
import pandas as pd
from io import BytesIO
import boto3
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import mlflow
import mlflow.sklearn
import json
from dotenv import load_dotenv
from datetime import datetime
from config import BUCKET_NAME, WEEKS_PREFIX, LATEST_MODEL_PATH

load_dotenv()

s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION", "eu-central-1")
)

EXPERIMENT_NAME = "fraud_detection"
mlflow.set_experiment(EXPERIMENT_NAME)

def load_last_n_weeks(n=4):
    resp = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=WEEKS_PREFIX)
    files = sorted([obj['Key'] for obj in resp.get("Contents", []) if obj['Key'].endswith(".csv")])[-n:]
    dfs = [pd.read_csv(BytesIO(s3_client.get_object(Bucket=BUCKET_NAME, Key=key)['Body'].read())) for key in files]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def upload_model_to_s3(model_file, metrics_file, input_example_file):
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M")
    s3_folder = f"model_backups/{timestamp}-Model"
    for file in [model_file, metrics_file, input_example_file]:
        if os.path.exists(file):
            s3_client.upload_file(file, BUCKET_NAME, f"{s3_folder}/{os.path.basename(file)}")
    print(f"Model, metrics, and input example uploaded to s3://{BUCKET_NAME}/{s3_folder}")

def run_retraining():
    df = load_last_n_weeks(4)
    if df.empty or "Class" not in df.columns:
        print("No data or target column missing. Aborting retraining.")
        return

    X = df.drop("Class", axis=1)
    y = df["Class"]

    model = RandomForestClassifier(
        n_estimators=25, max_depth=8, min_samples_split=5,
        min_samples_leaf=3, max_features='sqrt', bootstrap=True,
        n_jobs=-1, random_state=42
    )

    with mlflow.start_run(run_name="retrain_local"):
        model.fit(X, y)
        preds = model.predict(X)

        # Compute metrics
        acc = accuracy_score(y, preds)
        prec = precision_score(y, preds)
        rec = recall_score(y, preds)
        f1 = f1_score(y, preds)
        print(f"Training Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        os.makedirs(os.path.dirname(LATEST_MODEL_PATH), exist_ok=True)
        joblib.dump(model, LATEST_MODEL_PATH)

        mlflow.sklearn.log_model(model, "model", input_example=X.head(1))

        metrics_file = "metrics.json"
        input_example_file = "input_example.csv"
        with open(metrics_file, "w") as f:
            json.dump({"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}, f)
        X.head(1).to_csv(input_example_file, index=False)

        upload_model_to_s3(LATEST_MODEL_PATH, metrics_file, input_example_file)

    print("Retraining completed.")

if __name__ == "__main__":
    run_retraining()
