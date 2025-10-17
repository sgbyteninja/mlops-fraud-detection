from flask import Flask, request, jsonify
from functools import wraps
import pandas as pd
import logging
import joblib
from io import BytesIO
import boto3
import os
from dotenv import load_dotenv
import threading
import time
from datetime import datetime

# Load environment variables
load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-central-1")
BUCKET_NAME = os.getenv("BUCKET_NAME", "fraud-detection-project-data-science")
MODEL_BACKUPS_PREFIX = os.getenv("MODEL_BACKUPS_PREFIX", "model_backups")
MODEL_RELOAD_INTERVAL = int(os.getenv("MODEL_RELOAD_INTERVAL", 300))
API_TOKEN = os.getenv("API_TOKEN")
WEEKLY_UPLOAD_PREFIX = os.getenv("WEEKLY_UPLOAD_PREFIX", "weekly_data")

# Flask App
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# boto3 S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION
)

# Authentification Decorator
def require_api_token(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization")
        if not token or token.replace("Bearer ", "") != API_TOKEN:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated

# Global model variables
model = None
latest_model_key = None
lock = threading.Lock()

# Model handling
def get_latest_model_key():
    resp = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=MODEL_BACKUPS_PREFIX)
    keys = [obj['Key'] for obj in resp.get('Contents', []) if obj['Key'].endswith("model.pkl")]
    if not keys:
        raise FileNotFoundError("No model found in S3.")
    return sorted(keys, reverse=True)[0]

def load_model_from_s3(key):
    obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
    m = joblib.load(BytesIO(obj['Body'].read()))
    logging.info(f"Model loaded from s3://{BUCKET_NAME}/{key}")
    return m

def reload_model_periodically():
    global model, latest_model_key
    while True:
        try:
            key = get_latest_model_key()
            if key != latest_model_key:
                logging.info(f"New model detected: {key}. Reloading...")
                new_model = load_model_from_s3(key)
                with lock:
                    model = new_model
                    latest_model_key = key
        except Exception as e:
            logging.error(f"Error loading model: {e}")
        time.sleep(MODEL_RELOAD_INTERVAL)

# Request logging
REQUEST_LOG_PATH = "requests_log.csv"

def log_request_to_csv(df, preds_proba, threshold=0.5):
    """Append incoming requests + predicted class to local CSV."""
    try:
        df_to_log = df.copy()
        # add Class column to the df
        df_to_log["Class"] = (preds_proba >= threshold).astype(int)

        if os.path.exists(REQUEST_LOG_PATH):
            df_to_log.to_csv(REQUEST_LOG_PATH, mode="a", header=False, index=False)
        else:
            df_to_log.to_csv(REQUEST_LOG_PATH, mode="w", header=True, index=False)
    except Exception as e:
        logging.error(f"Error logging request: {e}")

def upload_weekly_data():
    """Upload collected requests to S3 once per week."""
    while True:
        try:
            now = datetime.utcnow()
            year, week_num, _ = now.isocalendar()
            s3_key = f"{WEEKLY_UPLOAD_PREFIX}/week_{year}_{week_num}.csv"

            if os.path.exists(REQUEST_LOG_PATH):
                logging.info(f"Uploading weekly data to s3://{BUCKET_NAME}/{s3_key}")
                s3_client.upload_file(REQUEST_LOG_PATH, BUCKET_NAME, s3_key)
                os.remove(REQUEST_LOG_PATH)  # clear local log after upload
            else:
                logging.info("No new data to upload this week.")

        except Exception as e:
            logging.error(f"Error uploading weekly data: {e}")

        # Sleep for one week (604800 seconds)
        time.sleep(7 * 24 * 60 * 60)

# Background Threads
threading.Thread(target=reload_model_periodically, daemon=True).start()
threading.Thread(target=upload_weekly_data, daemon=True).start()

# Routes
@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
@require_api_token
def predict():
    try:
        data = request.get_json().get("data")
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        df = pd.DataFrame(data)
        with lock:
            preds_proba = model.predict_proba(df)[:, 1]

        # Log requests inkl. Class
        log_request_to_csv(df, preds_proba, threshold=0.5)

        # API-Output als "Fraud" / "No Fraud"
        predicted_class = (preds_proba >= 0.5).astype(int)
        predicted_label = ["Fraud" if c == 1 else "No Fraud" for c in predicted_class]

        return jsonify({"class": predicted_label})

    except Exception as e:
        logging.exception("Prediction error")
        return jsonify({"error": str(e)}), 400

# MAIN
if __name__ == "__main__":
    try:
        latest_model_key = get_latest_model_key()
        model = load_model_from_s3(latest_model_key)
    except Exception as e:
        logging.error(f"Failed to load initial model: {e}")
    app.run(host="0.0.0.0", port=8000)
