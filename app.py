from flask import Flask, request, jsonify
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

# lask App
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# boto3 S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION
)

# Global model
model = None
latest_model_key = None
lock = threading.Lock()

def get_latest_model_key():
    """Retrieve the latest model key from S3."""
    resp = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=MODEL_BACKUPS_PREFIX)
    keys = [obj['Key'] for obj in resp.get('Contents', []) if obj['Key'].endswith("model.pkl")]
    if not keys:
        raise FileNotFoundError("No model found in S3.")
    latest = sorted(keys, reverse=True)[0]
    return latest

def load_model_from_s3(key):
    """Load model from S3 using the specified key."""
    obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
    m = joblib.load(BytesIO(obj['Body'].read()))
    logging.info(f"Model loaded from s3://{BUCKET_NAME}/{key}")
    return m

def reload_model_periodically():
    """Periodically check S3 for a new model and reload it."""
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

# Start background thread
thread = threading.Thread(target=reload_model_periodically, daemon=True)
thread.start()

# Health Check Endpoint
@app.route("/health")
def health():
    return jsonify({"status": "ok"})

# Prediction Endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json().get("data")
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        df = pd.DataFrame(data)
        with lock:
            preds = model.predict_proba(df)[:, 1]
        return jsonify({"predictions": preds.tolist()})
    except Exception as e:
        logging.exception("Prediction error")
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Load initial model
    try:
        latest_model_key = get_latest_model_key()
        model = load_model_from_s3(latest_model_key)
    except Exception as e:
        logging.error(f"Failed to load initial model: {e}")
    app.run(host="0.0.0.0", port=8000)
