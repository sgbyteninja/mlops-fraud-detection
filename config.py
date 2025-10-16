BUCKET_NAME = "fraud-detection-project-data-science-2025"
WEEKS_PREFIX = "weekly_data/"

MLFLOW_TRACKING_URI = "file:./mlruns"
LATEST_MODEL_PATH = "./models/latest_model/model.pkl"

CHECK_INTERVAL = 150

NUM_WEEKS_FOR_TRAINING = 12

ENABLE_S3_MODEL_BACKUP = True
S3_MODEL_PATH = "models/latest_model/model.pkl"
S3_MODEL_BACKUP_PREFIX = "model_backups"
