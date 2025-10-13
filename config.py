# S3 settings
BUCKET_NAME = "fraud-detection-project-data-science-2025"  
MONTHS_PREFIX = "dataset/months/"  

# Paths for local MLflow tracking and model storage
MLFLOW_TRACKING_URI = "file:./mlruns"  # Local MLflow directory
LATEST_MODEL_PATH = "./mlruns/latest_model/model.pkl"  # Local path for latest model

# Drift-check interval (in seconds)
CHECK_INTERVAL = 300  

# S3 model backup settings
ENABLE_S3_MODEL_BACKUP = True  
S3_MODEL_PATH = "models/latest_model/model.pkl" 

# Prefix for model version backups on S3 
S3_MODEL_BACKUP_PREFIX = "model_backups"  # Folder for timestamped model backups