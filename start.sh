#!/bin/bash

# Start Flask App auf 8000
echo "Starting Flask API..."
python app.py &

# Start MLflow UI auf Port 5001
echo "Starting MLflow UI..."
mlflow ui --host 0.0.0.0 --port 5001 &

# Start Drift Watchdog
echo "Starting Drift Watchdog..."
python drift_watchdog.py &

wait