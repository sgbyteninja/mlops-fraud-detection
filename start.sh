#!/bin/bash
# Start Flask App
echo "Starting Flask API..."
python app.py &

# Start Drift Watchdog
echo "Starting Drift Watchdog..."
python drift_watchdog.py &
wait