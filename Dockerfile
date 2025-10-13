FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install boto3 mlflow pandas scikit-learn joblib requests watchdog

#Copy the training data
#COPY dataset/months ./dataset/months

# Copy scripts
COPY app.py .
COPY config.py .
COPY drift_check.py .
COPY drift_watchdog.py .
COPY train.py .
COPY retrain.py .
COPY simulate_year.py .

# Copy start script
COPY start.sh .
RUN chmod +x start.sh

# Make sure mlruns folder exists for MLflow
RUN mkdir -p ./mlruns

# Expose Flask port
EXPOSE 8000

# Start all processes
CMD ["./start.sh"]
