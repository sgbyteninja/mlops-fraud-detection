import pandas as pd
from io import BytesIO
import boto3
import os
from dotenv import load_dotenv
from config import BUCKET_NAME, MONTHS_PREFIX
from scipy.stats import ttest_ind

# Load environment variables
load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-central-1")

# boto3 S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION
)

def load_latest_month():
    """Load the latest CSV month from S3."""
    resp = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=MONTHS_PREFIX)
    files = sorted([obj['Key'] for obj in resp.get('Contents', []) if obj['Key'].endswith('.csv')])
    if not files:
        return None
    latest = files[-1]
    df = pd.read_csv(BytesIO(s3_client.get_object(Bucket=BUCKET_NAME, Key=latest)['Body'].read()))
    return df

def check_drift(reference_df, latest_df, alpha=0.05):
    """Check for drift using t-test per feature."""
    drifted_features = []
    for col in reference_df.columns:
        if col not in latest_df.columns:
            continue
        t_stat, p_value = ttest_ind(reference_df[col], latest_df[col], equal_var=False)
        if p_value < alpha:
            drifted_features.append(col)
    return drifted_features

if __name__ == "__main__":
    df_latest = load_latest_month()
    if df_latest is not None:
        # Use the first month as reference
        resp = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=MONTHS_PREFIX)
        first_key = sorted([obj['Key'] for obj in resp.get('Contents', []) if obj['Key'].endswith('.csv')])[0]
        df_ref = pd.read_csv(BytesIO(s3_client.get_object(Bucket=BUCKET_NAME, Key=first_key)['Body'].read()))
        
        drift_features = check_drift(df_ref, df_latest)
        print("Drift detected in features:", drift_features)
    else:
        print("No new data found.")
