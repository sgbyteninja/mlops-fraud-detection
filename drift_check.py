import pandas as pd
from io import BytesIO
import boto3
import os
from dotenv import load_dotenv
from config import BUCKET_NAME, WEEKS_PREFIX
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

def load_latest_weeks(n_weeks=3):
    """Load the latest week as new data, and previous n_weeks as reference."""
    resp = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=WEEKS_PREFIX)
    files = sorted([obj["Key"] for obj in resp.get("Contents", []) if obj["Key"].endswith(".csv")])
    if len(files) < n_weeks + 1:
        print(f"Not enough data: need {n_weeks+1} weeks, found {len(files)}")
        return None, None

    # Newest week
    latest_key = files[-1]
    latest_obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=latest_key)
    df_latest = pd.read_csv(BytesIO(latest_obj["Body"].read()))

    # Reference: previous n_weeks
    ref_keys = files[-(n_weeks+1):-1]
    df_ref_list = []
    for key in ref_keys:
        obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
        df_ref_list.append(pd.read_csv(BytesIO(obj["Body"].read())))
    df_ref = pd.concat(df_ref_list, ignore_index=True)

    return df_ref, df_latest

def check_drift(reference_df, latest_df, alpha=0.05):
    """Check for drift using t-tests per numeric feature with Bonferroni correction."""
    drifted_features = []
    numeric_cols = [c for c in reference_df.columns if pd.api.types.is_numeric_dtype(reference_df[c])]

    n_tests = len(numeric_cols)
    if n_tests == 0:
        print("No numeric features found to test.")
        return []

    alpha_adj = alpha / n_tests
    print(f"Performing {n_tests} t-tests with Bonferroni correction (α={alpha}, adjusted α={alpha_adj:.6f})")

    for col in numeric_cols:
        if col not in latest_df.columns:
            continue
        ref_col = reference_df[col].dropna()
        new_col = latest_df[col].dropna()
        if len(ref_col) < 2 or len(new_col) < 2:
            continue
        try:
            t_stat, p_value = ttest_ind(ref_col, new_col, equal_var=False)
            if p_value < alpha_adj:
                drifted_features.append(col)
        except Exception as e:
            print(f"Skipping {col} due to error: {e}")
    return drifted_features

if __name__ == "__main__":
    df_ref, df_latest = load_latest_weeks(n_weeks=3)
    if df_latest is not None and df_ref is not None:
        drift_features = check_drift(df_ref, df_latest, alpha=0.05)
        if drift_features:
            print("Drift detected in features:", drift_features)
        else:
            print("No significant drift detected.")
    else:
        print("Not enough data to perform drift check.")
