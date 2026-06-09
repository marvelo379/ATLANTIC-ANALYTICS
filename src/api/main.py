from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import os
app = FastAPI(title="Music ML Dashboard Backend")
# 1. BASE_DIR points directly to: .../unified_mlproject1/src/ml/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Track the root of the entire repository by going up two levels from src/ml
# This gives you the root folder (.../unified_mlproject1) regardless of the computer name
REPO_ROOT = os.path.normpath(os.path.join(BASE_DIR, "../.."))

# 3. Define absolute paths dynamically using the REPO_ROOT as the anchor point
MERGED_DATA_PATH = os.path.normpath(os.path.join(
    REPO_ROOT, 
    "src/ml/ml_project_complete_workspace/merged_historical_and_predictions.csv"
))

CONTENT_PROFILES_PATH = os.path.normpath(os.path.join(
    REPO_ROOT, 
    "data/processed/song_content_profiles.csv"
))
def load_merged_data():
    try:
        df = pd.read_csv(MERGED_DATA_PATH)
        if "period" in df.columns:
            df["period"] = df["period"].astype(str)
        # Ensure NaNs are handled nicely for JSON conversion
        return df.replace({np.nan: None})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading merged data: {str(e)}")

def load_content_data():
    try:
        content_df = pd.read_csv(CONTENT_PROFILES_PATH)
        content_df["is_explicit"] = content_df["is_explicit"].astype(str).str.lower()
        content_df["content_type"] = np.where(
            content_df["is_explicit"].isin(["true", "1"]), "Explicit", "Clean"
        )
        return content_df.replace({np.nan: None})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading content data: {str(e)}")

@app.get("/api/merged-data")
def get_merged_data():
    df = load_merged_data()
    return df.to_dict(orient="records")

@app.get("/api/content-profiles")
def get_content_profiles():
    df = load_content_data()
    return df.to_dict(orient="records")