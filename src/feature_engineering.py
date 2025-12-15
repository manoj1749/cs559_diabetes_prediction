import numpy as np
import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering and recoding for NHANES diabetes dataset.
    Returns a clean dataframe with engineered features.
    """
    
    df = df.copy()
    
    
    # -----------------------------
    # Outcome variable
    # -----------------------------
    # DIQ010: 1 = Yes, 2 = No
    df = df[df["DIQ010"].isin([1, 2])]
    df["diabetes"] = df["DIQ010"].map({1: 1, 2: 0}).astype(int)

    # -----------------------------
    # Demographics
    # -----------------------------
    if "RIAGENDR" in df.columns:
        df["female"] = df["RIAGENDR"].map({1: 0, 2: 1})

    # -----------------------------
    # Family history
    # -----------------------------
    if "MCQ300C" in df.columns:
        df["family_history"] = df["MCQ300C"].map({1: 1, 2: 0})

    # -----------------------------
    # Smoking
    # -----------------------------
    if "SMQ020" in df.columns:
        df["smoker"] = df["SMQ020"].map({1: 1, 2: 0})

    # -----------------------------
    # Physical activity
    # -----------------------------
    if "PAQ650" in df.columns:
        if set(df["PAQ650"].dropna().unique()).issubset({1, 2}):
            df["vigorous_pa"] = df["PAQ650"].map({1: 1, 2: 0})
        else:
            df["vigorous_pa"] = df["PAQ650"]

    # -----------------------------
    # Dietary feature engineering
    # -----------------------------
    if {"DR1TSUGR", "DR1TKCAL"}.issubset(df.columns):
        kcal = df["DR1TKCAL"].replace(0, np.nan)
        df["sugar_density"] = (df["DR1TSUGR"] / kcal) * 1000

    # -----------------------------
    # Drop raw columns not needed for modeling
    # -----------------------------
    drop_cols = [
        "DIQ010", "RIAGENDR", "MCQ300C",
        "SMQ020", "PAQ650"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df