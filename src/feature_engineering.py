import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for NHANES diabetes prediction.
    Includes demographic, behavioral, dietary, and family history variables.
    """

    df = df.copy()
    
    # -----------------------------
    # REMOVE LEAKAGE VARIABLES & METADATA
    # -----------------------------
    leakage_cols = ["DID040"]
    non_informative_cols = ["SEQN", "cycle"]
    df = df.drop(columns=[c for c in leakage_cols + non_informative_cols if c in df.columns])

    # -----------------------------
    # Outcome variable
    # -----------------------------
    # DIQ010: 1 = Yes, 2 = No
    df = df[df["DIQ010"].isin([1, 2])]
    df["diabetes"] = df["DIQ010"].map({1: 1, 2: 0}).astype(int)

    # -----------------------------
    # Demographics
    # -----------------------------
    if "RIDAGEYR" in df.columns:
        df["age"] = df["RIDAGEYR"]

    if "RIAGENDR" in df.columns:
        df["female"] = df["RIAGENDR"].map({1: 0, 2: 1})

    if "BMXBMI" in df.columns:
        df["bmi"] = df["BMXBMI"]

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
    # Drop raw NHANES columns
    # -----------------------------
    drop_cols = [
        "DIQ010", "RIDAGEYR", "RIAGENDR", "BMXBMI",
        "MCQ300C", "SMQ020", "PAQ650"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df
