import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from feature_engineering import engineer_features
from models import get_models
from evaluation import (
    evaluate_model,
    plot_roc_curve,
    plot_calibration_curve
)


DATA_PATH = "data/nhanes_diabetes_2007_2020_clean.csv"
OUTPUT_DIR = "outputs"
RANDOM_STATE = 42


def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    return preprocessor


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -----------------------------
    # Load & engineer data
    # -----------------------------
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)

    y = df["diabetes"].values
    X = df.drop(columns=["diabetes"])

    # -----------------------------
    # Train / test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(X_train)
    models = get_models(RANDOM_STATE)

    results = []
    prob_outputs = {}

    # -----------------------------
    # Train models
    # -----------------------------
    for name, model in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]

        metrics = evaluate_model(y_test, probs)
        metrics["model"] = name
        results.append(metrics)

        prob_outputs[name] = probs

    # -----------------------------
    # Save results
    # -----------------------------
    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    results_df.to_csv(f"{OUTPUT_DIR}/model_results.csv", index=False)

    plot_roc_curve(prob_outputs, y_test, f"{OUTPUT_DIR}/roc_curves.png")
    plot_calibration_curve(prob_outputs, y_test, f"{OUTPUT_DIR}/calibration_curves.png")

    print("\nTraining complete. Results:")
    print(results_df)


if __name__ == "__main__":
    main()
