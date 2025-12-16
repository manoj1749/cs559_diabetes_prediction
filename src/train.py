import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from feature_engineering import engineer_features
from models import get_models
from evaluation import (
    evaluate_model,
    plot_roc_curve,
    plot_calibration_curve,
    plot_decision_curve,
    plot_confusion_matrix
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

    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])


def train_neural_network(X_train, y_train, X_val, y_val, input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")]
    )

    es = callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=10,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=128,
        verbose=0,
        callbacks=[es]
    )

    return model


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)

    y = df["diabetes"].values
    X = df.drop(columns=["diabetes"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(X_train)
    preprocessor.fit(X_train)
    
    # -----------------------------
    # Prepare data for Neural Network
    # -----------------------------
    X_train_nn = preprocessor.transform(X_train)
    X_test_nn = preprocessor.transform(X_test)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_nn, y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=RANDOM_STATE
    )

    models_dict = get_models(RANDOM_STATE)

    metrics_list = []
    prob_outputs = {}

    for name, model in models_dict.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]

        plot_confusion_matrix(
            y_test, probs, name,
            f"{OUTPUT_DIR}/confusion_matrix_{name}.png"
        )

        metrics = evaluate_model(y_test, probs)
        metrics["model"] = name
        metrics_list.append(metrics)
        prob_outputs[name] = probs
        
    # -----------------------------
    # Train & evaluate Neural Network
    # -----------------------------
    nn_model = train_neural_network(
        X_tr, y_tr, X_val, y_val,
        input_dim=X_train_nn.shape[1]
    )

    nn_probs = nn_model.predict(X_test_nn).ravel()
    nn_probs = np.clip(nn_probs, 1e-6, 1 - 1e-6)

    plot_confusion_matrix(
        y_test, nn_probs, "NeuralNetwork",
        f"{OUTPUT_DIR}/confusion_matrix_NeuralNetwork.png"
    )

    nn_metrics = evaluate_model(y_test, nn_probs)
    nn_metrics["model"] = "NeuralNetwork"

    metrics_list.append(nn_metrics)
    prob_outputs["NeuralNetwork"] = nn_probs

    # SHAP for XGBoost
    if "XGBoost" in models_dict:
        xgb_pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", models_dict["XGBoost"])
        ])
        xgb_pipe.fit(X_train, y_train)

        X_train_trans = preprocessor.transform(X_train)
        feature_names = preprocessor.get_feature_names_out()

        explainer = shap.TreeExplainer(xgb_pipe.named_steps["model"])
        shap_values = explainer.shap_values(X_train_trans)

        shap.summary_plot(
            shap_values,
            X_train_trans,
            feature_names=feature_names,
            show=False
        )
        plt.savefig(f"{OUTPUT_DIR}/shap_summary_xgboost.png")
        plt.close()

    results_df = pd.DataFrame(metrics_list).sort_values("roc_auc", ascending=False)
    results_df.to_csv(f"{OUTPUT_DIR}/model_results.csv", index=False)

    plot_roc_curve(prob_outputs, y_test, f"{OUTPUT_DIR}/roc_curves.png")
    plot_calibration_curve(prob_outputs, y_test, f"{OUTPUT_DIR}/calibration_curves.png")
    plot_decision_curve(prob_outputs, y_test, f"{OUTPUT_DIR}/decision_curve.png")

    print("\nFinal Results:")
    print(results_df)


if __name__ == "__main__":
    main()
