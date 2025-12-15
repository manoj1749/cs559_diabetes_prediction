import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    brier_score_loss
)
from sklearn.calibration import calibration_curve


def evaluate_model(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
    }


def plot_roc_curve(prob_dict, y_true, save_path):
    plt.figure()
    for name, probs in prob_dict.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_calibration_curve(prob_dict, y_true, save_path):
    plt.figure()
    for name, probs in prob_dict.items():
        frac_pos, mean_pred = calibration_curve(
            y_true, probs, n_bins=10, strategy="quantile"
        )
        plt.plot(mean_pred, frac_pos, marker="o", label=name)

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Probability")
    plt.title("Calibration Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def decision_curve(y_true, y_prob, thresholds):
    N = len(y_true)
    net_benefits = []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))

        if t == 1:
            net_benefit = 0
        else:
            net_benefit = (tp / N) - (fp / N) * (t / (1 - t))

        net_benefits.append(net_benefit)

    return net_benefits


def plot_decision_curve(prob_dict, y_true, save_path):
    thresholds = np.linspace(0.01, 0.99, 99)

    plt.figure()
    for name, probs in prob_dict.items():
        nb = decision_curve(y_true, probs, thresholds)
        plt.plot(thresholds, nb, label=name)

    prevalence = np.mean(y_true)
    treat_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    plt.plot(thresholds, treat_all, linestyle="--", label="Treat All")
    plt.plot(thresholds, np.zeros_like(thresholds), linestyle="--", label="Treat None")

    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
