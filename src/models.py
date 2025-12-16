from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def get_models(random_state: int = 42):
    """
    Returns all predictive models used in the project.
    """
    models = {}

    # Logistic Regression (baseline)
    models["LogisticRegression"] = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs"
    )

    # Random Forest
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1
    )

    # Gradient Boosted Trees (XGBoost)
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state
        )

    return models
