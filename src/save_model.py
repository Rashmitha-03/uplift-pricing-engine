"""Train the final churn model + Platt calibrator and save them to disk."""

from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.churn import PARAMS, logit
from src.features import get_splits

MODELS = Path("models")


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    s = get_splits()
    X_tr, y_tr = s["train"]
    X_ca, y_ca = s["calib"]
    model = XGBClassifier(**PARAMS).fit(X_tr, y_tr)
    platt = LogisticRegression(C=1e6).fit(logit(model.predict_proba(X_ca)[:, 1]), y_ca)
    joblib.dump({"model": model, "platt": platt, "columns": list(X_tr.columns)},
                MODELS / "churn.joblib")
    print("saved", MODELS / "churn.joblib", "| slope, intercept:",
          platt.coef_.round(3).tolist(), platt.intercept_.round(3).tolist())


if __name__ == "__main__":
    main()
