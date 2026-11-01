"""Baseline XGBoost churn classifier, evaluated on the calib split."""

from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from src.features import get_splits


def main() -> None:
    s = get_splits()
    X_tr, y_tr = s["train"]
    X_ca, y_ca = s["calib"]
    model = XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, eval_metric="logloss", random_state=42,
    )
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_ca)[:, 1]
    print("features:", X_tr.shape[1])
    print("baseline rate (PR-AUC of random):", round(y_ca.mean(), 4))
    print("PR-AUC :", round(average_precision_score(y_ca, p), 4))
    print("ROC-AUC:", round(roc_auc_score(y_ca, p), 4))


if __name__ == "__main__":
    main()
