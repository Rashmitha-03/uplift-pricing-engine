"""XGBoost churn model + isotonic calibration, logged to MLflow."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from src.features import get_splits  # noqa: E402

OUT = Path("docs/churn")
PARAMS = dict(
    n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, eval_metric="logloss", random_state=42,
)


def metrics(y, p) -> dict:
    return {
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    s = get_splits()
    X_tr, y_tr = s["train"]
    X_ca, y_ca = s["calib"]
    X_te, y_te = s["test"]

    model = XGBClassifier(**PARAMS).fit(X_tr, y_tr)
    iso = IsotonicRegression(out_of_bounds="clip").fit(model.predict_proba(X_ca)[:, 1], y_ca)

    p_raw = model.predict_proba(X_te)[:, 1]
    p_cal = iso.predict(p_raw)
    m_raw, m_cal = metrics(y_te, p_raw), metrics(y_te, p_cal)

    fig, ax = plt.subplots(figsize=(5, 5))
    for p, label in [(p_raw, "raw"), (p_cal, "isotonic")]:
        frac, mean_p = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
        ax.plot(mean_p, frac, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed churn rate")
    ax.set_title("Calibration curve (test)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "calibration.png", dpi=120)

    mlflow.set_experiment("churn")
    with mlflow.start_run(run_name="xgb_isotonic"):
        mlflow.log_params(PARAMS)
        mlflow.log_metrics({f"test_raw_{k}": v for k, v in m_raw.items()})
        mlflow.log_metrics({f"test_cal_{k}": v for k, v in m_cal.items()})
        mlflow.log_artifact(str(OUT / "calibration.png"))

    print("TEST raw       :", {k: round(v, 4) for k, v in m_raw.items()})
    print("TEST calibrated:", {k: round(v, 4) for k, v in m_cal.items()})


if __name__ == "__main__":
    main()
