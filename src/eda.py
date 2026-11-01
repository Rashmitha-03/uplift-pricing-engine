"""Quick EDA on the training split only (no peeking at calib/test)."""

from pathlib import Path
import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DB = "data/processed/telco.duckdb"
OUT = Path("docs/eda")

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(DB, read_only=True)
    df = con.execute("SELECT * FROM customers WHERE split = 'train'").df()
    con.close()

    print("Missing values per column (should be all 0):")
    print(df.isna().sum()[df.isna().sum() > 0])

    for col in ["Contract", "InternetService", "PaymentMethod", "PaperlessBilling"]:
        print(f"\nChurn rate by {col}:")
        print(df.groupby(col)["churn"].agg(["count", "mean"]).round(3))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    df.groupby("Contract")["churn"].mean().plot.bar(ax=axes[0], title="Churn by contract")
    df.boxplot(column="tenure", by="churn", ax=axes[1])
    axes[1].set_title("Tenure by churn")
    df.boxplot(column="MonthlyCharges", by="churn", ax=axes[2])
    axes[2].set_title("Monthly charges by churn")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(OUT / "eda_overview.png", dpi=120)
    print(f"\nSaved {OUT / 'eda_overview.png'}")


if __name__ == "__main__":
    main()
