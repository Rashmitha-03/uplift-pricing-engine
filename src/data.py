"""Load the Telco CSV, clean it, split it, and store it in DuckDB."""

from pathlib import Path

import duckdb
import pandas as pd
from sklearn.model_selection import train_test_split

RAW = Path("data/raw/telco.csv")
DB = Path("data/processed/telco.duckdb")
SEED = 42


def load_clean(path: Path = RAW) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 11 customers with tenure 0 have a blank TotalCharges; they have not been billed yet
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["churn"] = (df["Churn"] == "Yes").astype(int)
    return df.drop(columns="Churn")


def add_split(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Stratified 60/20/20 split into train / calib / test."""
    train_idx, rest_idx = train_test_split(
        df.index, test_size=0.4, stratify=df["churn"], random_state=seed
    )
    calib_idx, test_idx = train_test_split(
        rest_idx, test_size=0.5, stratify=df.loc[rest_idx, "churn"], random_state=seed
    )
    out = df.copy()
    out["split"] = "train"
    out.loc[calib_idx, "split"] = "calib"
    out.loc[test_idx, "split"] = "test"
    return out


def main() -> None:
    df = add_split(load_clean())
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB))
    con.execute("CREATE OR REPLACE TABLE customers AS SELECT * FROM df")
    summary = con.execute(
        "SELECT split, COUNT(*) AS n, ROUND(AVG(churn), 4) AS churn_rate "
        "FROM customers GROUP BY split ORDER BY split"
    ).df()
    con.close()
    print(df.shape)
    print(summary)


if __name__ == "__main__":
    main()
