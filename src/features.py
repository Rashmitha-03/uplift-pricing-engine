"""Build model features from the DuckDB customers table."""

import duckdb
import pandas as pd

DB = "data/processed/telco.duckdb"
SENSITIVE = ["gender", "SeniorCitizen"]
DROP = ["customerID", "churn", "split"] + SENSITIVE
CAT_COLS = [
    "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]


def load_table() -> pd.DataFrame:
    con = duckdb.connect(DB, read_only=True)
    df = con.execute("SELECT * FROM customers").df()
    con.close()
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=DROP)
    X = pd.get_dummies(X, columns=CAT_COLS, dtype=int)
    return X


def get_splits() -> dict:
    df = load_table()
    X = build_features(df)
    out = {}
    for name in ["train", "calib", "test"]:
        m = (df["split"] == name).values
        out[name] = (X[m].reset_index(drop=True), df.loc[m, "churn"].reset_index(drop=True))
    return out
