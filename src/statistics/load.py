import pandas as pd

from src.paths import CSV_PATH, DATA_DIR


def load_data(path=None):
    path = path or CSV_PATH
    df = pd.read_csv(path)
    print(f"[LOAD] Datensatz geladen: {df.shape}")
    return df


def load_data_email(filename):
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Email file not found: {path}")

    df = pd.read_table(path)
    print(f"[LOAD] Datensatz geladen: {df.shape}")
    return df


def clean_data(df, num_cols, cat_cols):
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        mode_val = df[col].mode()
        df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else "Unbekannt")

    print("[CLEAN] Fehlende Werte wurden ersetzt")
    return df


def encode_target(df):
    df["Layoff_Risk_Numeric"] = df["Layoff_Risk"].map({
        "Low": 1,
        "Medium": 2,
        "High": 3
    })

    print("[ENCODE] Zielvariable kodiert")
    return df
