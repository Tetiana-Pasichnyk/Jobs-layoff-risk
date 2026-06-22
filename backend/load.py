import pandas as pd


dateiname = "../data/rohdaten-ai-impact-jobs-layoff-risk-dataset.csv"


def load_data(path):
    df = pd.read_csv(path)
    print(f"[LOAD] Datensatz geladen: {df.shape}")
    return df

df = load_data(dateiname)

def clean_data(df, num_cols, cat_cols):
    # numerische Spalten
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # kategoriale Spalten
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