"""
data_cleaning.py - curățarea setului de date cars.csv

Deciziile de mai jos sunt explicate pe larg în notebooks/01_eda.ipynb.

Rulare de test:
    python src/data_cleaning.py
"""
import numpy as np
import pandas as pd

from config import DATA_PATH, REFERENCE_YEAR, TARGET

# Nume de coloane mai ușor de folosit în cod (fără paranteze și unități în nume)
COLUMN_NAMES = {
    "priceUSD": "price_usd",
    "mileage(kilometers)": "mileage_km",
    "volume(cm3)": "volume_cm3",
}

TEXT_COLUMNS = ["make", "model", "condition", "fuel_type", "color",
                "transmission", "drive_unit", "segment"]

# Limitele pentru rânduri valide (vezi EDA)
MIN_PRICE, MAX_PRICE = 100, 100_000     # sub 100 $ sau peste 100.000 $ = greșeli sau mașini de colecție
MIN_YEAR = 1980                         # mașinile mai vechi sunt de colecție; prețul lor urmează alte reguli
MAX_MILEAGE = 1_000_000                 # peste 1 mil. km = valori „de umplutură” (ex. 9.999.999)
MAX_ENGINE_CM3 = 8_000                  # motoare de 16.000–20.000 cm³ = zero în plus (20000 → 2000)


def load_raw_data(path=DATA_PATH):
    return pd.read_csv(path)


def rename_columns(df):
    """priceUSD - price_usd, mileage(kilometers) → mileage_km, volume(cm3) - volume_cm3"""
    return df.rename(columns=COLUMN_NAMES)


def standardize_text(df):
    """Text fără spații la capete și cu litere mici, ca 'BMW ' și 'bmw' să fie aceeași valoare."""
    df = df.copy()
    for col in TEXT_COLUMNS:
        if col in df.columns:
            # valorile lipsă rămân lipsă (NaN), restul devin text curat
            cleaned = df[col].astype(str).str.strip().str.lower()
            df[col] = cleaned.where(df[col].notna(), np.nan).astype(object)
    # segmentul auto e o literă (A, B, C...), îl las cu majusculă, cum e folosit în mod normal
    if "segment" in df.columns:
        df["segment"] = df["segment"].str.upper()
    return df


def fix_numeric_values(df):
    """Corectează valorile numerice imposibile, fără să șteargă mașina.

    - kilometraj > 1.000.000 km - necunoscut (NaN, completat mai târziu cu mediana);
    - kilometraj < 1.000 km la o mașină mai veche de 2 ani - probabil scris în mii de km → necunoscut;
    - capacitate > 8.000 cm³ - are un zero în plus (20000 - 2000);
    - mașini electrice - capacitatea motorului nu există, o setez 0.
    """
    df = df.copy()
    df["mileage_km"] = df["mileage_km"].astype(float)
    df["volume_cm3"] = df["volume_cm3"].astype(float)

    df.loc[df["mileage_km"] > MAX_MILEAGE, "mileage_km"] = np.nan

    if "year" in df.columns:
        age = REFERENCE_YEAR - df["year"]
        df.loc[(df["mileage_km"] < 1000) & (age > 2), "mileage_km"] = np.nan

    too_big = df["volume_cm3"] > MAX_ENGINE_CM3
    df.loc[too_big, "volume_cm3"] = df.loc[too_big, "volume_cm3"] / 10

    df.loc[df["fuel_type"] == "electrocar", "volume_cm3"] = 0
    return df


def remove_invalid_rows(df):
    """Doar pentru antrenare: elimin duplicatele și rândurile cu preț sau an nerealist."""
    before = len(df)
    df = df.drop_duplicates()
    df = df[df[TARGET].between(MIN_PRICE, MAX_PRICE)]
    df = df[df["year"] >= MIN_YEAR]
    print(f"[curățare] rânduri: {before} → {len(df)} (eliminate {before - len(df)})")
    return df.reset_index(drop=True)


def clean_data(df, training=True):
    """Pașii de curățare în ordine.
    training=False - pentru o mașină nouă (predicție): nu elimin rânduri, doar standardizez și corectez."""
    df = rename_columns(df)
    df = standardize_text(df)
    df = fix_numeric_values(df)
    if training:
        df = remove_invalid_rows(df)
    return df


if __name__ == "__main__":
    raw = load_raw_data()
    clean = clean_data(raw)
    print(clean.head())
    print("\nValori lipsă după curățare:")
    print(clean.isna().sum()[clean.isna().sum() > 0])
