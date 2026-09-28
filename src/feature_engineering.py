"""
feature_engineering.py - caracteristici noi create din coloanele existente

| Caracteristică       | De ce?                                                                 |
|----------------------|------------------------------------------------------------------------|
| car_age              | prețul scade cu vârsta; „vârsta” e mai intuitivă decât anul            |
| mileage_per_year     | 200.000 km la 20 de ani e normal, la 3 ani înseamnă uzură mare          |
| engine_volume_liters | cum se vorbește de obicei despre motoare (1.6, 2.0)                    |
| is_high_mileage      | peste 300.000 km cumpărătorii se tem de reparații mari                  |
| is_newer_car         | mașinile de după 2010 au alt nivel de dotări și de preț                 |
| brand_model          | „golf” sau „a6” spun mai mult împreună cu marca (volkswagen_golf)       |

Rulare de test:
    python src/feature_engineering.py
"""
import numpy as np

from config import REFERENCE_YEAR

HIGH_MILEAGE_KM = 300_000
NEWER_CAR_YEAR = 2010


def add_features(df):
    df = df.copy()
    df["car_age"] = (REFERENCE_YEAR - df["year"]).clip(lower=0)
    # clip(lower=1): o mașină din anul curent are vârsta 0 – evit împărțirea la 0
    df["mileage_per_year"] = df["mileage_km"] / df["car_age"].clip(lower=1)
    df["engine_volume_liters"] = df["volume_cm3"] / 1000
    # dacă kilometrajul e necunoscut, și indicatorul rămâne necunoscut (îl completează imputer-ul)
    df["is_high_mileage"] = np.where(df["mileage_km"].isna(), np.nan,
                                     (df["mileage_km"] > HIGH_MILEAGE_KM).astype(float))
    df["is_newer_car"] = (df["year"] >= NEWER_CAR_YEAR).astype(int)
    df["brand_model"] = df["make"] + "_" + df["model"]
    return df


if __name__ == "__main__":
    from data_cleaning import clean_data, load_raw_data

    df = add_features(clean_data(load_raw_data()))
    cols = ["year", "car_age", "mileage_km", "mileage_per_year", "volume_cm3",
            "engine_volume_liters", "is_high_mileage", "is_newer_car", "brand_model"]
    print(df[cols].head(10))
