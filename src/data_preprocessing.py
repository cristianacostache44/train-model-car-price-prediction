"""
data_preprocessing.py - ce coloane intră în model și cum sunt transformate

Două variante de preprocesare, pentru că modelele „gândesc” diferit:
  - linear  - pentru Linear Regression: StandardScaler pe numere + OneHotEncoder pe categorii
              (un model liniar are nevoie ca fiecare categorie să fie o coloană 0/1);
  - tree    - pentru modelele bazate pe arbori: OrdinalEncoder pe categorii
              (arborii nu au nevoie de scalare, iar one-hot pe ~1.000 de modele ar fi foarte lent).
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from config import RANDOM_STATE, TARGET, TEST_SIZE

NUMERIC_FEATURES = [
    "car_age",
    "mileage_km",
    "mileage_per_year",
    "engine_volume_liters",
    "is_high_mileage",
    "is_newer_car",
]

CATEGORICAL_FEATURES = [
    "make",
    "brand_model",
    "condition",
    "fuel_type",
    "color",
    "transmission",
    "drive_unit",
    "segment",
]

# Coloane care NU intră în model:
#   year        – înlocuit de car_age (aceeași informație)
#   model       – înlocuit de brand_model (mai precis)
#   volume_cm3  – înlocuit de engine_volume_liters


def split_features_target(df):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    return X, y


def get_train_test(df):
    """Aceeași împărțire 80/20 pentru toate modelele → comparație corectă."""
    X, y = split_features_target(df)
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def build_preprocessor(kind="tree"):
    # Numere: valorile lipsă (kilometraj necunoscut) → mediana
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if kind == "linear":
        numeric_steps.append(("scaler", StandardScaler()))

    # Categorii: valorile lipsă (drive_unit, segment) devin o categorie separată „unknown”
    if kind == "linear":
        encoder = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5)
    else:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encoder", encoder),
    ])

    return ColumnTransformer([
        ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


if __name__ == "__main__":
    from data_cleaning import clean_data, load_raw_data
    from feature_engineering import add_features

    df = add_features(clean_data(load_raw_data()))
    X_train, X_test, y_train, y_test = get_train_test(df)
    print("Train:", X_train.shape, "| Test:", X_test.shape)
    for kind in ["linear", "tree"]:
        Xt = build_preprocessor(kind).fit_transform(X_train)
        print(f"Preprocesare '{kind}': {Xt.shape[1]} coloane după transformare")
