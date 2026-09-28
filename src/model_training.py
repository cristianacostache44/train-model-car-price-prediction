"""
model_training.py - antrenează modelul final și îl salvează în models/car_price_model.joblib

Rulare:
    python src/model_training.py

Modelul final (ales în model_comparison.py / notebooks/02_modeling.ipynb):
    Gradient Boosting Regressor, antrenat pe log(preț).

De ce log(preț)?
    Prețurile au o distribuție foarte „lungă” spre dreapta (majoritatea sub 10.000 $, câteva peste 50.000 $).
    Pe scara logaritmică modelul învață diferențe relative („cu 20% mai scumpă”), nu absolute,
    iar mașinile foarte scumpe nu mai domină antrenarea. TransformedTargetRegressor face automat
    log la antrenare și exp la predicție, deci rezultatul iese tot în dolari.
"""
import time

import joblib
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from config import MODEL_PATH, MODELS_DIR, RANDOM_STATE
from data_cleaning import clean_data, load_raw_data
from data_preprocessing import build_preprocessor, get_train_test
from feature_engineering import add_features


def log_target(model):
    """Învelește un model astfel încât să învețe pe log(1 + preț)."""
    return TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)


def build_final_model():
    gbr = GradientBoostingRegressor(
        n_estimators=800,      # câți arbori mici se adaugă unul după altul
        learning_rate=0.05,    # cât de mult corectează fiecare arbore (mic = mai stabil)
        max_depth=6,           # cât de „complex” e fiecare arbore
        subsample=0.8,         # fiecare arbore vede 80% din date → mai puțin overfitting
        random_state=RANDOM_STATE,
    )
    return Pipeline([
        ("preprocessing", build_preprocessor(kind="tree")),
        ("model", log_target(gbr)),
    ])


def load_prepared_data():
    """Date brute → curățare → caracteristici noi (pașii comuni tuturor scripturilor)."""
    return add_features(clean_data(load_raw_data()))


def main():
    df = load_prepared_data()
    X_train, X_test, y_train, y_test = get_train_test(df)

    print(f"Antrenez Gradient Boosting pe {len(X_train)} mașini...")
    start = time.time()
    model = build_final_model().fit(X_train, y_train)
    print(f"Gata în {time.time() - start:.0f} s")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model salvat: {MODEL_PATH.name} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")
    print("Pentru metrici rulează:  python src/model_evaluation.py")


if __name__ == "__main__":
    main()
