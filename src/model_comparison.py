"""
model_comparison.py - compară mai mulți algoritmi de regresie pe EXACT aceleași date de train/test

Rulare:
    python src/model_comparison.py      (durează ~2-3 minute)

Rezultatul se salvează și în models/model_comparison.csv.
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from config import COMPARISON_PATH, MODELS_DIR, RANDOM_STATE
from data_preprocessing import build_preprocessor, get_train_test
from model_evaluation import regression_metrics
from model_training import build_final_model, load_prepared_data, log_target


def get_candidates():
    """(nume, pipeline). Primul model e un „baseline” pe prețul brut, ca să văd cât ajută log(preț)."""
    return {
        "Linear Regression (preț brut)": Pipeline([
            ("prep", build_preprocessor("linear")), ("model", LinearRegression())]),
        "Linear Regression (log preț)": Pipeline([
            ("prep", build_preprocessor("linear")), ("model", log_target(LinearRegression()))]),
        "Decision Tree": Pipeline([
            ("prep", build_preprocessor("tree")),
            ("model", log_target(DecisionTreeRegressor(min_samples_leaf=5, random_state=RANDOM_STATE)))]),
        "Random Forest": Pipeline([
            ("prep", build_preprocessor("tree")),
            ("model", log_target(RandomForestRegressor(n_estimators=100, min_samples_leaf=2,
                                                       n_jobs=-1, random_state=RANDOM_STATE)))]),
        "Gradient Boosting (setări implicite)": Pipeline([
            ("prep", build_preprocessor("tree")),
            ("model", log_target(GradientBoostingRegressor(random_state=RANDOM_STATE)))]),
        "Gradient Boosting (final, reglat)": build_final_model(),
    }


def compare_models():
    X_train, X_test, y_train, y_test = get_train_test(load_prepared_data())
    rows = []
    for name, pipeline in get_candidates().items():
        start = time.time()
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        rows.append({"Model": name, **regression_metrics(y_test, pred),
                     "Timp (s)": round(time.time() - start, 1)})
        print(f"[OK] {name}")
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


if __name__ == "__main__":
    results = compare_models()
    shown = results.copy()
    for col in ["MAE", "RMSE"]:
        shown[col] = shown[col].map("{:,.0f} $".format)
    shown["MSE"] = shown["MSE"].map("{:,.0f}".format)
    shown["R2"] = shown["R2"].map("{:.3f}".format)
    shown["MedAPE_%"] = shown["MedAPE_%"].map("{:.1f}%".format)
    print("\n" + shown.to_string(index=False))

    MODELS_DIR.mkdir(exist_ok=True)
    results.round(4).to_csv(COMPARISON_PATH, index=False)
    print(f"\nRezultate salvate în {COMPARISON_PATH.name}")
    best = results.iloc[0]
    print(f"Cel mai mic MAE: {best['Model']} ({best['MAE']:,.0f} $, R2 = {best['R2']:.3f})")
