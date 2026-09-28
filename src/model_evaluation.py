"""
model_evaluation.py - evaluează modelul salvat pe setul de test (20% date nevăzute la antrenare)

Rulare:
    python src/model_evaluation.py

Metrici:
  MAE  - eroarea medie absolută, în dolari. Cea mai ușor de explicat: „greșim în medie cu X $”.
  MSE  - media erorilor la pătrat; penalizează mult greșelile mari (unitate: $², greu de interpretat).
  RMSE - rădăcina din MSE, din nou în dolari; e mai mare decât MAE dacă există câteva greșeli mari.
  R²   - ce proporție din variația prețurilor explică modelul (1 = perfect, 0 = cât media).
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import MODEL_PATH
from data_preprocessing import get_train_test
from model_training import load_prepared_data


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true, y_pred),
        # eroarea procentuală mediană: „pentru o mașină tipică greșim cu X%”
        "MedAPE_%": np.median(np.abs(y_true - y_pred) / y_true) * 100,
    }


def examples_table(y_true, y_pred, n=10, seed=42):
    table = pd.DataFrame({"preț real": y_true.values, "preț estimat": np.round(y_pred).astype(int)})
    table["eroare"] = (table["preț estimat"] - table["preț real"]).abs()
    return table.sample(n, random_state=seed).reset_index(drop=True)


def error_by_price_range(y_true, y_pred):
    """Cât greșim pe fiecare segment de preț – eroarea absolută crește odată cu prețul."""
    bins = [0, 3000, 7000, 15000, 30000, np.inf]
    labels = ["< 3k", "3k–7k", "7k–15k", "15k–30k", "> 30k"]
    df = pd.DataFrame({"real": y_true.values, "abs_err": np.abs(y_true.values - y_pred)})
    df["interval preț ($)"] = pd.cut(df["real"], bins=bins, labels=labels)
    out = df.groupby("interval preț ($)", observed=True).agg(
        mașini=("real", "size"), MAE=("abs_err", "mean"), preț_mediu=("real", "mean"))
    out["MAE % din preț"] = (out["MAE"] / out["preț_mediu"] * 100).round(1)
    return out.round(0)


def main():
    model = joblib.load(MODEL_PATH)
    X_train, X_test, y_train, y_test = get_train_test(load_prepared_data())
    y_pred = model.predict(X_test)

    m = regression_metrics(y_test, y_pred)
    print("=== Metrici pe setul de test ===")
    print(f"MAE  : {m['MAE']:,.0f} $")
    print(f"MSE  : {m['MSE']:,.0f}")
    print(f"RMSE : {m['RMSE']:,.0f} $")
    print(f"R²   : {m['R2']:.3f}")
    print(f"Eroare procentuală mediană: {m['MedAPE_%']:.1f}%")

    print("\nInterpretare:")
    print(f"În medie, prețul estimat diferă de cel real cu ~{m['MAE']:,.0f} $.")
    print(f"Pentru jumătate dintre mașini eroarea este sub {m['MedAPE_%']:.0f}% din preț.")
    print(f"Modelul explică {m['R2']:.0%} din variația prețurilor.")
    print(f"RMSE ({m['RMSE']:,.0f} $) > MAE → există câteva greșeli mari, mai ales la mașinile scumpe.")

    print("\n=== Exemple (preț real vs. estimat) ===")
    print(examples_table(y_test, y_pred).to_string())

    print("\n=== Eroarea pe intervale de preț ===")
    print(error_by_price_range(y_test, y_pred).to_string())


if __name__ == "__main__":
    main()
