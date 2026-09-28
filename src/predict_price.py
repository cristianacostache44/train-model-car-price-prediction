"""
predict_price.py - estimează prețul unei mașini cu modelul salvat (bonus, pentru testare rapidă)

Rulare:
    python src/predict_price.py              → exemplul din enunț (VW Golf 2014)
    python src/predict_price.py --interactive → introduci tu datele mașinii
"""
import sys

import joblib
import pandas as pd

from config import MODEL_PATH
from data_cleaning import clean_data
from data_preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from feature_engineering import add_features

EXAMPLE_CAR = {
    "make": "volkswagen", "model": "golf", "year": 2014, "condition": "with mileage",
    "mileage(kilometers)": 180000, "fuel_type": "diesel", "volume(cm3)": 1600,
    "color": "black", "transmission": "mechanics", "drive_unit": "front-wheel drive", "segment": "C",
}

QUESTIONS = [
    ("make", "Marca (ex. volkswagen)", str),
    ("model", "Modelul (ex. golf)", str),
    ("year", "Anul fabricației", int),
    ("condition", "Starea (with mileage / with damage / for parts)", str),
    ("mileage(kilometers)", "Kilometraj", float),
    ("fuel_type", "Combustibil (petrol / diesel / electrocar)", str),
    ("volume(cm3)", "Capacitate motor în cm³ (ex. 1600)", float),
    ("color", "Culoarea", str),
    ("transmission", "Transmisia (mechanics / auto)", str),
    ("drive_unit", "Tracțiunea (front-wheel drive / rear drive / all-wheel drive / part-time four-wheel drive)", str),
    ("segment", "Segmentul (A-F, J, M, S)", str),
]


def predict_price(model, car: dict) -> float:
    """Aceeași curățare și aceleași caracteristici ca la antrenare, apoi predicția."""
    df = pd.DataFrame([car])
    df = add_features(clean_data(df, training=False))
    return float(model.predict(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[0])


def ask_car():
    car = {}
    for key, question, cast in QUESTIONS:
        while True:
            answer = input(f"{question}: ").strip()
            try:
                car[key] = cast(answer)
                break
            except ValueError:
                print("  Valoare invalidă, încearcă din nou.")
    return car


def main():
    if not MODEL_PATH.exists():
        sys.exit("Modelul nu există. Rulează mai întâi:  python src/model_training.py")
    model = joblib.load(MODEL_PATH)

    if "--interactive" in sys.argv:
        car = ask_car()
    else:
        car = EXAMPLE_CAR
        print("Exemplul din enunț:")
        for k, v in car.items():
            print(f"  {k}: {v}")

    price = predict_price(model, car)
    print(f"\nPreț estimat: {price:,.0f} $")


if __name__ == "__main__":
    main()
