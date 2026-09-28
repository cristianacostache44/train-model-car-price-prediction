"""
config.py - setări comune pentru tot proiectul (căi, constante).
Le țin într-un singur loc ca să nu le repet în fiecare script.
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "cars.csv"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "car_price_model.joblib"
COMPARISON_PATH = MODELS_DIR / "model_comparison.csv"

TARGET = "price_usd"

# Anul „de referință” pentru vârsta mașinii.
# Cel mai nou an din date este 2019, deci anunțurile sunt din ~2019–2020.
# Dacă aș folosi anul curent (2026), toate mașinile ar părea cu 6 ani mai vechi decât erau la momentul anunțului.
REFERENCE_YEAR = 2020

# Împărțirea train/test este aceeași în toate scripturile, ca să compar modelele corect
TEST_SIZE = 0.2
RANDOM_STATE = 42
