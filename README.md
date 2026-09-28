# Car Price Prediction - predicția prețului mașinilor second-hand

Model de **regresie** care estimează prețul de piață (în USD) al unei mașini second-hand pe baza caracteristicilor ei:
marcă, model, an, kilometraj, combustibil, motor, transmisie, tracțiune, segment etc.

> *„Cât ar putea valora această mașină?”* - un preț orientativ care îi ajută pe vânzători să nu ceară nici prea mult, nici prea puțin.

```
VW Golf, 2014, diesel 1.6, 180.000 km, manuală, segment C
→ Preț estimat: 11,992 $   (Golf-uri similare din date: 10.500 – 14.260 $)
```

## Rezultate pe scurt

| | |
|---|---|
| **Date** | 56.244 anunțuri → 55.856 după curățare |
| **Model final** | Gradient Boosting Regressor, antrenat pe log(preț) |
| **MAE** | **~1.023 $** – eroarea medie a estimării |
| **RMSE** | ~2.010 $ |
| **R²** | **0,931** |
| **Eroare procentuală mediană** | ~12,5% |

---

## Structura proiectului

```
car-price-prediction/
├── data/
│   └── cars.csv                    # setul de date original
├── notebooks/
│   ├── 01_eda.ipynb                # analiză exploratorie, curățare, caracteristici noi
│   └── 02_modeling.ipynb           # preprocesare, comparare modele, evaluare, interpretare
├── src/
│   ├── config.py                   # căi și constante comune
│   ├── data_cleaning.py            # curățarea datelor
│   ├── feature_engineering.py      # caracteristici noi (car_age, mileage_per_year...)
│   ├── data_preprocessing.py       # coloane + pipeline de preprocesare (imputer, scaler, encoder)
│   ├── model_training.py           # antrenează și salvează modelul final
│   ├── model_evaluation.py         # MAE, MSE, RMSE, R² + exemple + eroare pe intervale de preț
│   ├── model_comparison.py         # compară 6 variante de modele
│   └── predict_price.py            # (bonus) estimarea prețului unei mașini
├── models/
│   ├── car_price_model.joblib      # modelul final antrenat
│   └── model_comparison.csv        # rezultatele comparației
├── requirements.txt
└── README.md
```

---

## 🚀 Cum rulezi proiectul

### 1. Instalare
```bash
git clone https://github.com/cristianacostache44/car-price-prediction.git
cd car-price-prediction
pip install -r requirements.txt
```

### 2. Rularea pipeline-ului (din folderul principal al proiectului)
```bash
python src/model_comparison.py     # compară modelele (~2 min) → models/model_comparison.csv
python src/model_training.py       # antrenează modelul final (~1 min) → models/car_price_model.joblib
python src/model_evaluation.py     # metrici, exemple, eroare pe intervale de preț
```
Fiecare script de pregătire poate fi rulat și separat, ca verificare:
```bash
python src/data_cleaning.py
python src/feature_engineering.py
python src/data_preprocessing.py
```

### 3. Estimarea prețului unei mașini
```bash
python src/predict_price.py                  # exemplul din enunț (VW Golf 2014)
python src/predict_price.py --interactive    # introduci tu datele mașinii
```

> Modelul salvat e inclus în repozitoriu. Dacă la încărcare apare o eroare de versiune scikit-learn,
> rulează `python src/model_training.py` ca să îl generezi din nou pe calculatorul tău.

### 4. Notebook-urile
```bash
jupyter notebook notebooks/
```
Ambele notebook-uri sunt salvate cu rezultatele, deci pot fi citite direct pe GitHub.

---

## 🔍 Ce am descoperit și ce am decis

### Curățarea datelor (`data_cleaning.py`)
| Problemă | Decizie |
|---|---|
| Coloanele `mileage(kilometers)`, `volume(cm3)`, `priceUSD` | redenumite: `mileage_km`, `volume_cm3`, `price_usd` |
| 87 de rânduri duplicate | eliminate |
| Kilometraj > 1.000.000 km (ex. 9.999.999) | kilometraj **necunoscut** → completat cu mediana |
| Kilometraj < 1.000 km la mașini > 2 ani (2.454 cazuri, probabil scris în mii) | kilometraj **necunoscut** |
| Motor de 16.000–20.000 cm³ (un zero în plus) | împărțit la 10 |
| Mașini electrice fără capacitate motor | capacitate = 0 |
| Preț < 100 $ sau > 100.000 $ | eliminate (greșeli / mașini exotice foarte rare) |
| An < 1980 | eliminate (mașini de colecție, cu alte reguli de preț) |
| `drive_unit`, `segment` lipsă | categoria **„unknown”** (în preprocesare) |

Principiu: **corectez** valorile când pot deduce ce s-a vrut și **șterg** rândul doar când prețul nu poate fi folosit.

### Ingineria caracteristicilor (`feature_engineering.py`)
| Caracteristică | Scop | A ajutat? |
|---|---|---|
| `car_age` | vârsta mașinii (2020 − an) | cea mai importantă (~74%) |
| `brand_model` | ex. `volkswagen_golf` | a treia ca importanță |
| `engine_volume_liters` | capacitatea în litri | a doua ca importanță |
| `mileage_per_year` | uzura pe an | ➖ importanță mică, corelație pozitivă surprinzătoare (explicată în EDA) |
| `is_newer_car` (≥ 2010) | generație nouă | ➖ mică (arborii o deduc din vârstă) |
| `is_high_mileage` (> 300.000 km) | kilometraj mare | ~0, redundantă, candidată la eliminare |

### Preprocesare (`data_preprocessing.py`)
- numerice: `SimpleImputer(median)`, plus `StandardScaler` pentru modelul liniar;
- categorice: `SimpleImputer("unknown")`, apoi `OneHotEncoder` pentru modelul liniar sau `OrdinalEncoder` pentru arbori.
  Cu ~1.000 de modele auto, one-hot ar fi foarte lent pentru arbori;
- ținta este antrenată pe **log(preț)** (`TransformedTargetRegressor`). Prețurile sunt foarte asimetrice,
  iar predicția iese automat înapoi în dolari.

### Compararea modelelor (`model_comparison.py`, același train/test 80/20)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Gradient Boosting (reglat)** | **1.023 $** | **2.010 $** | **0,931** |
| Random Forest | 1.085 $ | 2.318 $ | 0,908 |
| Linear Regression (log preț) | 1.167 $ | 2.487 $ | 0,894 |
| Decision Tree | 1.302 $ | 2.754 $ | 0,870 |
| Gradient Boosting (setări implicite) | 1.409 $ | 2.782 $ | 0,867 |
| Linear Regression (preț brut) – baseline | 1.920 $ | 3.540 $ | 0,785 |

**De ce Gradient Boosting reglat?**
- are cel mai bun rezultat la **toate** metricile;
- fișierul modelului are doar **~6 MB**, față de >50 MB pentru Random Forest. Contează pentru GitHub și pentru o aplicație reală.

### Cum interpretez rezultatul
- **MAE ≈ 1.000 $**: estimarea diferă în medie cu ~1.000 $ de prețul din anunț, la un preț median de 5.350 $.
- Pentru **jumătate dintre mașini**, eroarea este sub **~12,5%** din preț.
- Eroarea în dolari crește cu prețul (~430 $ sub 3.000 $, ~7.400 $ peste 30.000 $), dar în procente rămâne între 11% și 16%.
  Excepție fac mașinile foarte ieftine (< 3.000 $), cu ~28%, unde apar des mașini avariate sau „for parts”.
- Cele mai mari greșeli: **mașini premium scumpe** (Porsche, Mercedes S-Klasse, subestimate) și **anunțuri probabil greșite**.

### Ce am învățat
1. **Transformarea log a țintei** a adus cel mai mare câștig: MAE a scăzut de la 1.920 $ la 1.167 $ cu același algoritm.
2. **Setările contează:** Gradient Boosting cu parametri impliciți a fost printre cele mai slabe modele, iar reglat a fost cel mai bun.
3. **Nu orice caracteristică nouă ajută:** `is_high_mileage` s-a dovedit redundantă.

### Pași următori
- `RandomizedSearchCV` + validare încrucișată pentru parametri;
- interval de preț în loc de o singură valoare (ex. quantile regression);
- date suplimentare: echipare, număr de proprietari, regiune.

---

**Autor:** Cristiana-Raluca Costache · Python, pandas, scikit-learn, matplotlib, seaborn, joblib
