"""Одноразовая конвертация весов MXene из joblib (pickle) в родной формат XGBoost (.json).

Зачем: pickle от старой версии xgboost открывается с предупреждением и может сломаться
при смене версии. Формат .json от версии не зависит.

Запуск из корня проекта:  python scripts/convert_mxene_weights.py
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "weights" / "best_model_Eads.joblib"
DST = ROOT / "weights" / "mxene_eads.json"
META = ROOT / "weights" / "mxene_eads_meta.json"
CSV = ROOT / "data" / "mxene_clean.csv"
NON_FEATURES = ["mxene_formula", "adsorbate", "adsorption_site", "Eads", "split"]


def metrics(y, pred):
    return {
        "MAE": round(float(mean_absolute_error(y, pred)), 4),
        "RMSE": round(float(mean_squared_error(y, pred) ** 0.5), 4),
        "R2": round(float(r2_score(y, pred)), 4),
    }


def main():
    for path in (SRC, CSV):
        if not path.exists():
            sys.exit(f"Не найден файл: {path}")

    df = pd.read_csv(CSV)
    features = [c for c in df.columns if c not in NON_FEATURES]
    test = df[df["split"] == "test"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # предупреждение о версии xgboost ожидаемо
        old = joblib.load(SRC)
    if old.n_features_in_ != len(features):
        sys.exit(f"Признаков в модели {old.n_features_in_}, а в CSV {len(features)}: файлы не совпадают.")

    old_pred = old.predict(test[features].values)
    m = metrics(test["Eads"], old_pred)
    print("Старая модель (joblib), тест:", m)

    old.save_model(DST)
    new = XGBRegressor()
    new.load_model(DST)
    new_pred = new.predict(test[features].values)
    diff = float(np.max(np.abs(old_pred - new_pred)))
    print(f"Новая модель (json): макс. расхождение предсказаний = {diff:.2e}")
    if diff > 1e-4:
        sys.exit("Предсказания после конвертации отличаются, не используйте json и пришлите этот вывод.")

    META.write_text(
        json.dumps({"features": features, "target": "Eads", "unit": "eV", "metrics_test": m},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Готово: {DST.name} и {META.name} сохранены в weights/")


if __name__ == "__main__":
    main()