"""MXene Gas Sensor ML Model: предсказание E_ads газа на MXene (XGBoost, табличная модель).

Вход: {"mxene_formula": "Ti2CO2", "adsorbate": "NO2"}.
Признаки (17 дескрипторов) берутся из data/mxene_clean.csv: модель работает на готовых
образцах (240 MXene x 8 газов), для произвольной формулы нужны pymatgen/matminer.
Всё локально, интернет не нужен.
"""

import difflib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from models.base import BaseAdsorptionModel

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = ROOT / "weights" / "mxene_eads.json"
META_PATH = ROOT / "weights" / "mxene_eads_meta.json"
DATA_PATH = ROOT / "data" / "mxene_clean.csv"

NON_FEATURES = ["mxene_formula", "adsorbate", "adsorption_site", "Eads", "split"]


class MXeneModel(BaseAdsorptionModel):
    id = "mxene_xgb"
    name = "MXene Gas Sensor ML (XGBoost)"
    description = (
        "Табличная ML-модель (XGBoost): предсказывает энергию адсорбции газа на MXene "
        "по 17 физико-химическим дескрипторам. Работает на готовых образцах из датасета "
        "(сайт адсорбции top_M)."
    )
    supported_inputs = [
        "mxene_formula: формула MXene из data/mxene_clean.csv, например Ti2CO2",
        "adsorbate: газ из набора CH4, CO, H2S, HCHO, NH3, NO, NO2, SO2",
    ]

    def __init__(self):
        super().__init__()
        self._model = None
        self._table = None  # данные образцов, читаются лениво
        self.metrics = {}
        try:  # метрики нужны интерфейсу до первого расчёта; при ошибке просто пусто
            self.metrics = json.loads(META_PATH.read_text(encoding="utf-8")).get("metrics_test", {})
        except Exception:
            pass

    # ---- данные ------------------------------------------------------------

    def _load_table(self):
        """Читает CSV один раз. При проблеме кидает исключение с понятным текстом."""
        if self._table is not None:
            return
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"не найден файл данных {DATA_PATH.name} в папке data/")
        df = pd.read_csv(DATA_PATH)
        missing = [c for c in ("mxene_formula", "adsorbate", "Eads", "split") if c not in df.columns]
        if missing:
            raise ValueError(f"в {DATA_PATH.name} нет колонок: {', '.join(missing)}")
        self._table = {
            "df": df,
            "formulas": {f.casefold(): f for f in df["mxene_formula"].unique()},
            "gases": {g.casefold(): g for g in df["adsorbate"].unique()},
            "index": {(f.casefold(), g.casefold()): i
                      for i, (f, g) in enumerate(zip(df["mxene_formula"], df["adsorbate"]))},
        }

    # ---- контракт ----------------------------------------------------------

    def validate_input(self, input_data: dict) -> tuple:
        errors = []
        for key in ("mxene_formula", "adsorbate"):
            value = input_data.get(key)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"укажите «{key}» непустой строкой")
        if errors:
            return False, errors

        try:
            self._load_table()
        except Exception as exc:
            return False, [f"не удалось прочитать данные: {exc}"]

        formula = input_data["mxene_formula"].strip()
        gas = input_data["adsorbate"].strip()
        if formula.casefold() not in self._table["formulas"]:
            hint = difflib.get_close_matches(formula, list(self._table["formulas"].values()), n=3, cutoff=0.6)
            errors.append(f"MXene «{formula}» нет в датасете" + (f". Возможно: {', '.join(hint)}" if hint else ""))
        if gas.casefold() not in self._table["gases"]:
            errors.append(f"газ «{gas}» не поддерживается. Доступны: {', '.join(sorted(self._table['gases'].values()))}")
        if errors:
            return False, errors
        if (formula.casefold(), gas.casefold()) not in self._table["index"]:
            return False, [f"для пары {formula} + {gas} нет данных в датасете"]
        return True, []

    def load(self) -> None:
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise RuntimeError("не установлен пакет xgboost (pip install xgboost)")
        for path in (WEIGHTS_PATH, META_PATH):
            if not path.exists():
                raise FileNotFoundError(
                    f"не найден файл {path.name} в папке weights/ "
                    "(запустите python scripts/convert_mxene_weights.py)"
                )
        self._load_table()
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        features = meta["features"]
        absent = [c for c in features if c not in self._table["df"].columns]
        if absent:
            raise ValueError(f"в данных нет признаков модели: {', '.join(absent)}")
        model = XGBRegressor()
        model.load_model(WEIGHTS_PATH)
        self._features = features
        self._model = model
        self.metrics = meta.get("metrics_test", {})

    def _predict_impl(self, input_data: dict) -> dict:
        key = (input_data["mxene_formula"].strip().casefold(), input_data["adsorbate"].strip().casefold())
        row = self._table["df"].iloc[self._table["index"][key]]
        x = row[self._features].to_numpy(dtype=float).reshape(1, -1)
        if not np.isfinite(x).all():
            raise ValueError("в признаках образца есть пропуски")
        e_ads = float(self._model.predict(x)[0])
        reference = float(row["Eads"])
        return {
            "e_ads_ev": e_ads,
            "details": {
                "mxene_formula": row["mxene_formula"],
                "adsorbate": row["adsorbate"],
                "adsorption_site": row.get("adsorption_site", "top_M"),
                "reference_e_ads_ev": reference,
                "abs_error_vs_reference_ev": round(abs(e_ads - reference), 4),
                "sample_split": row["split"],  # train/test: на train ошибка оптимистична
                "model_metrics_test": self.metrics,
            },
            "warnings": [],
        }
