"""Оценка моделей адсорбции: метрики MXene, проверка защиты от битых входов, время MACE.

Запуск из корня проекта:  python scripts/evaluate_models.py
Результат: reports/evaluation_report.md (и краткая сводка в консоли).
"""

import platform
import sys
import time
import warnings
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

REPORT_PATH = ROOT / "reports" / "evaluation_report.md"
MXENE_CSV = ROOT / "data" / "mxene_clean.csv"

MXENE_BAD_INPUTS = {
    "пустой словарь": {},
    "None": None,
    "не словарь": [1, 2],
    "нет газа": {"mxene_formula": "Ti2CO2"},
    "число вместо строки": {"mxene_formula": 5, "adsorbate": "CO"},
    "пустые строки": {"mxene_formula": "", "adsorbate": ""},
    "неизвестный MXene": {"mxene_formula": "Xx9Y3", "adsorbate": "CO"},
    "неизвестный газ": {"mxene_formula": "Ti2CO2", "adsorbate": "H2"},
}
MACE_BAD_INPUTS = {
    "пустой словарь": {},
    "None": None,
    "неизвестный пресет": {"preset": "Fe110_CO"},
    "preset не строка": {"preset": 5},
    "оба варианта сразу": {"preset": "Cu111_CO", "system_path": "a.xyz"},
    "часть путей": {"system_path": "a.xyz"},
    "файл не найден": {"system_path": "no.xyz", "surface_path": "no.xyz", "molecule_path": "no.xyz"},
    "неверный формат": {"system_path": "a.txt", "surface_path": "b.txt", "molecule_path": "c.txt"},
}


def version(package: str) -> str:
    try:
        return metadata.version(package)
    except Exception:
        return "не установлен"


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    ss_res, ss_tot = float((err ** 2).sum()), float(((y_true - y_true.mean()) ** 2).sum())
    return {
        "n": len(y_true),
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "R2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def evaluate_mxene(model) -> dict:
    """Прогоняет все образцы через predict и считает метрики отдельно для train и test."""
    df = pd.read_csv(MXENE_CSV)
    result = {}
    for part in ("test", "train"):
        sub = df[df["split"] == part]
        preds, failed = [], 0
        for formula, gas in zip(sub["mxene_formula"], sub["adsorbate"]):
            r = model.predict({"mxene_formula": formula, "adsorbate": gas})
            if r["status"] == "ok":
                preds.append(r["e_ads_ev"])
            else:
                preds.append(np.nan)
                failed += 1
        ok = ~np.isnan(preds)
        result[part] = {**regression_metrics(sub["Eads"].to_numpy()[ok], np.asarray(preds)[ok]), "failed": failed}
    return result


def check_bad_inputs(model, cases: dict) -> list:
    """Каждый битый вход должен дать status=error с сообщением и без исключения."""
    rows = []
    for title, data in cases.items():
        try:
            r = model.predict(data)
            passed = r["status"] == "error" and bool(r["error_message"]) and r["e_ads_ev"] is None
            rows.append((title, passed, r["error_message"] or "(нет сообщения)"))
        except Exception as exc:  # исключение наружу нарушает контракт
            rows.append((title, False, f"ИСКЛЮЧЕНИЕ: {type(exc).__name__}: {exc}"))
    return rows


def time_mace(model) -> dict:
    """Загрузка весов и расчёт всех пресетов (по два прохода: первый и повторный)."""
    start = time.perf_counter()
    loaded, error = model.warmup()
    load_time = time.perf_counter() - start
    if not loaded:
        return {"available": False, "error": error}
    rows = []
    for preset in model.available_presets():
        first = model.predict({"preset": preset})
        again = model.predict({"preset": preset})
        rows.append({
            "preset": preset,
            "status": first["status"],
            "e_ads": first["e_ads_ev"],
            "interpretation": first["interpretation"],
            "t_first": first["inference_time_sec"],
            "t_repeat": again["inference_time_sec"],
            "error": first["error_message"],
        })
    return {"available": True, "load_time": load_time, "rows": rows}


def fmt(value, digits=4):
    return "—" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.{digits}f}"


def build_report(mxene, bad_mxene, bad_mace, mace) -> str:
    lines = [
        "# Отчёт об оценке моделей адсорбции",
        "",
        f"Дата: {datetime.now():%Y-%m-%d %H:%M}. Python {platform.python_version()}, {platform.system()}, CPU.",
        f"Версии: scikit-learn {version('scikit-learn')}, xgboost {version('xgboost')}, "
        f"torch {version('torch')}, mace-torch {version('mace-torch')}, ase {version('ase')}.",
        "",
        "## 1. MXene Gas Sensor ML Model (XGBoost, E_ads, эВ)",
        "",
        "| Выборка | Образцов | MAE, эВ | RMSE, эВ | R2 | Ошибок predict |",
        "|---|---|---|---|---|---|",
    ]
    for part, title in (("test", "test (разбиение авторов)"), ("train", "train")):
        m = mxene[part]
        lines.append(f"| {title} | {m['n']} | {fmt(m['MAE'])} | {fmt(m['RMSE'])} | {fmt(m['R2'])} | {m['failed']} |")
    lines += [
        "",
        "Оговорки: разбиение авторов случайное по строкам, поэтому MXene из теста встречаются и в train; "
        "модель работает только на готовых образцах датасета (сайт top_M). "
        "Значения Eads в датасете взяты как эталон, их происхождение стоит уточнить.",
        "",
        "## 2. Защита от некорректных входов",
        "",
        "Ожидание: status = error, понятное сообщение, никаких исключений наружу.",
        "",
        "| Модель | Случай | Результат | Сообщение |",
        "|---|---|---|---|",
    ]
    for name, rows in (("MXene", bad_mxene), ("MACE", bad_mace)):
        for title, passed, message in rows:
            lines.append(f"| {name} | {title} | {'OK' if passed else 'СБОЙ'} | {message.replace('|', '/')[:110]} |")
    total = len(bad_mxene) + len(bad_mace)
    passed_total = sum(1 for _, p, _ in bad_mxene + bad_mace if p)
    lines += ["", f"Итого: {passed_total} из {total} проверок пройдено.", "", "## 3. MACE-MP-0 small (single-point, CPU)", ""]
    if not mace["available"]:
        lines.append(f"MACE недоступен: {mace['error']}")
    else:
        lines += [
            f"Загрузка весов: {mace['load_time']:.1f} с (один раз при старте, `warmup()`).",
            "",
            "| Пресет | Статус | E_ads, эВ | Трактовка | 1-й расчёт, с | Повторный, с |",
            "|---|---|---|---|---|---|",
        ]
        for r in mace["rows"]:
            interp = (r["interpretation"] or r["error"] or "").split(":")[0]
            lines.append(f"| {r['preset']} | {r['status']} | {fmt(r['e_ads'], 3)} | {interp} | {fmt(r['t_first'], 2)} | {fmt(r['t_repeat'], 2)} |")
        lines += [
            "",
            "Геометрия не релаксирована, высоты адсорбции заданы приблизительно: значения E_ads ориентировочные "
            "и годятся для демонстрации, а не для количественных выводов.",
        ]
    return "\n".join(lines) + "\n"


def main():
    from models.mace_model import MACEModel
    from models.mxene_model import MXeneModel

    print("Оценка MXene...")
    mxene_model = MXeneModel()
    loaded, error = mxene_model.warmup()
    if not loaded:
        sys.exit(f"MXene недоступна: {error}")
    mxene = evaluate_mxene(mxene_model)
    bad_mxene = check_bad_inputs(mxene_model, MXENE_BAD_INPUTS)

    print("Проверка MACE (загрузка может занять до минуты)...")
    mace_model = MACEModel()
    bad_mace = check_bad_inputs(mace_model, MACE_BAD_INPUTS)
    mace = time_mace(mace_model)

    report = build_report(mxene, bad_mxene, bad_mace, mace)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"Отчёт сохранён: {REPORT_PATH}")
    if not all(p for _, p, _ in bad_mxene + bad_mace):
        sys.exit(1)


if __name__ == "__main__":
    main()
