"""Модуль для работы с локальной базой данных SQLite проекта «Зоопарк ИИ»."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "database" / "zoo.db"

MODELS_SEED_DATA: list[dict[str, str]] = [
    {
        "id": "mxene_ml",
        "name": "MXene Gas Sensor ML (XGBoost)",
        "architecture": "Табличный ML (XGBoost / scikit-learn)",
        "status": "local_cpu",
        "hardware_req": "Локальный CPU (Windows / Linux, < 100 МБ ОЗУ)",
        "input_format": (
            "Пара {mxene_formula, adsorbate} из data/mxene_clean.csv (240 MXene × 8 газов: "
            "CH4, CO, H2S, HCHO, NH3, NO, NO2, SO2); 17 дескрипторов подставляются автоматически"
        ),
        "output_format": (
            "Энергия адсорбции E_ads (эВ) для сайта top_M, эталон из датасета, "
            "пометка train/test, время инференса и интерпретация"
        ),
        "suitable_for": (
            "Быстрый скрининг чувствительности газовых сенсоров на основе MXene "
            "(CH4, CO, H2S, HCHO, NH3, NO, NO2, SO2) на локальном CPU."
        ),
        "limitations": (
            "Работает только на готовых образцах из data/mxene_clean.csv (сайт адсорбции top_M); "
            "произвольная формула или новый газ требуют внешнего расчёта дескрипторов (pymatgen/matminer). "
            "На выборке train ошибка занижена — ориентируйтесь на пометку train/test."
        ),
        "metrics_summary": (
            "Локальная модель XGBoost (test): MAE 0.0124 эВ, RMSE 0.0167 эВ, R² = 0.9993 "
            "(случайное разбиение по строкам); ориентир статьи (RFR): MAE 0.32 эВ, R² = 0.95"
        ),
    },
    {
        "id": "mace_mp0",
        "name": "MACE-MP-0 (small)",
        "architecture": "Универсальный эквивариантный межатомный потенциал (MLIP, MACE + ASE)",
        "status": "local_cpu",
        "hardware_req": "Локальный CPU (Windows / Linux, от 4 ГБ ОЗУ; поддерживается также GPU)",
        "input_format": (
            "Имя готового пресета (например, Cu111_CO) или три 3D-структуры (.xyz / .cif): "
            "system_path (поверхность + молекула), surface_path (поверхность), molecule_path (молекула)"
        ),
        "output_format": (
            "E_ads = E(система) − E(поверхность) − E(молекула) в эВ, энергии трёх структур, "
            "число атомов, время single-point расчёта и предупреждения"
        ),
        "suitable_for": (
            "Универсальный single-point расчёт E_ads для произвольных элементов периодической таблицы "
            "по готовым пресетам или пользовательским файлам .xyz/.cif."
        ),
        "limitations": (
            "Single-point без релаксации геометрии на CPU (float64) — значения ориентировочные "
            "(для точных выводов нужен DFT); максимум 300 атомов, мин. расстояние ≥ 0.6 Å. "
            "Веса: weights/mace_mp_small.model."
        ),
        "metrics_summary": (
            "Ориентиры CatBench (с релаксацией): MAE ~0.36 эВ (металлы), 0.181 эВ (MOFs), "
            "0.250 эВ (оксиды); single-point пресета на CPU ~0.5–1.3 с после warmup()"
        ),
    },
    {
        "id": "adsmt",
        "name": "AdsMT",
        "architecture": "Мультимодальный трансформер (AdsGT/GNN + MLP адсорбата + cross-attention)",
        "status": "catalog_gpu_only",
        "hardware_req": "Linux + NVIDIA GPU (обучение: A100; в текущем MVP — каталог-советчик)",
        "input_format": "Периодический граф поверхности + 208 RDKit-дескрипторов адсорбата (без заранее заданного сайта)",
        "output_format": "Предсказанная энергия адсорбции (GMAE / E_ads, эВ) и карты внимания",
        "suitable_for": (
            "Высокоточный скрининг адсорбции на сложных каталитических поверхностях и сплавах, "
            "когда положение адсорбата (сайт) заранее не задано."
        ),
        "limitations": (
            "Валидированный пайплайн авторов требует Linux и NVIDIA GPU (тяжёлые энкодеры вроде "
            "GemNet-OC без GPU непригодны); запрос GPU в техподдержке ИРИТ вынесен на следующий спринт."
        ),
        "metrics_summary": (
            "Тест MAE: 0.095 эВ (FG-GMAE), 0.143 эВ (Alloy-GMAE), 0.389 эВ (OCD-GMAE с transfer learning); "
            "Nature Communications 2025"
        ),
    },
    {
        "id": "lei_framework",
        "name": "LEI-framework (LERN)",
        "architecture": "Дескрипторы локального окружения (Вороной) + RF / ResNet-34 (LERN)",
        "status": "catalog_gpu_only",
        "hardware_req": "NVIDIA GPU + PyTorch Geometric (LERN обучался на RTX 3080; в MVP — советчик)",
        "input_format": "Кристаллическая структура «поверхность + адсорбат» и сайт -> матрица локального окружения 6×11 (3 слоя соседей)",
        "output_format": "Энергия адсорбции E_ads (эВ) по признакам ближайшего окружения",
        "suitable_for": (
            "Предсказание E_ads по локальному окружению места адсорбции: катализ (HER), "
            "газовые сенсоры, захват углерода, хранение энергии."
        ),
        "limitations": (
            "Требует сборки пайплайна графов Вороного / PyTorch Geometric и знания места адсорбции; "
            "нейросеть LERN рассчитана на GPU (RF-ветка — кандидат на перенос на CPU)."
        ),
        "metrics_summary": (
            "RF-ветка: MAE 0.14 эВ (H на 3D-материалах); LERN (ResNet-34): наименьший MAE среди "
            "CGCNN/SchNet/ALIGNN на 3D-выборке; JMI 2024"
        ),
    },
    {
        "id": "oc20_fairchem",
        "name": "OC20 / FAIRChem",
        "architecture": "Предобученные графовые нейросети (SchNet, DimeNet++, GemNet-OC, eSCN)",
        "status": "catalog_gpu_only",
        "hardware_req": "Серверный NVIDIA GPU (CUDA, от 16–24+ ГБ VRAM) и крупные веса чекпоинтов",
        "input_format": "3D-граф каталитических слэбов и адсорбатов (ASE Atoms / LMDB / .cif) для задач S2EF и IS2RE",
        "output_format": "Энергия адсорбции E_ads (эВ), межатомные силы и релаксация геометрии",
        "suitable_for": (
            "Крупномасштабный гетерогенный катализ и релаксация систем «слэб + адсорбат» "
            "в химическом пространстве Open Catalyst 2020 (OC20)."
        ),
        "limitations": (
            "Тяжёлые чекпойнты и критическая зависимость от серверного GPU; "
            "на текущем этапе используется как рекомендация до развёртывания на GPU ИРИТ."
        ),
        "metrics_summary": (
            "Базлайны OC20 IS2RE-direct (test ID): SchNet MAE 0.639 эВ, DimeNet++ MAE 0.562 эВ; "
            "Relaxation-based DimeNet++: 0.512 эВ"
        ),
    },
]


def _get_connection() -> sqlite3.Connection:
    """Создает каталог БД при необходимости и возвращает подключение SQLite."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Инициализирует таблицы SQLite и заполняет каталог 5 моделей без дублирования."""
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS models_catalog (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                architecture TEXT,
                status TEXT,
                hardware_req TEXT,
                input_format TEXT,
                output_format TEXT,
                suitable_for TEXT,
                limitations TEXT,
                metrics_summary TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS runs_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                input_summary TEXT,
                status TEXT,
                e_ads_ev REAL,
                interpretation TEXT,
                inference_time_sec REAL,
                error_message TEXT
            )
            """
        )

        for model in MODELS_SEED_DATA:
            cursor.execute(
                """
                INSERT INTO models_catalog (
                    id, name, architecture, status, hardware_req,
                    input_format, output_format, suitable_for,
                    limitations, metrics_summary
                ) VALUES (
                    :id, :name, :architecture, :status, :hardware_req,
                    :input_format, :output_format, :suitable_for,
                    :limitations, :metrics_summary
                )
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    architecture = excluded.architecture,
                    status = excluded.status,
                    hardware_req = excluded.hardware_req,
                    input_format = excluded.input_format,
                    output_format = excluded.output_format,
                    suitable_for = excluded.suitable_for,
                    limitations = excluded.limitations,
                    metrics_summary = excluded.metrics_summary
                """,
                model,
            )

        conn.commit()
    finally:
        conn.close()


def get_all_models() -> list[dict[str, Any]]:
    """Возвращает список всех моделей из каталога."""
    init_db()
    conn = _get_connection()
    try:
        rows = conn.execute(
            """
            SELECT id, name, architecture, status, hardware_req,
                   input_format, output_format, suitable_for,
                   limitations, metrics_summary
            FROM models_catalog
            ORDER BY CASE status WHEN 'local_cpu' THEN 0 ELSE 1 END, name
            """
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_model_by_id(model_id: str) -> dict[str, Any] | None:
    """Возвращает карточку модели по ее идентификатору или None."""
    init_db()
    conn = _get_connection()
    try:
        row = conn.execute(
            """
            SELECT id, name, architecture, status, hardware_req,
                   input_format, output_format, suitable_for,
                   limitations, metrics_summary
            FROM models_catalog
            WHERE id = ?
            """,
            (model_id,),
        ).fetchone()
        return dict(row) if row is not None else None
    finally:
        conn.close()


def log_run(
    model_id: str,
    model_name: str,
    input_summary: str,
    status: str,
    e_ads_ev: float | None = None,
    interpretation: str = "",
    inference_time_sec: float = 0.0,
    error_message: str | None = None,
) -> int:
    """Записывает результат запуска модели в runs_history и возвращает id записи."""
    init_db()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    normalized_status = "ok" if status == "ok" else "error"
    safe_time = float(inference_time_sec) if inference_time_sec is not None else 0.0

    conn = _get_connection()
    try:
        cursor = conn.execute(
            """
            INSERT INTO runs_history (
                timestamp, model_id, model_name, input_summary,
                status, e_ads_ev, interpretation, inference_time_sec, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                model_id,
                model_name,
                input_summary,
                normalized_status,
                e_ads_ev,
                interpretation,
                safe_time,
                error_message,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)
    finally:
        conn.close()


def get_runs_history(limit: int = 100) -> list[dict[str, Any]]:
    """Возвращает историю запусков (новые записи сверху)."""
    init_db()
    safe_limit = max(1, int(limit))
    conn = _get_connection()
    try:
        rows = conn.execute(
            """
            SELECT id, timestamp, model_id, model_name, input_summary,
                   status, e_ads_ev, interpretation, inference_time_sec, error_message
            FROM runs_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_runs_stats() -> dict[str, Any]:
    """Возвращает агрегированную статистику по таблице runs_history."""
    init_db()
    conn = _get_connection()
    try:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total_runs,
                COALESCE(SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END), 0) AS successful_runs,
                COALESCE(AVG(inference_time_sec), 0.0) AS avg_time
            FROM runs_history
            """
        ).fetchone()
        total_runs = int(row["total_runs"]) if row else 0
        successful_runs = int(row["successful_runs"]) if row else 0
        avg_time = float(row["avg_time"]) if row else 0.0
        success_rate = round((successful_runs / total_runs) * 100.0, 1) if total_runs > 0 else 0.0
        return {
            "total_runs": total_runs,
            "successful_runs": successful_runs,
            "success_rate": success_rate,
            "avg_inference_time_sec": round(avg_time, 4),
        }
    finally:
        conn.close()


def clear_runs_history() -> None:
    """Очищает всю историю запусков в таблице runs_history."""
    init_db()
    conn = _get_connection()
    try:
        conn.execute("DELETE FROM runs_history")
        conn.commit()
    finally:
        conn.close()
