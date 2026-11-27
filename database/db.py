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
        "name": "MXene Gas Sensor ML",
        "architecture": "Табличный ML (scikit-learn / joblib)",
        "status": "local_cpu",
        "hardware_req": "Локальный CPU (Windows / Linux, < 100 МБ ОЗУ)",
        "input_format": "Табличные признаки системы «MXene + газ» (материал, адсорбат, физико-химические дескрипторы)",
        "output_format": "Энергия адсорбции E_ads (эВ), время инференса и интерпретация прочности связывания",
        "suitable_for": (
            "Быстрый скрининг чувствительности газовых сенсоров на основе MXene "
            "(NH3, NO2, CO, H2S и др.), массовый перебор табличных дескрипторов на CPU."
        ),
        "limitations": (
            "Применима только в рамках домена обучающей выборки MXene (Zenodo); "
            "не учитывает произвольную 3D-геометрию и сложные дефекты поверхности вне таблицы признаков."
        ),
        "metrics_summary": "Быстрый отклик на CPU (< 0.05 с); оценка качества по MAE / RMSE / R² на выборке Zenodo",
    },
    {
        "id": "mace_mp0",
        "name": "MACE-MP-0 (small)",
        "architecture": "Универсальный эквивариантный межатомный потенциал (MLIP, MACE + ASE)",
        "status": "local_cpu",
        "hardware_req": "Локальный CPU (Windows / Linux, от 4 ГБ ОЗУ; поддерживается также GPU)",
        "input_format": "3D-структура атомной системы в формате .xyz или .cif",
        "output_format": "Оценка полной энергии / энергии адсорбции E_ads (эВ) и время single-point расчета",
        "suitable_for": (
            "Универсальный single-point расчет атомных конфигураций «адсорбат + поверхность», "
            "работа с произвольными элементами периодической таблицы из .xyz/.cif файлов."
        ),
        "limitations": (
            "На CPU расчет крупных суперъячеек (> 100–150 атомов) занимает больше времени; "
            "для точной E_ads без отдельного расчёта референсов поверхности и молекулы даётся приближённая оценка / полная энергия комплекса."
        ),
        "metrics_summary": "Универсальный потенциал Materials Project (small); оффлайн single-point расчет на CPU",
    },
    {
        "id": "adsmt",
        "name": "AdsMT",
        "architecture": "Мультимодальный трансформер для адсорбции",
        "status": "catalog_gpu_only",
        "hardware_req": "Linux + NVIDIA GPU (CUDA, от 12–16 ГБ VRAM)",
        "input_format": "Текстовое/структурное описание каталитической поверхности, сайта адсорбции и адсорбата",
        "output_format": "Предсказанная энергия адсорбции E_ads (эВ) и карты внимания",
        "suitable_for": (
            "Высокоточный скрининг адсорбции на сложных каталитических поверхностях и сплавах "
            "при наличии мультимодальных описаний."
        ),
        "limitations": (
            "Требует NVIDIA GPU и Linux-окружение; не запускается локально на Windows + CPU. "
            "По согласованию с куратором запрос GPU в техподдержке ИРИТ вынесен на следующий спринт."
        ),
        "metrics_summary": "Высокая точность на каталитических бенчмарках (требуется GPU-кластер для инференса)",
    },
    {
        "id": "lei_framework",
        "name": "LEI-framework",
        "architecture": "Графовая нейросеть (GNN / PyTorch Geometric) с локальным окружением",
        "status": "catalog_gpu_only",
        "hardware_req": "NVIDIA GPU + PyTorch Geometric (CUDA, от 8–16 ГБ VRAM)",
        "input_format": "Атомный граф комплекса «поверхность + адсорбат» (.cif / .traj / graph)",
        "output_format": "Энергия адсорбции E_ads (эВ) и локальные вклады узлов графа",
        "suitable_for": (
            "Моделирование локальных электронных и геометрических взаимодействий "
            "в активных центрах катализаторов и пористых материалах."
        ),
        "limitations": (
            "Зависит от CUDA-сборки PyTorch Geometric и требует GPU для приемлемого времени расчета. "
            "Запуск на сервере ИРИТ с GPU запланирован на следующий этап проекта."
        ),
        "metrics_summary": "Специализированная GNN-архитектура для локальных взаимодействий (только каталог-советчик)",
    },
    {
        "id": "oc20_fairchem",
        "name": "OC20 / FAIRChem",
        "architecture": "Предобученные GNN-потенциалы (EquiformerV2 / GemNet-OC / PaiNN)",
        "status": "catalog_gpu_only",
        "hardware_req": "Серверный NVIDIA GPU (CUDA, от 16–24+ ГБ VRAM) и крупные веса чекпоинтов",
        "input_format": "Атомные структуры каталитических слэбов и адсорбатов (ASE Atoms / LMDB / .cif)",
        "output_format": "Энергия адсорбции E_ads (эВ), межатомные силы и релаксация геометрии",
        "suitable_for": (
            "Крупномасштабный гетерогенный катализ, релаксация систем «слэб + адсорбат» "
            "в химическом пространстве Open Catalyst 2020 (OC20)."
        ),
        "limitations": (
            "Тяжелые чекпоинты (сотни МБ — гигабайты) и критическая зависимость от серверного GPU. "
            "На текущем этапе используется как рекомендация; развертывание на GPU ИРИТ планируется в следующем спринте."
        ),
        "metrics_summary": "Индустриальный стандарт OC20 для катализа (MAE ~ 0.23–0.35 эВ на GPU-сервере)",
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
                INSERT OR IGNORE INTO models_catalog (
                    id, name, architecture, status, hardware_req,
                    input_format, output_format, suitable_for,
                    limitations, metrics_summary
                ) VALUES (
                    :id, :name, :architecture, :status, :hardware_req,
                    :input_format, :output_format, :suitable_for,
                    :limitations, :metrics_summary
                )
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
