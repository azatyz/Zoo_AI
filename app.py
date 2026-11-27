"""Веб-приложение «Зоопарк ИИ — Адсорбция» (Streamlit + SQLite) для Демо 2."""

from __future__ import annotations

import importlib
import math
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd
import streamlit as st

from database.db import (
    clear_runs_history,
    get_all_models,
    get_runs_history,
    get_runs_stats,
    init_db,
    log_run,
)


PROJECT_ROOT = Path(__file__).resolve().parent
MXENE_DATA_PATH = PROJECT_ROOT / "data" / "mxene_clean.csv"
STRUCTURES_DIR = PROJECT_ROOT / "data" / "structures"
BENCHMARKS_PATH = PROJECT_ROOT / "data" / "test_cases" / "benchmarks.csv"

ALLOWED_STRUCTURE_EXTS = {".xyz", ".cif"}
TARGET_COL_CANDIDATES = {
    "e_ads",
    "e_ads_ev",
    "eads",
    "eads_ev",
    "adsorption_energy",
    "adsorption_energy_ev",
    "target",
    "y",
}

DEFAULT_MXENE_TEXT_FIELDS: list[dict[str, str]] = [
    {
        "key": "mxene",
        "label": "Материал MXene",
        "default": "Ti3C2O2",
        "help": "Например: Ti3C2O2, Ti3C2(OH)2, Ti2CO2, V2CO2, Nb2CO2.",
    },
    {
        "key": "gas",
        "label": "Газ-адсорбат",
        "default": "NH3",
        "help": "Например: NH3, NO2, CO, CO2, H2S, SO2, CH4.",
    },
    {
        "key": "site",
        "label": "Сайт / терминация поверхности",
        "default": "O-terminated",
        "help": "Например: O-terminated, OH-terminated, F-terminated, top, hollow.",
    },
]

DEFAULT_MXENE_NUMERIC_FIELDS: list[dict[str, Any]] = [
    {
        "key": "distance_a",
        "label": "Расстояние газ–поверхность (Å)",
        "default": "2.35",
        "min": 0.5,
        "max": 15.0,
        "help": "Допустимый диапазон: от 0.5 до 15.0 Å.",
    },
    {
        "key": "charge_transfer_e",
        "label": "Перенос заряда (e)",
        "default": "-0.18",
        "min": -5.0,
        "max": 5.0,
        "help": "Допустимый диапазон: от -5.0 до 5.0 e.",
    },
    {
        "key": "band_gap_ev",
        "label": "Ширина запрещенной зоны (эВ)",
        "default": "0.45",
        "min": 0.0,
        "max": 15.0,
        "help": "Допустимый диапазон: от 0.0 до 15.0 эВ.",
    },
    {
        "key": "work_function_ev",
        "label": "Работа выхода (эВ)",
        "default": "4.85",
        "min": 0.5,
        "max": 12.0,
        "help": "Допустимый диапазон: от 0.5 до 12.0 эВ.",
    },
    {
        "key": "temperature_k",
        "label": "Температура (K)",
        "default": "300.0",
        "min": 1.0,
        "max": 3000.0,
        "help": "Допустимый диапазон: от 1.0 до 3000.0 K.",
    },
]


def _resolve_model_class(module: Any, candidate_names: tuple[str, ...]) -> type | None:
    """Находит класс модели с интерфейсом validate_input/predict в модуле."""
    for name in candidate_names:
        obj = getattr(module, name, None)
        if isinstance(obj, type) and callable(getattr(obj, "predict", None)):
            return obj

    for obj in vars(module).values():
        if (
            isinstance(obj, type)
            and getattr(obj, "__module__", "") == module.__name__
            and callable(getattr(obj, "validate_input", None))
            and callable(getattr(obj, "predict", None))
        ):
            return obj
    return None


@st.cache_resource
def load_ml_model(
    module_path: str,
    candidate_names: tuple[str, ...],
) -> dict[str, Any]:
    """Безопасно загружает и кэширует локальную ML-модель."""
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        missing_name = getattr(exc, "name", "") or ""
        if missing_name in {module_path, "models", "models.base_model"} or not missing_name:
            return {
                "available": False,
                "status": "pending_merge",
                "model": None,
                "message": "Модуль ML ожидает слияния ветки `feat/adsorption-models`",
            }
        return {
            "available": False,
            "status": "dep_error",
            "model": None,
            "message": f"Не удалось загрузить зависимость ML-модуля ({missing_name}): {exc}",
        }
    except Exception as exc:
        return {
            "available": False,
            "status": "load_error",
            "model": None,
            "message": f"Ошибка при импорте модуля `{module_path}`: {exc}",
        }

    model_cls = _resolve_model_class(module, candidate_names)
    if model_cls is None:
        return {
            "available": False,
            "status": "load_error",
            "model": None,
            "message": f"В модуле `{module_path}` не найден класс модели с методами validate_input/predict.",
        }

    try:
        instance = model_cls()
    except Exception as exc:
        return {
            "available": False,
            "status": "load_error",
            "model": None,
            "message": f"Ошибка инициализации модели `{model_cls.__name__}`: {exc}",
        }

    return {
        "available": True,
        "status": "connected",
        "model": instance,
        "message": "Модуль ML подключен и готов к локальному расчету на CPU.",
    }


@st.cache_data
def load_csv_safe(file_path_str: str) -> tuple[pd.DataFrame | None, str | None]:
    """Безопасно читает CSV-файл и возвращает DataFrame или текст ошибки."""
    path = Path(file_path_str)
    if not path.exists():
        return None, None
    try:
        df = pd.read_csv(path)
        return df, None
    except Exception as exc:
        return None, f"Не удалось прочитать файл `{path.name}`: {exc}"


def list_structure_presets(structures_dir: Path) -> list[Path]:
    """Возвращает отсортированный список .xyz и .cif файлов из папки пресетов."""
    if not structures_dir.exists() or not structures_dir.is_dir():
        return []
    files = [
        p
        for p in structures_dir.iterdir()
        if p.is_file() and p.suffix.lower() in ALLOWED_STRUCTURE_EXTS
    ]
    return sorted(files, key=lambda p: p.name.lower())


def _parse_float_strict(raw_value: str) -> tuple[float | None, str | None]:
    """Преобразует строку в конечное число float или возвращает причину ошибки."""
    text = str(raw_value).strip().replace(",", ".")
    if not text:
        return None, "значение не может быть пустым"
    try:
        num = float(text)
    except ValueError:
        return None, f"значение «{raw_value}» не является числом"
    if not math.isfinite(num):
        return None, f"значение «{raw_value}» не является конечным числом"
    return num, None


def validate_mxene_form(
    text_values: dict[str, str],
    numeric_values: dict[str, str],
    numeric_bounds: dict[str, tuple[float, float, str]],
) -> tuple[bool, dict[str, Any], list[str]]:
    """Проверяет текстовые и числовые поля формы MXene."""
    errors: list[str] = []
    parsed: dict[str, Any] = {}

    for key, val in text_values.items():
        cleaned = str(val).strip()
        if not cleaned:
            errors.append(f"Поле «{key}» не может быть пустым.")
        else:
            parsed[key] = cleaned

    for key, raw_val in numeric_values.items():
        min_val, max_val, label = numeric_bounds[key]
        num, parse_err = _parse_float_strict(raw_val)
        if parse_err is not None:
            errors.append(f"Поле «{label}»: {parse_err}.")
            continue
        assert num is not None
        if num < min_val or num > max_val:
            errors.append(
                f"Поле «{label}»: значение {num:g} выходит за допустимые границы [{min_val:g}; {max_val:g}]."
            )
            continue
        parsed[key] = num

    return len(errors) == 0, parsed, errors


def validate_structure_bytes(filename: str, raw_bytes: bytes) -> tuple[bool, str, list[str]]:
    """Проверяет расширение, непустоту и базовую корректность .xyz/.cif структуры."""
    errors: list[str] = []
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_STRUCTURE_EXTS:
        errors.append(
            f"Неподдерживаемое расширение файла «{suffix or 'без расширения'}». "
            "Загрузите структуру в формате .xyz или .cif."
        )
        return False, "", errors

    if not raw_bytes or len(raw_bytes.strip()) == 0:
        errors.append(f"Файл «{filename}» пуст.")
        return False, "", errors

    try:
        content = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        errors.append(
            f"Файл «{filename}» не является корректным текстовым файлом структуры (ошибка UTF-8)."
        )
        return False, "", errors

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        errors.append(f"Файл «{filename}» не содержит данных.")
        return False, "", errors

    if suffix == ".xyz":
        try:
            atom_count = int(lines[0])
        except ValueError:
            errors.append(
                "Некорректный формат .xyz: первая непустая строка должна содержать целое число атомов."
            )
            return False, content, errors
        if atom_count <= 0:
            errors.append("Некорректный формат .xyz: число атомов должно быть положительным.")
            return False, content, errors

        non_empty_all = content.splitlines()
        if len(non_empty_all) < atom_count + 2:
            errors.append(
                f"Некорректный формат .xyz: указано {atom_count} атомов, но строк в файле недостаточно."
            )
            return False, content, errors

        coord_lines = non_empty_all[2 : 2 + atom_count]
        for idx, coord_line in enumerate(coord_lines, start=3):
            parts = coord_line.split()
            if len(parts) < 4:
                errors.append(
                    f"Некорректный формат .xyz (строка {idx}): ожидаются символ элемента и 3 координаты X Y Z."
                )
                return False, content, errors
            for coord_str in parts[1:4]:
                _, coord_err = _parse_float_strict(coord_str)
                if coord_err is not None:
                    errors.append(
                        f"Некорректный формат .xyz (строка {idx}): координата «{coord_str}» не является числом."
                    )
                    return False, content, errors

    elif suffix == ".cif":
        lower_content = content.lower()
        has_cif_markers = (
            "data_" in lower_content
            or "_cell_length_" in lower_content
            or "_atom_site_" in lower_content
        )
        if not has_cif_markers:
            errors.append(
                "Некорректный формат .cif: не найдены обязательные заголовки (`data_`, `_cell_length_` или `_atom_site_`)."
            )
            return False, content, errors

    return True, content, errors


def _format_status_badge(model: dict[str, Any], ml_state: dict[str, Any] | None) -> str:
    """Формирует текстовый индикатор статуса модели для боковой панели и карточек."""
    if model["status"] == "local_cpu":
        if ml_state and ml_state.get("available"):
            return "🟢 Локально CPU — подключена"
        return "🟡 Локально CPU — ожидает ML-модуль"
    return "🔴 Каталог-советчик — нужен GPU"


def _render_sidebar(
    models: list[dict[str, Any]],
    ml_states: dict[str, dict[str, Any]],
) -> None:
    """Отрисовывает боковую панель со статусами моделей и памяткой по E_ads."""
    with st.sidebar:
        st.header("🧪 Зоопарк ИИ — Адсорбция")
        st.caption(
            "Учебный проект УрФУ для ООО «ЦИФРОВЫЕ РЕШЕНИЯ». "
            "Единое окно для подбора ML-модели и локального расчета энергии адсорбции на CPU."
        )

        st.divider()
        st.subheader("Статус моделей")
        for model in models:
            badge = _format_status_badge(model, ml_states.get(model["id"]))
            st.markdown(f"**{model['name']}**\n\n{badge}")

        st.divider()
        st.subheader("Памятка по $E_{\\text{ads}}$")
        st.markdown(
            """
- **$E_{\\text{ads}}$ (эВ)** — энергия адсорбции молекулы на поверхности.
- **$E_{\\text{ads}} < 0$** — выгодная (экзотермическая) адсорбция: чем более отрицательное значение, тем прочнее связь.
- **$-0.1 \\dots -0.5$ эВ** — слабая физическая адсорбция (часто оптимальна для обратимых газовых сенсоров).
- **$< -0.8$ эВ** — сильная хемосорбция (высокая прочность связывания / катализ, но медленная десорбция).
- **$E_{\\text{ads}} \\ge 0$** — энергетически невыгодная конфигурация.
            """.strip()
        )


def _is_recommended_for_task(model_id: str, task_choice: str) -> bool:
    """Определяет, рекомендуется ли модель под выбранный сценарий в советчике."""
    if task_choice == "Газовый сенсор на MXene":
        return model_id == "mxene_ml"
    if task_choice == "Атомная структура (.xyz/.cif)":
        return model_id == "mace_mp0"
    if task_choice == "Крупный катализ и GPU-модели":
        return model_id in {"adsmt", "lei_framework", "oc20_fairchem"}
    return False


def _render_advisor_tab(
    models: list[dict[str, Any]],
    ml_states: dict[str, dict[str, Any]],
) -> None:
    """Вкладка 1: Подбор модели (Советчик)."""
    st.subheader("Подбор модели под задачу и доступное железо")
    st.caption(
        "Мастер помогает выбрать модель из «зоопарка» по типу входных данных и вычислительным ресурсам."
    )

    col_task, col_hw = st.columns([2, 1])
    with col_task:
        task_choice = st.selectbox(
            "Тип вашей задачи",
            options=[
                "Все задачи",
                "Газовый сенсор на MXene",
                "Атомная структура (.xyz/.cif)",
                "Крупный катализ и GPU-модели",
            ],
        )
    with col_hw:
        hw_filter = st.radio(
            "Фильтр по ресурсам",
            options=[
                "Показать все 5 моделей",
                "Только доступные локально на CPU",
            ],
        )

    filtered_models = models
    if hw_filter == "Только доступные локально на CPU":
        filtered_models = [m for m in filtered_models if m["status"] == "local_cpu"]

    if task_choice != "Все задачи":
        recommended_first = sorted(
            filtered_models,
            key=lambda m: 0 if _is_recommended_for_task(m["id"], task_choice) else 1,
        )
        filtered_models = recommended_first

    if not filtered_models:
        st.info("По выбранным фильтрам модели не найдены.")
        return

    for model in filtered_models:
        ml_state = ml_states.get(model["id"])
        badge = _format_status_badge(model, ml_state)
        is_rec = _is_recommended_for_task(model["id"], task_choice)

        with st.container(border=True):
            header_cols = st.columns([3, 2])
            with header_cols[0]:
                title_suffix = " — ⭐ Рекомендовано под задачу" if is_rec else ""
                st.markdown(f"### {model['name']}{title_suffix}")
                st.caption(f"ID: `{model['id']}` | Архитектура: {model['architecture']}")
            with header_cols[1]:
                st.markdown(f"**Статус:** {badge}")
                st.caption(f"Железо: {model['hardware_req']}")

            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.markdown(f"**Формат входа:** {model['input_format']}")
                st.markdown(f"**Формат выхода:** {model['output_format']}")
                st.markdown(f"**Метрики / ориентиры:** {model['metrics_summary']}")
            with info_col2:
                st.markdown(f"**Для чего подходит:** {model['suitable_for']}")
                st.markdown(f"**Ограничения:** {model['limitations']}")

            if model["status"] == "catalog_gpu_only":
                st.info(
                    "Режим каталога-советчика: запуск требует NVIDIA GPU и недоступен локально на CPU. "
                    "По согласованию с куратором запрос GPU в техподдержке ИРИТ запланирован на следующий спринт."
                )
            elif ml_state and not ml_state.get("available"):
                st.warning(ml_state["message"])


def _render_prediction_output(result: dict[str, Any]) -> None:
    """Отображает результат успешного расчета ML-модели."""
    e_ads = result.get("e_ads_ev")
    elapsed = float(result.get("inference_time_sec") or 0.0)
    interpretation = result.get("interpretation") or "Интерпретация не предоставлена моделью."
    warnings_list = result.get("warnings") or []
    details = result.get("details") or {}

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        val_str = f"{float(e_ads):.4f} эВ" if e_ads is not None else "—"
        st.metric("Энергия адсорбции $E_{\\text{ads}}$", val_str)
    with m_col2:
        st.metric("Время расчета", f"{elapsed:.4f} сек")

    st.success(f"**Физическая интерпретация:** {interpretation}")

    for warn in warnings_list:
        st.warning(str(warn))

    if details:
        with st.expander("Подробности расчета (details)"):
            st.json(details)


def _apply_mxene_sample_to_state(
    sample_row: pd.Series,
    feature_cols: list[str],
    numeric_cols: set[str],
) -> None:
    """Записывает значения выбранного образца из mxene_clean.csv в session_state."""
    for col in feature_cols:
        val = sample_row[col]
        if pd.isna(val):
            st.session_state[f"mxene_csv_{col}"] = ""
        elif col in numeric_cols:
            st.session_state[f"mxene_csv_{col}"] = f"{val}"
        else:
            st.session_state[f"mxene_csv_{col}"] = str(val)


def _render_mxene_calculator(ml_state: dict[str, Any]) -> None:
    """Отрисовывает форму расчета для модели MXene Gas Sensor (CPU)."""
    st.markdown("#### MXene Gas Sensor ML (локальный CPU)")
    if ml_state["available"]:
        st.success(ml_state["message"])
    else:
        st.warning(ml_state["message"])

    df_mxene, csv_err = load_csv_safe(str(MXENE_DATA_PATH))
    if csv_err:
        st.error(csv_err)

    if df_mxene is not None and not df_mxene.empty:
        target_cols = [c for c in df_mxene.columns if c.strip().lower() in TARGET_COL_CANDIDATES]
        feature_cols = [c for c in df_mxene.columns if c not in target_cols]
        numeric_cols = {
            c for c in feature_cols if pd.api.types.is_numeric_dtype(df_mxene[c])
        }

        label_cols = [c for c in feature_cols if c not in numeric_cols][:2]
        sample_options = ["— Ввести вручную —"]
        for idx, row in df_mxene.head(500).iterrows():
            desc = " | ".join(f"{c}={row[c]}" for c in label_cols if pd.notna(row[c]))
            sample_options.append(f"Строка #{idx}" + (f" ({desc})" if desc else ""))

        selected_sample = st.selectbox(
            "Выбор готового образца из `data/mxene_clean.csv` (автоподстановка параметров)",
            options=sample_options,
            key="mxene_sample_selector",
        )

        if selected_sample != "— Ввести вручную —":
            selected_idx = int(selected_sample.split("#")[1].split()[0])
            sample_row = df_mxene.loc[selected_idx]
            if st.session_state.get("mxene_last_applied_sample") != selected_sample:
                _apply_mxene_sample_to_state(sample_row, feature_cols, numeric_cols)
                st.session_state["mxene_last_applied_sample"] = selected_sample
            if target_cols:
                ref_parts = [
                    f"{tc}: **{sample_row[tc]}**"
                    for tc in target_cols
                    if pd.notna(sample_row[tc])
                ]
                if ref_parts:
                    st.caption("Эталонное значение в датасете: " + ", ".join(ref_parts))
        else:
            st.session_state["mxene_last_applied_sample"] = None

        text_values: dict[str, str] = {}
        numeric_values: dict[str, str] = {}
        numeric_bounds: dict[str, tuple[float, float, str]] = {}

        cols = st.columns(2)
        for idx, col_name in enumerate(feature_cols):
            state_key = f"mxene_csv_{col_name}"
            if state_key not in st.session_state:
                first_valid = df_mxene[col_name].dropna()
                st.session_state[state_key] = str(first_valid.iloc[0]) if not first_valid.empty else ""

            with cols[idx % 2]:
                if col_name in numeric_cols:
                    series_num = pd.to_numeric(df_mxene[col_name], errors="coerce").dropna()
                    if not series_num.empty:
                        col_min = float(series_num.min())
                        col_max = float(series_num.max())
                        span = max(abs(col_max - col_min), 1.0)
                        bound_min = col_min - span
                        bound_max = col_max + span
                    else:
                        bound_min, bound_max = -1e6, 1e6
                    raw = st.text_input(
                        f"{col_name}",
                        key=state_key,
                        help=f"Числовое поле (ориентир: от {bound_min:.3g} до {bound_max:.3g}).",
                    )
                    numeric_values[col_name] = raw
                    numeric_bounds[col_name] = (bound_min, bound_max, col_name)
                else:
                    raw = st.text_input(f"{col_name}", key=state_key)
                    text_values[col_name] = raw
    else:
        st.info(
            "Файл `data/mxene_clean.csv` пока отсутствует — используется ручной ввод параметров."
        )
        text_values = {}
        numeric_values = {}
        numeric_bounds = {}

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("**Категориальные параметры**")
            for field in DEFAULT_MXENE_TEXT_FIELDS:
                text_values[field["label"]] = st.text_input(
                    field["label"],
                    value=field["default"],
                    help=field["help"],
                    key=f"mxene_manual_{field['key']}",
                )
        with col_right:
            st.markdown("**Числовые дескрипторы**")
            for field in DEFAULT_MXENE_NUMERIC_FIELDS:
                val_str = st.text_input(
                    field["label"],
                    value=field["default"],
                    help=field["help"],
                    key=f"mxene_manual_{field['key']}",
                )
                numeric_values[field["key"]] = val_str
                numeric_bounds[field["key"]] = (
                    float(field["min"]),
                    float(field["max"]),
                    field["label"],
                )

    if st.button("Рассчитать E_ads (MXene)", type="primary"):
        is_valid, parsed_data, val_errors = validate_mxene_form(
            text_values=text_values,
            numeric_values=numeric_values,
            numeric_bounds=numeric_bounds,
        )
        summary_parts = [f"{k}={v}" for k, v in list(text_values.items())[:2]]
        summary_parts += [f"{k}={v}" for k, v in list(numeric_values.items())[:3]]
        input_summary = "; ".join(summary_parts) if summary_parts else "MXene параметры"

        if not is_valid:
            err_text = " ".join(val_errors)
            for err in val_errors:
                st.error(err)
            log_run(
                model_id="mxene_ml",
                model_name="MXene Gas Sensor ML",
                input_summary=input_summary,
                status="error",
                inference_time_sec=0.0,
                error_message=err_text,
            )
            return

        if not ml_state["available"]:
            msg = (
                "Входные параметры прошли проверку, но расчет невозможен: "
                f"{ml_state['message']}"
            )
            st.warning(msg)
            log_run(
                model_id="mxene_ml",
                model_name="MXene Gas Sensor ML",
                input_summary=input_summary,
                status="error",
                inference_time_sec=0.0,
                error_message=ml_state["message"],
            )
            return

        model = ml_state["model"]
        try:
            ok, model_errors = model.validate_input(parsed_data)
            if not ok:
                err_text = "; ".join(model_errors) if model_errors else "Ошибка валидации входных данных моделью."
                st.error(err_text)
                log_run(
                    model_id="mxene_ml",
                    model_name="MXene Gas Sensor ML",
                    input_summary=input_summary,
                    status="error",
                    inference_time_sec=0.0,
                    error_message=err_text,
                )
                return

            with st.spinner("Выполняется локальный расчет MXene на CPU..."):
                result = model.predict(parsed_data)
        except Exception as exc:
            err_text = f"Внутренняя ошибка при запуске MXene-модели: {exc}"
            st.error(err_text)
            log_run(
                model_id="mxene_ml",
                model_name="MXene Gas Sensor ML",
                input_summary=input_summary,
                status="error",
                inference_time_sec=0.0,
                error_message=err_text,
            )
            return

        if result.get("status") == "ok":
            _render_prediction_output(result)
            log_run(
                model_id="mxene_ml",
                model_name="MXene Gas Sensor ML",
                input_summary=input_summary,
                status="ok",
                e_ads_ev=result.get("e_ads_ev"),
                interpretation=result.get("interpretation", ""),
                inference_time_sec=float(result.get("inference_time_sec") or 0.0),
                error_message=None,
            )
        else:
            err_text = result.get("error_message") or "Модель вернула ошибку без описания."
            st.error(err_text)
            log_run(
                model_id="mxene_ml",
                model_name="MXene Gas Sensor ML",
                input_summary=input_summary,
                status="error",
                inference_time_sec=float(result.get("inference_time_sec") or 0.0),
                error_message=err_text,
            )


def _render_mace_calculator(ml_state: dict[str, Any]) -> None:
    """Отрисовывает форму расчета для модели MACE-MP-0 (small, CPU)."""
    st.markdown("#### MACE-MP-0 (small, локальный CPU)")
    if ml_state["available"]:
        st.success(ml_state["message"])
    else:
        st.warning(ml_state["message"])

    presets = list_structure_presets(STRUCTURES_DIR)
    source_mode = st.radio(
        "Источник атомной структуры",
        options=[
            "Готовый пресет из data/structures/",
            "Загрузить свой файл (.xyz / .cif)",
        ],
        horizontal=True,
        key="mace_source_mode",
    )

    selected_preset: Path | None = None
    uploaded_file = None

    if source_mode == "Готовый пресет из data/structures/":
        if not presets:
            st.info(
                "В папке `data/structures/` пока нет файлов `.xyz` или `.cif`. "
                "Переключитесь на загрузку своего файла."
            )
        else:
            preset_names = [p.name for p in presets]
            chosen_name = st.selectbox(
                "Выберите пресет структуры",
                options=preset_names,
                key="mace_preset_select",
            )
            selected_preset = next(p for p in presets if p.name == chosen_name)
            try:
                preview_lines = selected_preset.read_text(encoding="utf-8").splitlines()[:15]
                with st.expander("Предпросмотр файла структуры (первые 15 строк)"):
                    st.code("\n".join(preview_lines), language="text")
            except Exception as exc:
                st.warning(f"Не удалось прочитать превью пресета: {exc}")
    else:
        uploaded_file = st.file_uploader(
            "Загрузите файл атомной структуры (ожидаются форматы .xyz или .cif)",
            key="mace_file_uploader",
        )

    if st.button("Рассчитать E_ads (MACE-MP-0)", type="primary"):
        if source_mode == "Готовый пресет из data/structures/":
            if selected_preset is None:
                err_msg = "Не выбран файл пресета из `data/structures/`."
                st.error(err_msg)
                log_run(
                    model_id="mace_mp0",
                    model_name="MACE-MP-0 (small)",
                    input_summary="Пресет не выбран",
                    status="error",
                    inference_time_sec=0.0,
                    error_message=err_msg,
                )
                return
            filename = selected_preset.name
            try:
                raw_bytes = selected_preset.read_bytes()
            except Exception as exc:
                err_msg = f"Ошибка чтения пресета «{filename}»: {exc}"
                st.error(err_msg)
                log_run(
                    model_id="mace_mp0",
                    model_name="MACE-MP-0 (small)",
                    input_summary=f"Пресет: {filename}",
                    status="error",
                    inference_time_sec=0.0,
                    error_message=err_msg,
                )
                return
            source_label = f"Пресет: {filename}"
        else:
            if uploaded_file is None:
                err_msg = "Загрузите файл структуры (.xyz или .cif) перед запуском расчета."
                st.error(err_msg)
                log_run(
                    model_id="mace_mp0",
                    model_name="MACE-MP-0 (small)",
                    input_summary="Файл не загружен",
                    status="error",
                    inference_time_sec=0.0,
                    error_message=err_msg,
                )
                return
            filename = uploaded_file.name
            raw_bytes = uploaded_file.getvalue()
            source_label = f"Файл: {filename} ({len(raw_bytes)} байт)"

        is_valid, content_str, val_errors = validate_structure_bytes(filename, raw_bytes)
        if not is_valid:
            err_msg = " ".join(val_errors)
            for err in val_errors:
                st.error(err)
            log_run(
                model_id="mace_mp0",
                model_name="MACE-MP-0 (small)",
                input_summary=source_label,
                status="error",
                inference_time_sec=0.0,
                error_message=err_msg,
            )
            return

        if not ml_state["available"]:
            msg = (
                f"Файл «{filename}» прошел проверку формата, но расчет невозможен: "
                f"{ml_state['message']}"
            )
            st.warning(msg)
            log_run(
                model_id="mace_mp0",
                model_name="MACE-MP-0 (small)",
                input_summary=source_label,
                status="error",
                inference_time_sec=0.0,
                error_message=ml_state["message"],
            )
            return

        model = ml_state["model"]
        temp_path: Path | None = None
        try:
            if selected_preset is not None:
                structure_path = selected_preset
            else:
                suffix = Path(filename).suffix.lower()
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix, mode="wb"
                ) as tmp_file:
                    tmp_file.write(raw_bytes)
                    temp_path = Path(tmp_file.name)
                structure_path = temp_path

            input_data = {
                "structure_path": str(structure_path),
                "file_path": str(structure_path),
                "file_name": filename,
                "file_format": Path(filename).suffix.lower().lstrip("."),
                "file_content": content_str,
            }

            ok, model_errors = model.validate_input(input_data)
            if not ok:
                err_msg = "; ".join(model_errors) if model_errors else "Ошибка валидации структуры моделью."
                st.error(err_msg)
                log_run(
                    model_id="mace_mp0",
                    model_name="MACE-MP-0 (small)",
                    input_summary=source_label,
                    status="error",
                    inference_time_sec=0.0,
                    error_message=err_msg,
                )
                return

            with st.spinner("Выполняется single-point расчет MACE-MP-0 на CPU..."):
                result = model.predict(input_data)
        except Exception as exc:
            err_msg = f"Внутренняя ошибка при запуске MACE-MP-0: {exc}"
            st.error(err_msg)
            log_run(
                model_id="mace_mp0",
                model_name="MACE-MP-0 (small)",
                input_summary=source_label,
                status="error",
                inference_time_sec=0.0,
                error_message=err_msg,
            )
            return
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)

        if result.get("status") == "ok":
            _render_prediction_output(result)
            log_run(
                model_id="mace_mp0",
                model_name="MACE-MP-0 (small)",
                input_summary=source_label,
                status="ok",
                e_ads_ev=result.get("e_ads_ev"),
                interpretation=result.get("interpretation", ""),
                inference_time_sec=float(result.get("inference_time_sec") or 0.0),
                error_message=None,
            )
        else:
            err_msg = result.get("error_message") or "Модель вернула ошибку без описания."
            st.error(err_msg)
            log_run(
                model_id="mace_mp0",
                model_name="MACE-MP-0 (small)",
                input_summary=source_label,
                status="error",
                inference_time_sec=float(result.get("inference_time_sec") or 0.0),
                error_message=err_msg,
            )


def _render_local_calc_tab(ml_states: dict[str, dict[str, Any]]) -> None:
    """Вкладка 2: Локальный расчет на CPU."""
    st.subheader("Локальный расчет энергии адсорбции $E_{\\text{ads}}$ (CPU)")
    selected_model = st.radio(
        "Выберите локальную модель",
        options=["MXene Gas Sensor (CPU)", "MACE-MP-0 (small, CPU)"],
        horizontal=True,
    )
    st.divider()
    if selected_model == "MXene Gas Sensor (CPU)":
        _render_mxene_calculator(ml_states["mxene_ml"])
    else:
        _render_mace_calculator(ml_states["mace_mp0"])


def _render_datasets_tab() -> None:
    """Вкладка 3: Просмотр датасетов и эталонных бенчмарков."""
    st.subheader("Датасеты и эталонные бенчмарки")

    st.markdown("### 1. Подготовленный датасет MXene (`data/mxene_clean.csv`)")
    df_mxene, mxene_err = load_csv_safe(str(MXENE_DATA_PATH))
    if mxene_err:
        st.error(mxene_err)
    elif df_mxene is None:
        st.info(
            "Файл датасета пока отсутствует на диске. "
            "Ожидаемый путь: `data/mxene_clean.csv`."
        )
    else:
        s1, s2, s3 = st.columns(3)
        s1.metric("Строк в датасете", len(df_mxene))
        s2.metric("Признаков (колонок)", len(df_mxene.columns))
        s3.metric("Пропусков", int(df_mxene.isna().sum().sum()))

        search_query = st.text_input(
            "Текстовый поиск по таблице MXene",
            placeholder="Введите газ, материал или подстроку...",
            key="mxene_dataset_search",
        ).strip()

        cat_cols = [
            col
            for col in df_mxene.columns
            if not pd.api.types.is_numeric_dtype(df_mxene[col])
            and 1 < df_mxene[col].nunique(dropna=True) <= 50
        ][:3]

        filtered_df = df_mxene.copy()
        if cat_cols:
            filter_cols = st.columns(len(cat_cols))
            for idx, col_name in enumerate(cat_cols):
                unique_vals = sorted(filtered_df[col_name].dropna().astype(str).unique().tolist())
                with filter_cols[idx]:
                    chosen_vals = st.multiselect(
                        f"Фильтр: {col_name}",
                        options=unique_vals,
                        key=f"mxene_filter_{col_name}",
                    )
                if chosen_vals:
                    filtered_df = filtered_df[filtered_df[col_name].astype(str).isin(chosen_vals)]

        if search_query:
            mask = filtered_df.astype(str).apply(
                lambda col: col.str.contains(search_query, case=False, na=False)
            ).any(axis=1)
            filtered_df = filtered_df[mask]

        st.caption(f"Показано записей: {len(filtered_df)} из {len(df_mxene)}")
        st.dataframe(filtered_df, width="stretch")

    st.divider()
    st.markdown("### 2. Эталонные литературные бенчмарки (`data/test_cases/benchmarks.csv`)")
    df_bench, bench_err = load_csv_safe(str(BENCHMARKS_PATH))
    if bench_err:
        st.error(bench_err)
    elif df_bench is None:
        st.info(
            "Файл эталонных литературных бенчмарков пока отсутствует на диске. "
            "Ожидаемый путь: `data/test_cases/benchmarks.csv`."
        )
    else:
        b1, b2 = st.columns(2)
        b1.metric("Эталонных кейсов", len(df_bench))
        b2.metric("Колонок", len(df_bench.columns))

        bench_search = st.text_input(
            "Поиск по бенчмаркам",
            placeholder="Фильтр по системе, веществу или источнику...",
            key="bench_dataset_search",
        ).strip()
        filtered_bench = df_bench
        if bench_search:
            mask = filtered_bench.astype(str).apply(
                lambda col: col.str.contains(bench_search, case=False, na=False)
            ).any(axis=1)
            filtered_bench = filtered_bench[mask]
        st.dataframe(filtered_bench, width="stretch")


def _render_history_tab() -> None:
    """Вкладка 4: История расчетов и метрики из SQLite."""
    st.subheader("История запусков и логирование (SQLite)")
    stats = get_runs_stats()

    m1, m2, m3 = st.columns(3)
    m1.metric("Всего запусков", stats["total_runs"])
    m2.metric(
        "Успешных запусков",
        f"{stats['success_rate']:.1f}%",
        delta=f"{stats['successful_runs']} шт.",
        delta_color="off",
    )
    m3.metric("Среднее время расчета", f"{stats['avg_inference_time_sec']:.4f} сек")

    history = get_runs_history(limit=200)
    if not history:
        st.info("История запусков пока пуста. Выполните расчет во вкладке «Локальный расчет».")
        return

    df_history = pd.DataFrame(history)
    csv_bytes = df_history.to_csv(index=False).encode("utf-8-sig")

    btn_col1, btn_col2, _ = st.columns([1.5, 1.5, 3])
    with btn_col1:
        st.download_button(
            label="Скачать логи в CSV",
            data=csv_bytes,
            file_name="zoo_ai_runs_history.csv",
            mime="text/csv",
        )
    with btn_col2:
        if st.button("Очистить тестовую историю", type="secondary"):
            clear_runs_history()
            st.success("История запусков очищена.")
            st.rerun()

    st.dataframe(df_history, width="stretch")


def main() -> None:
    """Точка входа приложения Streamlit."""
    st.set_page_config(
        page_title="Зоопарк ИИ — Адсорбция",
        page_icon="🧪",
        layout="wide",
    )

    init_db()
    models = get_all_models()

    ml_states = {
        "mxene_ml": load_ml_model(
            "models.mxene_model",
            ("MXeneModel", "MxeneModel", "MXeneGasSensorModel", "MXeneAdsorptionModel"),
        ),
        "mace_mp0": load_ml_model(
            "models.mace_model",
            ("MACEModel", "MaceModel", "MACEMP0Model", "MaceMP0Model", "MACEAdsorptionModel"),
        ),
    }

    _render_sidebar(models, ml_states)

    st.title("🧪 Зоопарк ИИ — предсказание энергии адсорбции ($E_{\\text{ads}}$)")
    st.caption(
        "УрФУ (ИРИТ-РТФ) × ООО «ЦИФРОВЫЕ РЕШЕНИЯ» | Оффлайн-режим на Windows + CPU (Демо 2)"
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🧭 Подбор модели (Советчик)",
            "⚙️ Локальный расчет",
            "📊 Датасеты и бенчмарки",
            "🗂️ История и логи (SQLite)",
        ]
    )

    with tab1:
        _render_advisor_tab(models, ml_states)
    with tab2:
        _render_local_calc_tab(ml_states)
    with tab3:
        _render_datasets_tab()
    with tab4:
        _render_history_tab()


if __name__ == "__main__":
    main()
