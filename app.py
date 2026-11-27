"""Веб-приложение «Зоопарк ИИ — Адсорбция» (Streamlit + SQLite) для Демо 2."""

from __future__ import annotations

import html
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
MXENE_NON_FEATURES = {"mxene_formula", "adsorbate", "adsorption_site", "eads", "split"}
MXENE_MODEL_NAME = "MXene Gas Sensor ML (XGBoost)"
MACE_MODEL_NAME = "MACE-MP-0 (small)"


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
    """Безопасно загружает, прогревает (warmup) и кэширует локальную ML-модель."""
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        missing_name = getattr(exc, "name", "") or ""
        if missing_name in {module_path, "models", "models.base"} or not missing_name:
            return {
                "available": False,
                "status": "pending_merge",
                "model": None,
                "model_class": None,
                "message": "Модуль ML не найден или ожидает слияния ветки моделей.",
            }
        return {
            "available": False,
            "status": "dep_error",
            "model": None,
            "model_class": None,
            "message": f"Не удалось загрузить зависимость ML-модуля ({missing_name}): {exc}",
        }
    except Exception as exc:
        return {
            "available": False,
            "status": "load_error",
            "model": None,
            "model_class": None,
            "message": f"Ошибка при импорте модуля `{module_path}`: {exc}",
        }

    model_cls = _resolve_model_class(module, candidate_names)
    if model_cls is None:
        return {
            "available": False,
            "status": "load_error",
            "model": None,
            "model_class": None,
            "message": f"В модуле `{module_path}` не найден класс модели с методами validate_input/predict.",
        }

    try:
        instance = model_cls()
    except Exception as exc:
        return {
            "available": False,
            "status": "load_error",
            "model": None,
            "model_class": model_cls,
            "message": f"Ошибка инициализации модели `{model_cls.__name__}`: {exc}",
        }

    warmup_fn = getattr(instance, "warmup", None)
    if callable(warmup_fn):
        try:
            loaded, warmup_error = warmup_fn()
        except Exception as exc:
            return {
                "available": False,
                "status": "warmup_error",
                "model": None,
                "model_class": model_cls,
                "message": f"Ошибка при загрузке весов модели `{model_cls.__name__}`: {exc}",
            }
        if not loaded:
            return {
                "available": False,
                "status": "warmup_error",
                "model": None,
                "model_class": model_cls,
                "message": warmup_error or f"Не удалось загрузить веса модели `{model_cls.__name__}`.",
            }

    return {
        "available": True,
        "status": "connected",
        "model": instance,
        "model_class": model_cls,
        "message": "Модуль ML подключен, веса загружены и готовы к локальному расчету на CPU.",
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


def list_structure_presets(structures_dir: Path) -> list[str]:
    """Возвращает имена пресетов, для которых есть все три файла (_system, _surface, _molecule)."""
    if not structures_dir.exists() or not structures_dir.is_dir():
        return []
    names: list[str] = []
    for system_file in sorted(structures_dir.glob("*_system.xyz"), key=lambda p: p.name.lower()):
        preset_name = system_file.name[: -len("_system.xyz")]
        if (
            (structures_dir / f"{preset_name}_surface.xyz").exists()
            and (structures_dir / f"{preset_name}_molecule.xyz").exists()
        ):
            names.append(preset_name)
    return names


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
        if ml_state and ml_state.get("status") in {"dep_error", "warmup_error", "load_error"}:
            return "🟠 Локально CPU — не готова"
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
            title_suffix = " — ⭐ Рекомендовано под задачу" if is_rec else ""
            escaped_name = html.escape(str(model["name"]))
            escaped_suffix = html.escape(title_suffix)
            escaped_id = html.escape(str(model["id"]))
            escaped_arch = html.escape(str(model["architecture"]))
            escaped_badge = html.escape(badge)
            escaped_hw = html.escape(str(model["hardware_req"]))

            header_html = (
                '<div style="display: flex; justify-content: space-between; align-items: flex-start; '
                'gap: 20px; padding-bottom: 0.75rem; margin-bottom: 0.85rem; '
                'border-bottom: 1px solid rgba(128, 128, 128, 0.22);">'
                '<div style="flex: 1 1 auto; min-width: 0;">'
                f'<div style="font-size: 1.3rem; font-weight: 600; line-height: 1.3; margin: 0 0 0.25rem 0;">'
                f'{escaped_name}{escaped_suffix}</div>'
                f'<div style="font-size: 0.85rem; opacity: 0.75; line-height: 1.4; margin: 0;">'
                f'ID: <code>{escaped_id}</code> | Архитектура: {escaped_arch}</div>'
                '</div>'
                '<div style="flex: 0 0 auto; text-align: right; max-width: 44%;">'
                f'<div style="font-size: 0.95rem; line-height: 1.3; margin: 0 0 0.25rem 0; white-space: nowrap;">'
                f'<strong>Статус:</strong> {escaped_badge}</div>'
                f'<div style="font-size: 0.85rem; opacity: 0.75; line-height: 1.4; margin: 0;">'
                f'Железо: {escaped_hw}</div>'
                '</div>'
                '</div>'
            )
            st.markdown(header_html, unsafe_allow_html=True)

            info_col1, info_col2 = st.columns(2, gap="large")
            with info_col1:
                st.markdown(f"**Формат входа:** {model['input_format']}")
                st.markdown(f"**Формат выхода:** {model['output_format']}")
                st.markdown(f"**Метрики / ориентиры:** {model['metrics_summary']}")
            with info_col2:
                st.markdown(f"**Для чего подходит:** {model['suitable_for']}")
                st.markdown(f"**Ограничения:** {model['limitations']}")

            if model["status"] == "catalog_gpu_only":
                st.markdown('<div style="height: 0.5rem;"></div>', unsafe_allow_html=True)
                st.info(
                    "Режим каталога-советчика: запуск требует NVIDIA GPU и недоступен локально на CPU. "
                    "По согласованию с куратором запрос GPU в техподдержке ИРИТ запланирован на следующий спринт."
                )
            elif ml_state and not ml_state.get("available"):
                st.markdown('<div style="height: 0.5rem;"></div>', unsafe_allow_html=True)
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


def _render_mxene_calculator(ml_state: dict[str, Any]) -> None:
    """Отрисовывает форму расчета для модели MXene Gas Sensor ML (XGBoost, CPU)."""
    st.markdown(f"#### {MXENE_MODEL_NAME} (локальный CPU)")
    if ml_state["available"]:
        st.success(ml_state["message"])
    else:
        st.warning(ml_state["message"])

    df_mxene, csv_err = load_csv_safe(str(MXENE_DATA_PATH))
    if csv_err:
        st.error(csv_err)

    has_dataset = (
        df_mxene is not None
        and not df_mxene.empty
        and {"mxene_formula", "adsorbate"}.issubset(df_mxene.columns)
    )

    if has_dataset:
        input_mode = st.radio(
            "Способ задания образца",
            options=[
                "Выбрать пару из data/mxene_clean.csv",
                "Ввести формулу и газ вручную",
            ],
            horizontal=True,
            key="mxene_input_mode",
        )
    else:
        st.info(
            "Файл `data/mxene_clean.csv` отсутствует или не содержит колонок "
            "`mxene_formula` / `adsorbate` — используется ручной ввод."
        )
        input_mode = "Ввести формулу и газ вручную"

    if input_mode == "Выбрать пару из data/mxene_clean.csv" and has_dataset:
        formulas = sorted(df_mxene["mxene_formula"].dropna().astype(str).unique().tolist())
        col_formula, col_gas = st.columns(2)
        with col_formula:
            mxene_formula = st.selectbox(
                "Материал MXene (`mxene_formula`)",
                options=formulas,
                key="mxene_select_formula",
                help=f"Доступно {len(formulas)} уникальных MXene из всего датасета.",
            )
        available_gases = sorted(
            df_mxene.loc[
                df_mxene["mxene_formula"].astype(str) == mxene_formula,
                "adsorbate",
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        with col_gas:
            adsorbate = st.selectbox(
                "Газ-адсорбат (`adsorbate`)",
                options=available_gases,
                key="mxene_select_gas",
                help="Показаны газы, доступные для выбранного MXene в датасете.",
            )

        matched = df_mxene[
            (df_mxene["mxene_formula"].astype(str) == mxene_formula)
            & (df_mxene["adsorbate"].astype(str) == adsorbate)
        ]
        if not matched.empty:
            sample_row = matched.iloc[0]
            info_parts: list[str] = []
            if "adsorption_site" in df_mxene.columns and pd.notna(sample_row["adsorption_site"]):
                info_parts.append(f"Сайт: **{sample_row['adsorption_site']}**")
            if "split" in df_mxene.columns and pd.notna(sample_row["split"]):
                split_val = str(sample_row["split"])
                split_note = " (на train ошибка занижена)" if split_val == "train" else ""
                info_parts.append(f"Выборка: **{split_val}**{split_note}")
            if "Eads" in df_mxene.columns and pd.notna(sample_row["Eads"]):
                info_parts.append(f"Эталон `Eads`: **{float(sample_row['Eads']):.4f} эВ**")
            if info_parts:
                st.caption("Справка по образцу: " + " | ".join(info_parts))

            descriptor_cols = [
                col for col in df_mxene.columns if col.strip().lower() not in MXENE_NON_FEATURES
            ]
            if descriptor_cols:
                with st.expander(
                    f"Дескрипторы образца из data/mxene_clean.csv ({len(descriptor_cols)} признаков — подставляются моделью автоматически)"
                ):
                    desc_df = pd.DataFrame(
                        {
                            "Признак": descriptor_cols,
                            "Значение": [sample_row[c] for c in descriptor_cols],
                        }
                    )
                    st.dataframe(desc_df, width="stretch", hide_index=True)
    else:
        col_left, col_right = st.columns(2)
        with col_left:
            mxene_formula = st.text_input(
                "Материал MXene (`mxene_formula`)",
                value="Ti2CO2",
                help="Формула MXene из data/mxene_clean.csv, например: Ti2CO2, Ti3C2O2, V2CO2, Nb2CO2.",
                key="mxene_manual_formula",
            )
        with col_right:
            adsorbate = st.text_input(
                "Газ-адсорбат (`adsorbate`)",
                value="NO2",
                help="Поддерживаемые газы: CH4, CO, H2S, HCHO, NH3, NO, NO2, SO2.",
                key="mxene_manual_gas",
            )

    if st.button("Рассчитать E_ads (MXene)", type="primary"):
        clean_formula = str(mxene_formula or "").strip()
        clean_gas = str(adsorbate or "").strip()
        input_data = {
            "mxene_formula": clean_formula,
            "adsorbate": clean_gas,
        }
        input_summary = f"mxene_formula={clean_formula or '—'}; adsorbate={clean_gas or '—'}"

        val_errors: list[str] = []
        if not clean_formula:
            val_errors.append("Поле «mxene_formula» не может быть пустым.")
        if not clean_gas:
            val_errors.append("Поле «adsorbate» не может быть пустым.")

        if val_errors:
            err_text = " ".join(val_errors)
            for err in val_errors:
                st.error(err)
            log_run(
                model_id="mxene_ml",
                model_name=MXENE_MODEL_NAME,
                input_summary=input_summary,
                status="error",
                inference_time_sec=0.0,
                error_message=err_text,
            )
            return

        if not ml_state["available"]:
            msg = (
                "Входные параметры указаны, но расчёт невозможен: "
                f"{ml_state['message']}"
            )
            st.warning(msg)
            log_run(
                model_id="mxene_ml",
                model_name=MXENE_MODEL_NAME,
                input_summary=input_summary,
                status="error",
                inference_time_sec=0.0,
                error_message=ml_state["message"],
            )
            return

        model = ml_state["model"]
        try:
            ok, model_errors = model.validate_input(input_data)
            if not ok:
                err_text = "; ".join(model_errors) if model_errors else "Ошибка валидации входных данных моделью."
                st.error(err_text)
                log_run(
                    model_id="mxene_ml",
                    model_name=MXENE_MODEL_NAME,
                    input_summary=input_summary,
                    status="error",
                    inference_time_sec=0.0,
                    error_message=err_text,
                )
                return

            with st.spinner("Выполняется локальный расчет MXene на CPU..."):
                result = model.predict(input_data)
        except Exception as exc:
            err_text = f"Внутренняя ошибка при запуске MXene-модели: {exc}"
            st.error(err_text)
            log_run(
                model_id="mxene_ml",
                model_name=MXENE_MODEL_NAME,
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
                model_name=MXENE_MODEL_NAME,
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
                model_name=MXENE_MODEL_NAME,
                input_summary=input_summary,
                status="error",
                inference_time_sec=float(result.get("inference_time_sec") or 0.0),
                error_message=err_text,
            )


def _get_mace_presets(ml_state: dict[str, Any]) -> list[str]:
    """Получает список доступных пресетов через MACEModel.available_presets() или папку структур."""
    model_obj = ml_state.get("model") or ml_state.get("model_class")
    avail_fn = getattr(model_obj, "available_presets", None)
    if callable(avail_fn):
        try:
            return list(avail_fn())
        except Exception:
            pass
    return list_structure_presets(STRUCTURES_DIR)


def _render_mace_calculator(ml_state: dict[str, Any]) -> None:
    """Отрисовывает форму расчета для модели MACE-MP-0 (small, CPU)."""
    st.markdown(f"#### {MACE_MODEL_NAME} (локальный CPU)")
    if ml_state["available"]:
        st.success(ml_state["message"])
    else:
        st.warning(ml_state["message"])

    presets = _get_mace_presets(ml_state)
    source_mode = st.radio(
        "Источник атомной структуры",
        options=[
            "Готовый пресет из data/structures/",
            "Загрузить свои 3 файла (.xyz / .cif)",
        ],
        horizontal=True,
        key="mace_source_mode",
    )

    selected_preset: str | None = None
    uploaded_files: dict[str, Any] = {}

    if source_mode == "Готовый пресет из data/structures/":
        if not presets:
            st.info(
                "В папке `data/structures/` пока нет полных пресетов "
                "(`*_system.xyz`, `*_surface.xyz`, `*_molecule.xyz`). "
                "Переключитесь на загрузку своих 3 файлов."
            )
        else:
            selected_preset = st.selectbox(
                "Выберите готовый пресет",
                options=presets,
                key="mace_preset_select",
                help="Каждый пресет включает три файла: систему (поверхность+молекула), поверхность и молекулу.",
            )
            system_preview_path = STRUCTURES_DIR / f"{selected_preset}_system.xyz"
            st.caption(
                f"Файлы пресета: `{selected_preset}_system.xyz`, "
                f"`{selected_preset}_surface.xyz`, `{selected_preset}_molecule.xyz`"
            )
            if system_preview_path.exists():
                try:
                    preview_lines = system_preview_path.read_text(encoding="utf-8").splitlines()[:15]
                    with st.expander(f"Предпросмотр `{system_preview_path.name}` (первые 15 строк)"):
                        st.code("\n".join(preview_lines), language="text")
                except Exception as exc:
                    st.warning(f"Не удалось прочитать превью пресета: {exc}")
    else:
        st.caption(
            "Для расчёта $E_{\\text{ads}} = E(\\text{система}) - E(\\text{поверхность}) - E(\\text{молекула})$ "
            "загрузите все три структуры в формате `.xyz` или `.cif`."
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            uploaded_files["system_path"] = st.file_uploader(
                "1. Система (поверхность + молекула)",
                key="mace_upload_system",
            )
        with c2:
            uploaded_files["surface_path"] = st.file_uploader(
                "2. Чистая поверхность",
                key="mace_upload_surface",
            )
        with c3:
            uploaded_files["molecule_path"] = st.file_uploader(
                "3. Молекула отдельно",
                key="mace_upload_molecule",
            )

    if st.button("Рассчитать E_ads (MACE-MP-0)", type="primary"):
        file_titles = {
            "system_path": "система",
            "surface_path": "поверхность",
            "molecule_path": "молекула",
        }
        raw_uploads: dict[str, tuple[str, bytes]] = {}

        if source_mode == "Готовый пресет из data/structures/":
            if not selected_preset:
                err_msg = "Не выбран готовый пресет из `data/structures/`."
                st.error(err_msg)
                log_run(
                    model_id="mace_mp0",
                    model_name=MACE_MODEL_NAME,
                    input_summary="Пресет не выбран",
                    status="error",
                    inference_time_sec=0.0,
                    error_message=err_msg,
                )
                return
            source_label = f"Пресет: {selected_preset}"
        else:
            missing_labels = [
                file_titles[k] for k in ("system_path", "surface_path", "molecule_path")
                if uploaded_files.get(k) is None
            ]
            if missing_labels:
                err_msg = (
                    "Загрузите все три файла (.xyz или .cif). "
                    f"Не указаны: {', '.join(missing_labels)}."
                )
                st.error(err_msg)
                log_run(
                    model_id="mace_mp0",
                    model_name=MACE_MODEL_NAME,
                    input_summary="Не загружены все 3 файла",
                    status="error",
                    inference_time_sec=0.0,
                    error_message=err_msg,
                )
                return

            all_errors: list[str] = []
            summary_parts: list[str] = []
            for key in ("system_path", "surface_path", "molecule_path"):
                up_file = uploaded_files[key]
                fname = up_file.name
                fbytes = up_file.getvalue()
                raw_uploads[key] = (fname, fbytes)
                summary_parts.append(f"{file_titles[key]}={fname}")
                is_valid, _, val_errors = validate_structure_bytes(fname, fbytes)
                if not is_valid:
                    for err in val_errors:
                        all_errors.append(f"{file_titles[key].capitalize()} ({fname}): {err}")

            source_label = "Файлы: " + ", ".join(summary_parts)
            if all_errors:
                err_msg = " ".join(all_errors)
                for err in all_errors:
                    st.error(err)
                log_run(
                    model_id="mace_mp0",
                    model_name=MACE_MODEL_NAME,
                    input_summary=source_label,
                    status="error",
                    inference_time_sec=0.0,
                    error_message=err_msg,
                )
                return

        if not ml_state["available"]:
            msg = (
                "Входные данные проверены, но расчет невозможен: "
                f"{ml_state['message']}"
            )
            st.warning(msg)
            log_run(
                model_id="mace_mp0",
                model_name=MACE_MODEL_NAME,
                input_summary=source_label,
                status="error",
                inference_time_sec=0.0,
                error_message=ml_state["message"],
            )
            return

        model = ml_state["model"]
        temp_paths: list[Path] = []
        try:
            if source_mode == "Готовый пресет из data/structures/":
                input_data: dict[str, str] = {"preset": str(selected_preset)}
            else:
                input_data = {}
                for key, (fname, fbytes) in raw_uploads.items():
                    suffix = Path(fname).suffix.lower()
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix, mode="wb"
                    ) as tmp_file:
                        tmp_file.write(fbytes)
                        tmp_path = Path(tmp_file.name)
                    temp_paths.append(tmp_path)
                    input_data[key] = str(tmp_path)

            ok, model_errors = model.validate_input(input_data)
            if not ok:
                err_msg = "; ".join(model_errors) if model_errors else "Ошибка валидации структуры моделью."
                st.error(err_msg)
                log_run(
                    model_id="mace_mp0",
                    model_name=MACE_MODEL_NAME,
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
                model_name=MACE_MODEL_NAME,
                input_summary=source_label,
                status="error",
                inference_time_sec=0.0,
                error_message=err_msg,
            )
            return
        finally:
            for tmp_path in temp_paths:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)

        if result.get("status") == "ok":
            _render_prediction_output(result)
            log_run(
                model_id="mace_mp0",
                model_name=MACE_MODEL_NAME,
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
                model_name=MACE_MODEL_NAME,
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
