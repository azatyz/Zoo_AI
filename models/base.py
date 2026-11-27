"""Базовый контракт для моделей предсказания энергии адсорбции (E_ads, эВ).

Тимлид вызывает только model.predict(input_data) и всегда получает словарь.
Исключения наружу не выходят: любая ошибка превращается в status="error"
с понятным сообщением на русском.

Наследнику нужно реализовать три метода: validate_input, load, _predict_impl.
"""

import math
import time
from abc import ABC, abstractmethod

# Границы для интерпретации по |E_ads|, эВ. Рабочие значения: согласовать с куратором.
PHYSISORPTION_MAX_EV = 0.3   # слабее: слабая физическая адсорбция
REVERSIBLE_MAX_EV = 0.8      # между границами: оптимально для обратимого сенсора


def interpret_e_ads(e_ads_ev: float) -> str:
    """Короткая трактовка E_ads на русском. Отрицательное значение = выгодная адсорбция."""
    if e_ads_ev > 0:
        return "Адсорбция энергетически невыгодна (E_ads > 0): молекула не удерживается на поверхности."
    strength = abs(e_ads_ev)
    if strength < PHYSISORPTION_MAX_EV:
        return "Слабая физическая адсорбция: связь слабая, отклик сенсора может быть малым."
    if strength <= REVERSIBLE_MAX_EV:
        return "Оптимально для обратимого сенсора: адсорбция достаточно сильная и при этом обратимая."
    return "Сильная хемосорбция: молекула удерживается прочно, восстановление сенсора будет медленным."


class BaseAdsorptionModel(ABC):
    # Метаданные: наследник переопределяет их в теле класса.
    id: str = ""
    name: str = ""
    description: str = ""
    supported_inputs: list = []   # человекочитаемое описание допустимых входов
    metrics: dict = {}            # например {"MAE": 0.1, "RMSE": 0.2, "R2": 0.9}

    def __init__(self):
        self._loaded = False

    # ---- что реализует наследник ------------------------------------------

    @abstractmethod
    def validate_input(self, input_data: dict) -> tuple:
        """Вернуть (True, []) или (False, [список понятных ошибок на русском])."""

    @abstractmethod
    def load(self) -> None:
        """Загрузить веса/данные из локальных файлов. При проблеме кидать исключение с понятным текстом."""

    @abstractmethod
    def _predict_impl(self, input_data: dict) -> dict:
        """Сам расчёт. Вернуть {"e_ads_ev": float, "details": dict, "warnings": list[str]}.
        details и warnings необязательны. Исключения можно не ловить: их поймает predict()."""

    # ---- единая обёртка, её наследники не переопределяют ------------------

    def predict(self, input_data: dict) -> dict:
        start = time.perf_counter()

        def finish(status, e_ads=None, interp="", details=None, warnings=None, error=None):
            return {
                "status": status,
                "e_ads_ev": e_ads,
                "interpretation": interp,
                "inference_time_sec": round(time.perf_counter() - start, 4),
                "details": details or {},
                "warnings": list(warnings or []),
                "error_message": error,
            }

        try:
            if not isinstance(input_data, dict):
                return finish("error", error="Входные данные должны быть словарём (dict).")

            ok, errors = self.validate_input(input_data)
            if not ok:
                return finish("error", error="Некорректные входные данные: " + "; ".join(errors))

            if not self._loaded:
                try:
                    self.load()
                except Exception as exc:
                    return finish("error", error=f"Не удалось загрузить модель «{self.name}»: {exc}")
                self._loaded = True

            result = self._predict_impl(input_data)
            e_ads = result.get("e_ads_ev")
            if isinstance(e_ads, bool) or not isinstance(e_ads, (int, float)) or not math.isfinite(e_ads):
                return finish("error", error="Модель вернула некорректное значение энергии адсорбции.")
            e_ads = float(e_ads)

            return finish(
                "ok", e_ads, interpret_e_ads(e_ads),
                details=result.get("details"), warnings=result.get("warnings"),
            )
        except Exception as exc:  # последняя линия обороны: демо не должно падать
            return finish("error", error=f"Внутренняя ошибка модели «{self.name}»: {exc}")

    def describe(self) -> dict:
        """Метаданные для интерфейса (выбор модели, подсказки)."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "supported_inputs": self.supported_inputs,
            "metrics": self.metrics,
        }
