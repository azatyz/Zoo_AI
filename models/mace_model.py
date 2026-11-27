"""MACE-MP-0 (small): E_ads = E(поверхность + молекула) - E(поверхность) - E(молекула), эВ.

Только быстрый single-point на CPU, без релаксации геометрии. Веса лежат локально
в weights/mace_mp_small.model (скачать один раз: python scripts/download_mace_weights.py).

Вход, вариант 1 (готовый пресет из data/structures):
    {"preset": "Cu111_CO"}
Вход, вариант 2 (свои файлы .xyz/.cif):
    {"system_path": "...", "surface_path": "...", "molecule_path": "..."}
"""

from collections import Counter
from pathlib import Path

import numpy as np

from models.base import BaseAdsorptionModel

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = ROOT / "weights" / "mace_mp_small.model"
STRUCTURES_DIR = ROOT / "data" / "structures"

ALLOWED_SUFFIXES = (".xyz", ".cif")
MAX_ATOMS = 300           # больше на CPU считать слишком долго для демо
MIN_DISTANCE_A = 0.6      # ближе атомы значит сломанная геометрия
SUSPICIOUS_E_ADS_EV = 5.0
PATH_KEYS = ("system_path", "surface_path", "molecule_path")


class MACEModel(BaseAdsorptionModel):
    id = "mace_mp0_small"
    name = "MACE-MP-0 (small)"
    description = (
        "Универсальный межатомный потенциал MACE-MP-0: считает энергии системы, поверхности "
        "и молекулы и по ним E_ads = E(поверхность+молекула) - E(поверхность) - E(молекула). "
        "Быстрый single-point на CPU без релаксации, значения ориентировочные."
    )
    supported_inputs = [
        "preset: имя готового пресета из data/structures, например Cu111_CO",
        "или system_path, surface_path, molecule_path: пути к файлам .xyz или .cif",
    ]

    def __init__(self):
        super().__init__()
        self._calc = None

    # ---- пресеты -----------------------------------------------------------

    @staticmethod
    def available_presets() -> list:
        """Имена пресетов, для которых есть все три файла (для выпадающего списка в интерфейсе)."""
        if not STRUCTURES_DIR.exists():
            return []
        names = []
        for system in sorted(STRUCTURES_DIR.glob("*_system.xyz")):
            name = system.name[: -len("_system.xyz")]
            if (STRUCTURES_DIR / f"{name}_surface.xyz").exists() and (STRUCTURES_DIR / f"{name}_molecule.xyz").exists():
                names.append(name)
        return names

    # ---- чтение структур ---------------------------------------------------

    @staticmethod
    def _resolve(path_like) -> Path:
        path = Path(str(path_like).strip()).expanduser()
        if not path.is_absolute() and not path.exists():
            path = ROOT / path
        return path

    def _paths_from_input(self, input_data: dict):
        """Возвращает (метка, три пути, список ошибок)."""
        has_preset = input_data.get("preset") not in (None, "")
        has_paths = any(input_data.get(k) not in (None, "") for k in PATH_KEYS)
        if has_preset and has_paths:
            return None, None, ["укажите либо «preset», либо три пути к файлам, но не оба варианта сразу"]
        if not has_preset and not has_paths:
            return None, None, ["укажите «preset» (например Cu111_CO) или три пути: system_path, surface_path, molecule_path"]
        if has_preset:
            name = input_data["preset"]
            if not isinstance(name, str):
                return None, None, ["«preset» должен быть строкой"]
            name = name.strip()
            presets = self.available_presets()
            if name not in presets:
                shown = ", ".join(presets) if presets else "нет (запустите python scripts/generate_presets.py)"
                return None, None, [f"пресет «{name}» не найден. Доступные: {shown}"]
            paths = [STRUCTURES_DIR / f"{name}_{part}.xyz" for part in ("system", "surface", "molecule")]
            return name, paths, []
        missing = [k for k in PATH_KEYS if not isinstance(input_data.get(k), (str, Path)) or not str(input_data.get(k)).strip()]
        if missing:
            return None, None, [f"не указаны пути: {', '.join(missing)}"]
        return "свои файлы", [self._resolve(input_data[k]) for k in PATH_KEYS], []

    def _read_structures(self, input_data: dict):
        """Читает три структуры. Возвращает (метка, (system, surface, molecule), ошибки)."""
        label, paths, errors = self._paths_from_input(input_data)
        if errors:
            return label, None, errors
        from ase.io import read

        titles = ("система", "поверхность", "молекула")
        atoms_list = []
        for title, path in zip(titles, paths):
            if path.suffix.lower() not in ALLOWED_SUFFIXES:
                errors.append(f"{title}: допустимы только файлы .xyz и .cif, а не «{path.suffix}»")
                continue
            if not path.exists():
                errors.append(f"{title}: файл не найден ({path.name})")
                continue
            try:
                atoms = read(path, index=0)
            except Exception as exc:
                errors.append(f"{title}: не удалось прочитать {path.name} ({exc})")
                continue
            if len(atoms) == 0:
                errors.append(f"{title}: в файле {path.name} нет атомов")
            elif not np.isfinite(atoms.positions).all():
                errors.append(f"{title}: в {path.name} некорректные координаты")
            else:
                atoms_list.append(atoms)
        if errors:
            return label, None, errors

        system, surface, molecule = atoms_list
        if len(system) > MAX_ATOMS:
            errors.append(f"система слишком большая ({len(system)} атомов, максимум {MAX_ATOMS}): расчёт на CPU будет долгим")
        expected = Counter(surface.get_chemical_symbols()) + Counter(molecule.get_chemical_symbols())
        if Counter(system.get_chemical_symbols()) != expected:
            errors.append(
                f"состав не сходится: система {system.get_chemical_formula()}, "
                f"а поверхность + молекула дают {surface.get_chemical_formula()} + {molecule.get_chemical_formula()}"
            )
        if not errors and len(system) > 1:
            dist = system.get_all_distances(mic=bool(system.pbc.any()))
            np.fill_diagonal(dist, np.inf)
            if dist.min() < MIN_DISTANCE_A:
                errors.append(f"в системе атомы слишком близко ({dist.min():.2f} Å): проверьте геометрию")
        if errors:
            return label, None, errors
        return label, (system, surface, molecule), []

    # ---- контракт ----------------------------------------------------------

    def validate_input(self, input_data: dict) -> tuple:
        _, structures, errors = self._read_structures(input_data)
        return (not errors), errors

    def _create_calculator(self):
        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError(
                f"не найден файл весов {WEIGHTS_PATH.name} в папке weights/ "
                "(скачайте один раз: python scripts/download_mace_weights.py)"
            )
        try:
            from mace.calculators import MACECalculator, mace_mp
        except ImportError:
            raise RuntimeError("не установлен пакет mace-torch (pip install mace-torch)")
        try:
            return mace_mp(model=str(WEIGHTS_PATH), device="cpu", default_dtype="float64", dispersion=False)
        except Exception as first_error:
            try:
                return MACECalculator(model_paths=str(WEIGHTS_PATH), device="cpu", default_dtype="float64")
            except Exception as second_error:
                raise RuntimeError(f"{first_error}; запасной вариант: {second_error}")

    def load(self) -> None:
        self._calc = self._create_calculator()

    def _predict_impl(self, input_data: dict) -> dict:
        label, structures, errors = self._read_structures(input_data)
        if errors:
            raise ValueError("; ".join(errors))
        energies = {}
        for key, atoms in zip(("system", "surface", "molecule"), structures):
            atoms = atoms.copy()
            atoms.calc = self._calc
            energies[key] = float(atoms.get_potential_energy())
        e_ads = energies["system"] - energies["surface"] - energies["molecule"]

        warnings = ["Геометрия не релаксирована (single-point): значение ориентировочное."]
        if abs(e_ads) > SUSPICIOUS_E_ADS_EV:
            warnings.append(f"Значение {e_ads:.2f} эВ выглядит нереалистичным: проверьте геометрию структур.")
        return {
            "e_ads_ev": e_ads,
            "details": {
                "source": label,
                "formula": "E_ads = E(поверхность + молекула) - E(поверхность) - E(молекула)",
                "energies_ev": {k: round(v, 4) for k, v in energies.items()},
                "n_atoms": {k: len(a) for k, a in zip(("system", "surface", "molecule"), structures)},
                "model": "MACE-MP-0 small, CPU, float64",
                "relaxed": False,
            },
            "warnings": warnings,
        }
