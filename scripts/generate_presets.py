"""Генерация тестовых структур для MACE-модели (через ase.build).

Для каждого пресета создаются три файла в data/structures/:
    <name>_system.xyz   поверхность + молекула
    <name>_surface.xyz  чистая поверхность
    <name>_molecule.xyz молекула в вакууме
Формат extxyz (расширение .xyz): хранит ячейку и периодичность, обычный xyz их теряет.
Геометрия НЕ релаксирована: расчёт будет single-point, значения E_ads ориентировочные.

Запуск из корня проекта:  python scripts/generate_presets.py
"""

from pathlib import Path

from ase.build import add_adsorbate, fcc111, molecule
from ase.io import write

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "structures"

# металл -> высота атома C над верхним слоем, Å (позиция "on top")
PRESETS = {
    "Cu111_CO": ("Cu", 1.85),
    "Pt111_CO": ("Pt", 1.85),
    "Al111_CO": ("Al", 2.00),
    "Au111_CO": ("Au", 2.00),
}


def build_co():
    co = molecule("CO")
    # атом, который ставим на поверхность, должен быть углеродом (C вниз, O вверх)
    c, o = (a for a in co if a.symbol == "C"), (a for a in co if a.symbol == "O")
    if next(c).z > next(o).z:
        co.rotate(180, "x")
    co.translate(-co[[a.symbol for a in co].index("C")].position)
    co.set_pbc(False)
    return co


def build_preset(metal: str, height: float):
    surface = fcc111(metal, size=(3, 3, 4), vacuum=8.0, orthogonal=False)
    surface.set_pbc((True, True, False))
    co = build_co()
    # перестановка: C первым, чтобы add_adsorbate (mol_index=0) ставил именно его
    order = sorted(range(len(co)), key=lambda i: co[i].symbol != "C")
    co_for_ads = co[order]
    system = surface.copy()
    add_adsorbate(system, co_for_ads, height=height, position="ontop")
    # служебные данные ase (adsorbate_info) extxyz не умеет сохранять
    system.info.clear()
    surface.info.clear()
    gas = co.copy()
    gas.center(vacuum=7.0)
    gas.set_pbc(False)
    return system, surface, gas


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, (metal, height) in PRESETS.items():
        system, surface, gas = build_preset(metal, height)
        write(OUT_DIR / f"{name}_system.xyz", system, format="extxyz")
        write(OUT_DIR / f"{name}_surface.xyz", surface, format="extxyz")
        write(OUT_DIR / f"{name}_molecule.xyz", gas, format="extxyz")
        print(f"{name}: система {len(system)} атомов, поверхность {len(surface)}, молекула {len(gas)}")
    print(f"Готово, файлы в {OUT_DIR}")


if __name__ == "__main__":
    main()
