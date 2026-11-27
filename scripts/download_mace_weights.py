"""Одноразовая загрузка весов MACE-MP-0 (small) в weights/mace_mp_small.model.

После этого приложение работает полностью офлайн.
Запуск из корня проекта (нужен интернет):  python scripts/download_mace_weights.py
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "weights" / "mace_mp_small.model"
URLS = [
    "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model",
    "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model",
]
MIN_SIZE = 20 * 1024 * 1024  # файл около 31 МБ; меньше значит, что скачалось не то


def download(url: str, target: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as resp, open(target, "wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def main():
    if DST.exists() and DST.stat().st_size >= MIN_SIZE:
        print(f"Файл уже есть: {DST} ({DST.stat().st_size / 1e6:.1f} МБ)")
        return
    DST.parent.mkdir(parents=True, exist_ok=True)
    tmp = DST.with_suffix(".part")
    last_error = None
    for url in URLS:
        try:
            print("Скачиваю:", url)
            download(url, tmp)
            if tmp.stat().st_size < MIN_SIZE:
                raise RuntimeError(f"файл слишком маленький ({tmp.stat().st_size} байт)")
            tmp.replace(DST)
            digest = hashlib.sha256(DST.read_bytes()).hexdigest()
            print(f"Готово: {DST} ({DST.stat().st_size / 1e6:.1f} МБ)\nsha256: {digest}")
            return
        except Exception as exc:  # пробуем следующий адрес
            last_error = exc
            print("Не получилось:", exc)
    if tmp.exists():
        tmp.unlink()
    sys.exit(f"Не удалось скачать веса: {last_error}. Проверьте интернет и повторите.")


if __name__ == "__main__":
    main()
