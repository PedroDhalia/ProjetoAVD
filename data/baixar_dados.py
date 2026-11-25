"""Baixa os dois datasets do projeto (caso não estejam em data/).

Uso (na raiz do repositório):
    python data/baixar_dados.py

* Parte Comum — Obesity (UCI 544):
  https://archive.ics.uci.edu/dataset/544
* Parte Específica — Hotel Booking Demand (Kaggle: jessemostipak/hotel-booking-demand).
  O Kaggle exige login para baixar; usamos o espelho público idêntico do
  TidyTuesday (mesmo arquivo, 16.855.599 bytes, 119.390 linhas x 32 colunas).
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

PASTA = Path(__file__).resolve().parent

OBESITY_URL = (
    "https://archive.ics.uci.edu/static/public/544/"
    "estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition.zip"
)
OBESITY_CSV = PASTA / "parte_comum" / "ObesityDataSet_raw_and_data_sinthetic.csv"

HOTEL_URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2020/2020-02-11/hotels.csv"
HOTEL_CSV = PASTA / "parte_especifica" / "hotel_bookings.csv"


def _baixar(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "projeto-avd-download/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resposta:
        return resposta.read()


def main() -> None:
    if OBESITY_CSV.exists():
        print(f"[ok] {OBESITY_CSV.name} já existe")
    else:
        OBESITY_CSV.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(_baixar(OBESITY_URL))) as z:
            OBESITY_CSV.write_bytes(z.read(OBESITY_CSV.name))
        print(f"[baixado] {OBESITY_CSV}")

    if HOTEL_CSV.exists():
        print(f"[ok] {HOTEL_CSV.name} já existe")
    else:
        HOTEL_CSV.parent.mkdir(parents=True, exist_ok=True)
        HOTEL_CSV.write_bytes(_baixar(HOTEL_URL))
        print(f"[baixado] {HOTEL_CSV}")


if __name__ == "__main__":
    main()
