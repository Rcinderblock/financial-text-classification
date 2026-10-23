"""Загрузка фиксированной версии Financial PhraseBank и проверка CSV."""

import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data/raw/financial_news.csv"
REVISION = "8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964"
SOURCE_URL = (
    "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/"
    + REVISION
    + "/data/FinancialPhraseBank-v1.0.zip"
)
ARCHIVE_SHA256 = "0e1a06c4900fdae46091d031068601e3773ba067c7cecb5b0da1dcba5ce989a6"


def load_data(path: Path = DEFAULT_DATA) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8")


def download_data(destination: Path = DEFAULT_DATA):
    with urlopen(SOURCE_URL, timeout=60) as response:
        content = response.read(2_000_001)
    digest = hashlib.sha256(content).hexdigest()
    if len(content) > 2_000_000 or digest != ARCHIVE_SHA256:
        raise ValueError("Контрольная сумма исходного архива не совпадает")
    with ZipFile(io.BytesIO(content)) as archive:
        name = next(
            name
            for name in archive.namelist()
            if name.endswith("Sentences_AllAgree.txt")
        )
        lines = archive.read(name).decode("latin-1").splitlines()
    rows = []
    for line in lines:
        if line.strip():
            text, sentiment = line.rsplit("@", 1)
            rows.append({"text": text.strip(), "sentiment": sentiment.strip()})
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)
    metadata = {
        "dataset": "Financial PhraseBank v1.0",
        "subset": "Sentences_AllAgree",
        "source_url": SOURCE_URL,
        "revision": REVISION,
        "archive_sha256": digest,
        "csv_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "rows": len(rows),
        "license": "CC BY-NC-SA 3.0",
    }
    destination.with_suffix(".source.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Загрузить Financial PhraseBank")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    print(json.dumps(download_data(args.output), indent=2))
