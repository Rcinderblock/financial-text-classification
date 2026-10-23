"""Загрузка готовой модели и классификация строки, списка строк или CSV."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from load_data import ROOT
from preprocess import CLASS_NAMES, clean_text

DEFAULT_MODEL = ROOT / "models/best_model.joblib"


def predict_sentiment(texts: str | list[str], model=None, model_path=DEFAULT_MODEL):
    if isinstance(texts, str):
        texts = [texts]
    if not isinstance(texts, list) or not texts:
        raise ValueError("Нужна строка или непустой список строк")
    if any(
        not isinstance(text, str) or len(text) > 20000 or not clean_text(text)
        for text in texts
    ):
        raise ValueError(
            "Тексты должны быть непустыми строками длиной до 20000 символов"
        )
    model = joblib.load(model_path) if model is None else model
    predictions = model.predict(texts)
    probabilities = (
        model.predict_proba(texts) if hasattr(model, "predict_proba") else None
    )
    rows = []
    for index, (text, label) in enumerate(zip(texts, predictions, strict=True)):
        row = {"text": text, "label": int(label), "sentiment": CLASS_NAMES[int(label)]}
        if probabilities is not None:
            row["probabilities"] = {
                CLASS_NAMES[int(code)]: float(probability)
                for code, probability in zip(
                    model.classes_, probabilities[index], strict=True
                )
            }
        rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Тональность финансовых новостей")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", action="append")
    source.add_argument("--input", type=Path, help="CSV с колонкой text")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        frame = pd.read_csv(args.input) if args.input else None
        if frame is not None and "text" not in frame:
            raise ValueError("В CSV нужна колонка text")
        texts = frame.text.tolist() if frame is not None else args.text
        rows = predict_sentiment(texts, model_path=args.model)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(args.output, index=False)
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError) as error:
        parser.exit(2, f"Ошибка: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
