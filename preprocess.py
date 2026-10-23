"""Общая очистка для обучения и предсказаний, проверка размеченного CSV."""

import html
import re
import unicodedata

import pandas as pd

LABELS = {"negative": 0, "neutral": 1, "positive": 2}
CLASS_NAMES = list(LABELS)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("Текст должен быть строкой")
    text = unicodedata.normalize("NFKC", html.unescape(text))
    text = re.sub(r"</?[a-zA-Z][^>]*>", " ", text)
    # Сохраняем отрицания, числа, проценты и знаки денежных сумм.
    return " ".join(text.lower().split())


def preprocess_texts(texts):
    return [clean_text(text) for text in texts]


def label_code(value) -> int:
    if isinstance(value, str):
        value = value.strip().lower()
        if value in LABELS:
            return LABELS[value]
        if value in {"0", "1", "2"}:
            return int(value)
    elif not isinstance(value, bool) and value in {0, 1, 2}:
        return int(value)
    raise ValueError("Метка должна быть negative/neutral/positive или 0/1/2")


def prepare_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frame = raw.copy()
    frame.columns = [str(name).strip().lower() for name in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError("Названия колонок не должны повторяться")
    for alias, target in [("headline", "text"), ("label", "sentiment")]:
        if target not in frame and alias in frame:
            frame = frame.rename(columns={alias: target})
    if not {"text", "sentiment"} <= set(frame.columns):
        raise ValueError("В CSV нужны колонки text и sentiment")
    frame = frame[["text", "sentiment"]].copy()
    frame.insert(0, "row_id", range(len(frame)))
    report = {"input_rows": len(frame)}
    valid = frame["text"].map(lambda value: isinstance(value, str))
    valid &= frame["sentiment"].notna()
    report["missing_or_nontext_rows"] = int((~valid).sum())
    frame = frame[valid].copy()
    frame["label"] = frame["sentiment"].map(label_code)
    frame["text_clean"] = frame["text"].map(clean_text)
    empty = frame["text_clean"].str.len() == 0
    report["empty_after_cleaning"] = int(empty.sum())
    frame = frame[~empty].copy()
    conflicts = frame.groupby("text_clean")["label"].transform("nunique") > 1
    report["conflicting_rows"] = int(conflicts.sum())
    frame = frame[~conflicts].copy()
    duplicates = frame.duplicated("text_clean")
    report["duplicate_rows"] = int(duplicates.sum())
    frame = frame[~duplicates].copy()
    frame["sentiment"] = frame["label"].map(dict(enumerate(CLASS_NAMES)))
    frame["text_length"] = frame["text"].str.len()
    frame["word_count"] = frame["text"].str.split().str.len()
    frame["word_count_clean"] = frame["text_clean"].str.split().str.len()
    frame["char_count"] = frame["text_clean"].str.len()
    frame["dollar_count"] = frame["text_clean"].str.count(r"\$")
    counts = frame["label"].value_counts()
    if set(counts.index) != {0, 1, 2} or counts.min() < 5:
        raise ValueError("Для обучения нужны все три класса, минимум по 5 строк")
    report["output_rows"] = len(frame)
    report["class_counts"] = frame["sentiment"].value_counts().to_dict()
    return frame.reset_index(drop=True), report
