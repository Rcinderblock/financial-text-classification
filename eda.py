"""Исходные данные, распределения классов и длин, примеры очистки."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from preprocess import CLASS_NAMES


def describe_data(raw: pd.DataFrame, clean: pd.DataFrame, cleaning: dict, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "raw_shape": list(raw.shape),
        "raw_dtypes": raw.dtypes.astype(str).to_dict(),
        "raw_missing": raw.isna().sum().to_dict(),
        "cleaning": cleaning,
        "lengths": clean[["text_length", "word_count"]].describe().to_dict(),
        "examples": {
            name: clean.loc[clean.sentiment == name, "text"].iloc[0]
            for name in CLASS_NAMES
        },
    }
    (output / "eda.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    raw.head().to_csv(output / "first_rows.csv", index=False)
    clean.groupby("sentiment", sort=False).head(2).to_csv(
        output / "preprocessing_examples.csv", index=False
    )
    charts = output / "charts"
    charts.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = clean.sentiment.value_counts().reindex(CLASS_NAMES)
    ax.bar(counts.index, counts.values, color=["#c75b55", "#7195a5", "#4e9976"])
    ax.set(title="Sentiment distribution", ylabel="News sentences")
    fig.tight_layout()
    fig.savefig(charts / "sentiment_distribution.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, column, title in zip(
        axes, ["text_length", "word_count"], ["Characters", "Words"], strict=True
    ):
        ax.hist(clean[column], bins=40, color="#537b9e", edgecolor="white")
        ax.set(title=title, ylabel="News sentences")
    fig.tight_layout()
    fig.savefig(charts / "text_length_distribution.png", dpi=150)
    plt.close(fig)
    conclusions = [
        "Первичный анализ данных",
        f"Исходный размер: {raw.shape[0]} строк, {raw.shape[1]} столбца.",
        f"После очистки: {len(clean)} строк.",
        f"Классы: {counts.to_dict()}.",
        f"Отношение размера самого частого класса к редкому: "
        f"{counts.max() / counts.min():.2f}.",
        "Основная метрика — Macro F1: каждый класс имеет одинаковый вес.",
        f"Удалено строк с пропусками или нетекстовыми значениями: "
        f"{cleaning['missing_or_nontext_rows']}.",
        f"Удалено пустых текстов после очистки: {cleaning['empty_after_cleaning']}.",
        f"Удалено строк с конфликтными метками: {cleaning['conflicting_rows']}.",
        f"Удалено дубликатов очищенных текстов: {cleaning['duplicate_rows']}.",
        "Слова отрицания, числа и финансовые обозначения сохранены при очистке.",
        "Показатели длин описывают строки; они не доказывают причину ошибок.",
    ]
    (output / "eda.txt").write_text("\n".join(conclusions) + "\n", encoding="utf-8")
