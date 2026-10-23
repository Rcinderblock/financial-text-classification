"""Метрики и реальные ошибки на отложенной выборке."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from preprocess import CLASS_NAMES


def metrics(labels, predictions):
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels, predictions, labels=[0, 1, 2], average="macro", zero_division=0
            )
        ),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1, 2]
        ).tolist(),
    }


def analyze_errors(test, predictions, output: Path):
    result = test.copy()
    result["predicted_label"] = predictions
    result["predicted_sentiment"] = [CLASS_NAMES[int(value)] for value in predictions]
    result["correct"] = result.label == result.predicted_label
    result.to_csv(output / "test_predictions.csv", index=False)
    errors = result[~result.correct]
    errors.to_csv(output / "errors.csv", index=False)
    matrix = confusion_matrix(test.label, predictions, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set(xlabel="Predicted sentiment", ylabel="True sentiment", title="Test errors")
    fig.tight_layout()
    fig.savefig(output / "charts/confusion_matrix.png", dpi=150)
    plt.close(fig)
    lines = [
        "Анализ ошибок на отложенной тестовой выборке",
        f"Ошибок: {len(errors)} из {len(test)}.",
        "Матрица: строки — истинный класс, столбцы — предсказанный.",
        str(matrix),
        "",
        "Ошибки по направлениям:",
    ]
    for (true, predicted), group in errors.groupby(
        ["sentiment", "predicted_sentiment"], sort=True
    ):
        lines.append(f"{true} → {predicted}: {len(group)}")
        for row in group.head(2).itertuples():
            lines.append(f"  row_id={row.row_id}: {row.text}")
    lines.extend(["", "Доля ошибок по истинному классу:"])
    for name, group in result.groupby("sentiment", sort=True):
        lines.append(f"{name}: {(~group.correct).mean():.1%} ({len(group)} примеров)")
    number_mask = result.text.str.contains(r"\d", regex=True)
    lines.append("\nПроверка групп (описание, а не доказательство причин):")
    for name, mask in [("С числами", number_mask), ("Без чисел", ~number_mask)]:
        rate = float(np.mean(~result.loc[mask, "correct"])) if mask.any() else 0.0
        lines.append(f"{name}: {rate:.1%} ошибок, n={int(mask.sum())}")
    lines.extend(
        [
            "",
            "Гипотезы для следующего эксперимента:",
            "Проверить признаки финансовых сравнений: прибыль/убыток, рост/снижение.",
            "Разобрать влияние упоминаний нескольких периодов и смешанной тональности.",
            "Попробовать контекстную модель, сравнивая её на новом отложенном наборе.",
            "Дальнейшие изменения нельзя выбирать по уже просмотренному тесту.",
        ]
    )
    (output / "error_analysis.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return errors
