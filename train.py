"""Полный воспроизводимый цикл: EDA → CV-сравнение → тест → сохранение."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import sklearn
from sklearn.model_selection import train_test_split

from eda import describe_data
from error_analysis import analyze_errors, metrics
from load_data import DEFAULT_DATA, ROOT, load_data
from predict import predict_sentiment
from preprocess import CLASS_NAMES, LABELS, prepare_data
from train_improved import compare_models


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def split_data(frame):
    development, test = train_test_split(
        frame, test_size=0.2, stratify=frame.label, random_state=42
    )
    return development.copy(), test.copy()


def train(input_path=DEFAULT_DATA, output=ROOT):
    output = Path(output)
    reports, models_path = output / "reports", output / "models"
    reports.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)
    raw = load_data(input_path)
    clean, cleaning = prepare_data(raw)
    describe_data(raw, clean, cleaning, reports)
    development, test = split_data(clean)
    assert set(development.text_clean).isdisjoint(test.text_clean)
    splits = {
        "seed": 42,
        "test_fraction": 0.2,
        "cv_folds": 3,
        "development_row_ids": sorted(development.row_id.tolist()),
        "test_row_ids": sorted(test.row_id.tolist()),
        "development_classes": development.sentiment.value_counts().to_dict(),
        "test_classes": test.sentiment.value_counts().to_dict(),
    }
    write_json(reports / "splits.json", splits)
    models, comparison, searches, winner = compare_models(
        development.text.tolist(), development.label.tolist()
    )
    comparison.to_csv(reports / "comparison.csv", index=False)
    write_json(reports / "cv_search.json", searches)
    lines = [
        "# Сравнение методов",
        "",
        "Одинаковые 3 стратифицированных разбиения обучающих 80% данных.",
        "В каждом разбиении словарь TF-IDF обучается заново только на обучающей части.",
        "Параметры и победитель выбираются по средней Macro F1; тест не участвует.",
        "После выбора baseline и победитель обучены на всех обучающих 80%.",
        "",
        "| Метод | CV Macro F1 | Std | Прирост, п.п. | Прирост, % |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples():
        lines.append(
            f"| {row.method} | {row.cv_macro_f1:.4f} | {row.cv_std:.4f} | "
            f"{row.delta_pp:+.2f} | {row.relative_gain_percent:+.2f} |"
        )
    (reports / "comparison.md").write_text(
        "\n".join(lines) + f"\n\nВыбран: `{winner}`.\n", encoding="utf-8"
    )
    # Отложенный тест впервые используется после окончательного выбора модели.
    baseline_metrics = metrics(test.label, models["baseline"].predict(test.text))
    selected = models[winner]
    predictions = selected.predict(test.text)
    best_metrics = metrics(test.label, predictions)
    baseline_f1, best_f1 = baseline_metrics["macro_f1"], best_metrics["macro_f1"]
    summary = {
        "selected_method": winner,
        "selection": "mean development-only 3-fold CV macro F1",
        "baseline_test": baseline_metrics,
        "best_test": best_metrics,
        "absolute_gain_pp": (best_f1 - baseline_f1) * 100,
        "relative_gain_percent": (best_f1 / baseline_f1 - 1) * 100,
        "target_relative_5_percent_met": best_f1 >= baseline_f1 * 1.05,
        "target_absolute_5_pp_met": best_f1 >= baseline_f1 + 0.05,
    }
    write_json(reports / "metrics.json", summary)
    (reports / "baseline_result.txt").write_text(
        f"Baseline test Macro F1: {baseline_f1:.6f}\n", encoding="utf-8"
    )
    analyze_errors(test, predictions, reports)
    joblib.dump(selected, models_path / "best_model.joblib", compress=3)
    joblib.dump(
        selected.named_steps["features"], models_path / "vectorizer.joblib", compress=3
    )
    metadata = {
        "selected_method": winner,
        "labels": LABELS,
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "dataset_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
        "training_rows": len(development),
        "test_rows": len(test),
        "cv": searches[winner],
        "test_macro_f1": best_f1,
        "artifacts": {
            name: hashlib.sha256((models_path / name).read_bytes()).hexdigest()
            for name in ["best_model.joblib", "vectorizer.joblib"]
        },
    }
    write_json(models_path / "metadata.json", metadata)
    examples = json.loads((ROOT / "examples/demo.json").read_text())
    restored = joblib.load(models_path / "best_model.joblib")
    texts = [row["text"] for row in examples]
    assert list(restored.predict(texts)) == list(selected.predict(texts))
    demos = predict_sentiment(texts, model=restored)
    for row, example in zip(demos, examples, strict=True):
        row["expected_sentiment"] = example["expected_sentiment"]
    write_json(reports / "demo_predictions.json", demos)
    print(
        json.dumps(
            {
                "winner": winner,
                "classes": CLASS_NAMES,
                "baseline_f1": baseline_f1,
                "best_f1": best_f1,
                "gain_pp": summary["absolute_gain_pp"],
                "relative_gain_percent": summary["relative_gain_percent"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Обучить и сравнить модели тональности"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    train(args.input, args.output)
