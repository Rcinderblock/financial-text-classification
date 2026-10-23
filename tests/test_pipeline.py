import json

import joblib
import numpy as np
import pandas as pd
import pytest

from predict import main, predict_sentiment
from preprocess import prepare_data
from tests.test_preprocess import sample_frame
from train import split_data, train
from train_baseline import build_baseline


def test_split_is_reproducible_and_disjoint():
    frame, _ = prepare_data(sample_frame())
    development, test = split_data(frame)
    again, _ = split_data(frame)
    assert development.row_id.tolist() == again.row_id.tolist()
    assert set(development.text_clean).isdisjoint(test.text_clean)
    assert set(test.label) == {0, 1, 2}


def test_vectorizer_does_not_learn_test_words_and_pipeline_reloads(tmp_path):
    frame, _ = prepare_data(sample_frame())
    model = build_baseline().fit(frame.text, frame.label)
    vocabulary = model.named_steps["features"].vocabulary_.copy()
    prediction = model.predict(["noveltesttoken excellent growth"])
    assert "noveltesttoken" not in vocabulary
    assert vocabulary == model.named_steps["features"].vocabulary_
    assert np.array_equal(
        model.predict(["<b>GROWTH</b> report"]), model.predict(["growth report"])
    )
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    assert np.array_equal(
        prediction, restored.predict(["noveltesttoken excellent growth"])
    )
    rows = predict_sentiment("loss report", model=restored)
    assert len(rows) == 1 and rows[0]["label"] == 0
    assert set(rows[0]["probabilities"]) == {"negative", "neutral", "positive"}
    assert sum(rows[0]["probabilities"].values()) == pytest.approx(1)


@pytest.mark.parametrize(
    "value", ["", "<p> </p>", [], ["okay", None], [42], "a" * 20001]
)
def test_bad_inference_inputs_rejected_before_loading(value):
    with pytest.raises(ValueError):
        predict_sentiment(value, model_path="missing.joblib")


def test_full_workflow_on_small_independent_dataset(tmp_path, capsys):
    csv = tmp_path / "data.csv"
    sample_frame().to_csv(csv, index=False)
    summary = train(csv, tmp_path / "run")
    output = tmp_path / "run"
    comparison = pd.read_csv(output / "reports/comparison.csv")
    assert len(comparison) == 5
    assert (
        summary["selected_method"]
        == comparison.sort_values(
            ["cv_macro_f1", "method"], ascending=[False, True], kind="stable"
        ).iloc[0]["method"]
    )
    assert (output / "reports/charts/confusion_matrix.png").is_file()
    assert (output / "reports/errors.csv").is_file()
    assert (output / "reports/error_analysis.txt").is_file()
    assert (output / "models/vectorizer.joblib").is_file()
    path = output / "models/best_model.joblib"
    model = joblib.load(path)
    rows = predict_sentiment(["loss report", "growth report"], model=model)
    assert len(rows) == 2
    capsys.readouterr()
    assert main(["--model", str(path), "--text", "growth report"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 1
