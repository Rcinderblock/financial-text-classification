import pandas as pd
import pytest

from preprocess import clean_text, label_code, prepare_data


def sample_frame():
    return pd.DataFrame(
        [
            {"text": f"{word} report number {index}", "sentiment": name}
            for name, word in [
                ("negative", "loss"),
                ("neutral", "office"),
                ("positive", "growth"),
            ]
            for index in range(10)
        ]
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" <p>Profit   rose 5% &amp; sales $10.</p> ", "profit rose 5% & sales $10."),
        ("Profit did NOT rise.\nLoss was -$5.", "profit did not rise. loss was -$5."),
        ("Profit < 30 and revenue > 50", "profit < 30 and revenue > 50"),
        ("Ｆｉｎａｎｃｅ\tNews", "finance news"),
    ],
)
def test_cleanup_preserves_financial_information(raw, expected):
    assert clean_text(raw) == expected
    assert clean_text(expected) == expected


@pytest.mark.parametrize("bad", [None, 42, [], float("nan")])
def test_cleanup_rejects_nontexts(bad):
    with pytest.raises(ValueError):
        clean_text(bad)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(" NEGATIVE ", 0), ("neutral", 1), ("positive", 2), (0, 0), ("2", 2)],
)
def test_label_mapping(value, expected):
    assert label_code(value) == expected


@pytest.mark.parametrize("value", ["unknown", "", 3, True])
def test_unknown_labels_rejected(value):
    with pytest.raises(ValueError):
        label_code(value)


def test_missing_duplicate_empty_and_conflicting_rows_removed():
    frame = sample_frame()
    extras = pd.DataFrame(
        [
            {"text": None, "sentiment": "positive"},
            {"text": "<p> </p>", "sentiment": "negative"},
            {"text": "LOSS   REPORT NUMBER 0", "sentiment": "negative"},
            {"text": "conflicting text", "sentiment": "positive"},
            {"text": "conflicting text", "sentiment": "negative"},
        ]
    )
    cleaned, report = prepare_data(pd.concat([frame, extras], ignore_index=True))
    assert len(cleaned) == len(frame)
    assert report["missing_or_nontext_rows"] == 1
    assert report["empty_after_cleaning"] == 1
    assert report["duplicate_rows"] == 1
    assert report["conflicting_rows"] == 2
    assert cleaned.text_clean.is_unique
    assert {"word_count_clean", "char_count", "dollar_count"} <= set(cleaned)


def test_aliases_and_bad_schema():
    frame = sample_frame().rename(columns={"text": "headline", "sentiment": "label"})
    cleaned, _ = prepare_data(frame)
    assert len(cleaned) == 30
    with pytest.raises(ValueError, match="колонки"):
        prepare_data(pd.DataFrame({"wrong": ["x"]}))
    frame = sample_frame()
    frame.loc[0, "sentiment"] = "unexpected"
    with pytest.raises(ValueError, match="Метка"):
        prepare_data(frame)
