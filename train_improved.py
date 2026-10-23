"""Сравнение вариантов только на обучающей части; тест сюда не передаётся."""

import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from train_baseline import build_baseline, make_pipeline


def experiment_specs():
    def words():
        return TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

    def svm():
        return LinearSVC(class_weight="balanced", random_state=42, max_iter=10000)

    combined = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    max_features=12000, ngram_range=(1, 2), sublinear_tf=True
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=25000,
                    sublinear_tf=True,
                ),
            ),
        ],
        transformer_weights={"word": 1.0, "char": 0.7},
    )
    return [
        ("baseline", build_baseline(), {}),
        (
            "word_trigrams",
            make_pipeline(TfidfVectorizer(max_features=5000, ngram_range=(1, 3))),
            {},
        ),
        (
            "balanced_logreg",
            make_pipeline(
                words(),
                LogisticRegression(
                    class_weight="balanced", max_iter=2000, random_state=42
                ),
            ),
            {"classifier__C": [0.1, 1.0, 10.0]},
        ),
        (
            "linear_svc",
            make_pipeline(words(), svm()),
            {"classifier__C": [0.5, 1.0, 2.0]},
        ),
        (
            "word_char_svc",
            make_pipeline(combined, svm()),
            {"classifier__C": [0.5, 1.0, 2.0]},
        ),
    ]


def compare_models(texts, labels):
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score, average="macro", labels=[0, 1, 2], zero_division=0)
    rows, models, searches = [], {}, {}
    for name, pipeline, parameters in experiment_specs():
        print(f"Обучение: {name}", flush=True)
        search = GridSearchCV(
            pipeline,
            parameters,
            scoring=scorer,
            cv=folds,
            n_jobs=1,
            error_score="raise",
            return_train_score=False,
        )
        search.fit(texts, labels)
        index = search.best_index_
        scores = [
            float(search.cv_results_[f"split{fold}_test_score"][index])
            for fold in range(3)
        ]
        rows.append(
            {
                "method": name,
                "cv_macro_f1": float(search.best_score_),
                "cv_std": float(search.cv_results_["std_test_score"][index]),
                "best_parameters": json.dumps(search.best_params_, sort_keys=True),
            }
        )
        models[name] = search.best_estimator_
        searches[name] = {
            "best_parameters": search.best_params_,
            "fold_scores": scores,
            "candidates": [
                {"parameters": params, "mean_macro_f1": float(mean)}
                for params, mean in zip(
                    search.cv_results_["params"],
                    search.cv_results_["mean_test_score"],
                    strict=True,
                )
            ],
        }
    comparison = pd.DataFrame(rows)
    baseline = comparison.iloc[0].cv_macro_f1
    comparison["delta_pp"] = (comparison.cv_macro_f1 - baseline) * 100
    comparison["relative_gain_percent"] = (
        (comparison.cv_macro_f1 / baseline - 1) * 100 if baseline else 0
    )
    winner = comparison.sort_values(
        ["cv_macro_f1", "method"], ascending=[False, True], kind="stable"
    ).iloc[0]["method"]
    return models, comparison, searches, winner
