"""Зафиксированная базовая модель TF-IDF (1, 2) + Logistic Regression."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from preprocess import preprocess_texts


def make_pipeline(features=None, classifier=None):
    return Pipeline(
        [
            ("clean", FunctionTransformer(preprocess_texts, validate=False)),
            (
                "features",
                features
                if features is not None
                else TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
            ),
            (
                "classifier",
                classifier
                if classifier is not None
                else LogisticRegression(max_iter=2000, random_state=42),
            ),
        ]
    )


def build_baseline():
    return make_pipeline()
