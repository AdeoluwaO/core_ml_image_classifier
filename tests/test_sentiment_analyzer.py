"""
Tests for the SentimentAnalyzer.

I'm using small toy datasets here so the tests run fast. The point isn't to
prove the model is accurate (it can't be with this little data) - it's to
prove the class behaves the way I expect.
"""

import os
import tempfile
import pytest

from src.sentiment_analyzer import SentimentAnalyzer


# A tiny dataset that should be obviously separable.
TRAIN_TEXTS = [
    "great wonderful amazing love it",
    "fantastic excellent awesome super",
    "perfect happy good best ever",
    "love this so much amazing",
    "terrible awful hated bad",
    "worst horrible disappointing junk",
    "broken useless waste of money",
    "awful experience would not recommend",
]
TRAIN_LABELS = [
    "positive", "positive", "positive", "positive",
    "negative", "negative", "negative", "negative",
]


def make_trained_analyzer():
    """Helper - returns an analyzer trained on the toy dataset."""
    analyzer = SentimentAnalyzer()
    analyzer.train(TRAIN_TEXTS, TRAIN_LABELS, test_size=0.25)
    return analyzer


def test_predict_returns_label_per_input():
    analyzer = make_trained_analyzer()
    predictions = analyzer.predict(["amazing", "horrible"])
    assert len(predictions) == 2
    for p in predictions:
        assert p in ("positive", "negative")


def test_predict_score_is_probability():
    analyzer = make_trained_analyzer()
    score = analyzer.predict_score("amazing wonderful great")
    assert 0.0 <= score <= 1.0


def test_predict_before_training_raises():
    analyzer = SentimentAnalyzer()
    with pytest.raises(RuntimeError):
        analyzer.predict(["anything"])


def test_train_with_mismatched_lengths_raises():
    analyzer = SentimentAnalyzer()
    with pytest.raises(ValueError):
        analyzer.train(["one text"], ["positive", "negative"])


def test_save_and_load_round_trip():
    analyzer = make_trained_analyzer()

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.joblib")
        analyzer.save(path)
        assert os.path.exists(path)

        loaded = SentimentAnalyzer()
        loaded.load(path)

        # Both should agree on a clearly-positive sentence
        original = analyzer.predict(["amazing wonderful"])
        reloaded = loaded.predict(["amazing wonderful"])
        assert original == reloaded
