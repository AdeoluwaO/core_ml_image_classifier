"""Tests for the ImageClassifier class."""

import os
import sys
import tempfile

import pytest
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.image_classifier import ImageClassifier


@pytest.fixture(scope="module")
def classifier():
    """Load MobileNetV2 once per module to amortise the weight download."""
    return ImageClassifier()


@pytest.fixture
def sample_image(tmp_path):
    """Return a path to a small synthetic test image."""
    path = tmp_path / "sample.jpg"
    Image.new("RGB", (256, 256), (120, 80, 200)).save(path)
    return str(path)


def test_classify_returns_top_k_predictions(classifier, sample_image):
    """Classify should return top_k (label, confidence) tuples in order."""
    predictions = classifier.classify(sample_image, top_k=3)
    assert len(predictions) == 3
    confidences = [confidence for _, confidence in predictions]
    assert confidences == sorted(confidences, reverse=True)
    for label, confidence in predictions:
        assert isinstance(label, str)
        assert 0.0 <= confidence <= 1.0


def test_classify_batch_matches_single(classifier, sample_image):
    """Batch classify should produce one result list per image."""
    batch = classifier.classify_batch([sample_image, sample_image], top_k=5)
    assert len(batch) == 2
    assert batch[0] == batch[1]


def test_classify_accepts_pil_image(classifier):
    """Classify should accept PIL images directly, not only paths."""
    image = Image.new("RGB", (256, 256), (50, 150, 200))
    predictions = classifier.classify(image, top_k=2)
    assert len(predictions) == 2


def test_extract_features_shape(classifier, sample_image):
    """Features should have 1280 channels after global average pooling."""
    features = classifier.extract_features([sample_image, sample_image])
    assert features.shape == (2, 1280)


def test_labels_are_exposed(classifier):
    """The labels property should expose the 1000 ImageNet classes."""
    labels = classifier.labels
    assert len(labels) == 1000
    assert all(isinstance(label, str) for label in labels)


def test_save_and_load_roundtrip(classifier, sample_image):
    """A saved checkpoint should reproduce identical predictions."""
    original = classifier.classify(sample_image, top_k=3)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "weights.pt")
        classifier.save(path)

        restored = ImageClassifier()
        restored.load(path)
        assert restored.classify(sample_image, top_k=3) == original


def test_empty_batch_raises(classifier):
    """Passing no images should raise a clear error."""
    with pytest.raises(ValueError):
        classifier.classify_batch([])
