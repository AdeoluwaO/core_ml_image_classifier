"""Tests for the ImageIndexer class."""

import os
import sys

import pytest
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.image_classifier import ImageClassifier
from src.image_indexer import ImageIndexer


@pytest.fixture(scope="module")
def classifier():
    """Share a single MobileNetV2 backbone across the test module."""
    return ImageClassifier()


@pytest.fixture(scope="module")
def catalogue(tmp_path_factory):
    """Return a catalogue of visually distinct solid-colour tiles."""
    folder = tmp_path_factory.mktemp("catalogue")
    palette = {
        "red_tile": (220, 30, 30),
        "blue_tile": (30, 30, 220),
        "green_tile": (30, 220, 30),
        "yellow_tile": (220, 220, 30),
    }
    items, paths = [], []
    for name, color in palette.items():
        path = folder / f"{name}.jpg"
        Image.new("RGB", (224, 224), color).save(path)
        items.append(name)
        paths.append(str(path))
    return items, paths


def test_recommend_excludes_seed(classifier, catalogue):
    """The seed item must not appear in its own recommendations."""
    items, paths = catalogue
    indexer = ImageIndexer(classifier=classifier)
    indexer.fit(items, paths)

    recommendations = indexer.recommend("red_tile", top_k=3)
    assert all(item != "red_tile" for item, _ in recommendations)
    assert len(recommendations) == 3


def test_recommend_from_image_ranks_similar_first(classifier, catalogue, tmp_path):
    """A red query image should rank the red catalogue tile first."""
    items, paths = catalogue
    indexer = ImageIndexer(classifier=classifier)
    indexer.fit(items, paths)

    query_path = tmp_path / "query.jpg"
    Image.new("RGB", (224, 224), (230, 40, 40)).save(query_path)

    results = indexer.recommend_from_image(str(query_path), top_k=1)
    assert results[0][0] == "red_tile"


def test_unknown_seed_item_raises(classifier, catalogue):
    """Querying an item not in the catalogue should raise KeyError."""
    items, paths = catalogue
    indexer = ImageIndexer(classifier=classifier)
    indexer.fit(items, paths)
    with pytest.raises(KeyError):
        indexer.recommend("not_in_catalogue")


def test_fit_requires_matching_lengths(classifier):
    """Mismatched items and paths should raise ValueError."""
    indexer = ImageIndexer(classifier=classifier)
    with pytest.raises(ValueError):
        indexer.fit(["only_one"], ["a.jpg", "b.jpg"])


def test_recommend_before_fit_raises(classifier):
    """Calling recommend_from_image without fitting should raise RuntimeError."""
    indexer = ImageIndexer(classifier=classifier)
    with pytest.raises(RuntimeError):
        indexer.recommend_from_image("anything.jpg")
