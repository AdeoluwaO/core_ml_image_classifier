"""
Tests for the ContentRecommender.

Toy catalog with three obvious clusters: headphones, keyboards, and a mouse.
If the recommender works, headphone queries should return headphones and
keyboard queries should return keyboards.
"""

import pytest

from src.content_recommender import ContentRecommender


ITEM_IDS = ["a1", "a2", "b1", "b2", "c1"]
DESCRIPTIONS = [
    "wireless bluetooth headphones with noise cancelling",
    "over ear headphones with deep bass and bluetooth",
    "mechanical keyboard with rgb lighting",
    "compact mechanical keyboard with hot swap switches",
    "wireless ergonomic mouse with adjustable dpi",
]


def test_recommend_returns_similar_items_first():
    rec = ContentRecommender()
    rec.fit(ITEM_IDS, DESCRIPTIONS)

    results = rec.recommend("a1", top_k=2)
    top_ids = [r["item_id"] for r in results]

    # The other headphone should rank above any keyboard or mouse
    assert "a2" in top_ids


def test_recommend_excludes_query_item():
    rec = ContentRecommender()
    rec.fit(ITEM_IDS, DESCRIPTIONS)

    results = rec.recommend("b1", top_k=4)
    returned_ids = [r["item_id"] for r in results]
    assert "b1" not in returned_ids


def test_recommend_respects_top_k():
    rec = ContentRecommender()
    rec.fit(ITEM_IDS, DESCRIPTIONS)

    results = rec.recommend("a1", top_k=3)
    assert len(results) == 3


def test_unknown_item_raises():
    rec = ContentRecommender()
    rec.fit(ITEM_IDS, DESCRIPTIONS)

    with pytest.raises(KeyError):
        rec.recommend("does_not_exist")


def test_recommend_before_fit_raises():
    rec = ContentRecommender()
    with pytest.raises(RuntimeError):
        rec.recommend("a1")


def test_invalid_sentiment_weight_raises():
    with pytest.raises(ValueError):
        ContentRecommender(sentiment_weight=1.5)


def test_sentiment_boost_changes_ranking():
    """
    If two items are equally similar to the query, the one with higher
    sentiment should rank above the one with lower sentiment.
    """
    ids = ["q", "x", "y"]
    descriptions = [
        "wireless bluetooth headphones",
        "wireless bluetooth headphones",   # identical to query
        "wireless bluetooth headphones",   # also identical
    ]
    sentiment = [0.5, 0.1, 0.9]   # y is much more positive than x

    rec = ContentRecommender(sentiment_weight=0.5)
    rec.fit(ids, descriptions, sentiment_scores=sentiment)

    results = rec.recommend("q", top_k=2)
    assert results[0]["item_id"] == "y"
