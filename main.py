"""
Entry point for the project.

This script:
  1. Loads the sample review data
  2. Trains the SentimentAnalyzer and saves the model
  3. Loads the sample item catalog
  4. Scores each item's review with the trained model
  5. Builds the ContentRecommender and prints recommendations

I wanted main.py to be runnable end-to-end so that anyone cloning the
repo can just do `python main.py` and see the whole thing work.
"""

import os
import pandas as pd

from src.sentiment_analyzer import SentimentAnalyzer
from src.content_recommender import ContentRecommender


REVIEWS_PATH = os.path.join("data", "sample_reviews.csv")
ITEMS_PATH = os.path.join("data", "sample_items.csv")
MODEL_PATH = os.path.join("models", "sentiment_model.joblib")


def train_sentiment_model():
    """Load reviews, train the analyzer, save it, and return the trained one."""
    print("=== Step 1: Training sentiment model ===")
    df = pd.read_csv(REVIEWS_PATH)
    print("Loaded", len(df), "reviews from", REVIEWS_PATH)

    analyzer = SentimentAnalyzer()
    results = analyzer.train(df["text"].tolist(), df["label"].tolist())

    print("Test accuracy:", round(results["accuracy"], 3))
    print("Classification report:\n", results["report"])

    analyzer.save(MODEL_PATH)
    print("Saved model to", MODEL_PATH)
    return analyzer


def build_recommender(analyzer):
    """Score items with the analyzer, then fit the recommender."""
    print("\n=== Step 2: Scoring items with sentiment model ===")
    items = pd.read_csv(ITEMS_PATH)
    print("Loaded", len(items), "items from", ITEMS_PATH)

    # For each item, run the review through the sentiment model to get
    # a positive-ness score. This becomes the bias signal for the recommender.
    sentiment_scores = []
    for review in items["review"].tolist():
        score = analyzer.predict_score(review)
        sentiment_scores.append(score)

    items["sentiment_score"] = sentiment_scores
    print(items[["item_id", "sentiment_score"]].to_string(index=False))

    print("\n=== Step 3: Fitting recommender ===")
    recommender = ContentRecommender(sentiment_weight=0.3)
    recommender.fit(
        item_ids=items["item_id"].tolist(),
        descriptions=items["description"].tolist(),
        sentiment_scores=sentiment_scores,
    )
    return recommender


def show_recommendations(recommender, query_id, top_k=3):
    """Print recommendations for a given item id."""
    print("\n=== Recommendations for", query_id, "===")
    results = recommender.recommend(query_id, top_k=top_k)
    for r in results:
        print(
            "  ->", r["item_id"],
            "| score:", r["score"],
            "| similarity:", r["similarity"],
        )


def main():
    """Glue everything together."""
    analyzer = train_sentiment_model()
    recommender = build_recommender(analyzer)

    # Try a couple of queries so we can see different behavior
    show_recommendations(recommender, "item_01", top_k=3)
    show_recommendations(recommender, "item_04", top_k=3)
    show_recommendations(recommender, "item_06", top_k=3)


if __name__ == "__main__":
    main()
