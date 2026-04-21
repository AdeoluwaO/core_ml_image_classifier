"""
Content-based recommender.

The idea: given an item, find other items with similar descriptions.
We turn descriptions into TF-IDF vectors and use cosine similarity to
measure how "close" two items are.

I also wanted to try mixing in sentiment scores so that highly-rated items
get a small boost. That's the "heuristic" part - it's not a proper
collaborative filtering model, just a weighted blend.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentRecommender:
    """Recommend items based on description similarity (+ optional sentiment)."""

    def __init__(self, sentiment_weight=0.3):
        """
        sentiment_weight: how much the sentiment score matters compared to
        the similarity score. 0 = ignore sentiment, 1 = only sentiment.
        I picked 0.3 as a default after playing with values - it nudges
        results without overwhelming the similarity.
        """
        if sentiment_weight < 0 or sentiment_weight > 1:
            raise ValueError("sentiment_weight must be between 0 and 1")

        self.sentiment_weight = sentiment_weight
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            ngram_range=(1, 2),
        )

        # filled in by fit()
        self.item_ids = []
        self.descriptions = []
        self.sentiment_scores = None
        self.tfidf_matrix = None

    def fit(self, item_ids, descriptions, sentiment_scores=None):
        """
        Index a catalog of items.

        item_ids: list of unique IDs
        descriptions: list of text descriptions, one per item
        sentiment_scores: optional list of floats in [0, 1], one per item.
                         These come from the SentimentAnalyzer.
        """
        if len(item_ids) != len(descriptions):
            raise ValueError("item_ids and descriptions must be the same length")
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("item_ids must be unique")
        if sentiment_scores is not None:
            if len(sentiment_scores) != len(item_ids):
                raise ValueError("sentiment_scores must match item_ids length")

        self.item_ids = list(item_ids)
        self.descriptions = list(descriptions)
        self.sentiment_scores = (
            list(sentiment_scores) if sentiment_scores is not None else None
        )

        # fit_transform gives us the TF-IDF matrix for the whole catalog
        self.tfidf_matrix = self.vectorizer.fit_transform(self.descriptions)

    def recommend(self, item_id, top_k=5):
        """
        Return the top_k items most similar to item_id.

        Each result is a dict with: item_id, score, similarity.
        score = blended score, similarity = raw cosine similarity.
        """
        if self.tfidf_matrix is None:
            raise RuntimeError("Call fit() before recommend()")
        if item_id not in self.item_ids:
            raise KeyError("Unknown item_id: " + str(item_id))
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        # Find the index of the item we're querying
        query_index = self.item_ids.index(item_id)

        # Compute similarity between this item and every other item.
        # cosine_similarity returns a 2D array, so we flatten it with [0]
        similarities = cosine_similarity(
            self.tfidf_matrix[query_index],
            self.tfidf_matrix,
        )[0]

        # Build a list of (index, blended_score, similarity) tuples
        results = []
        for i in range(len(self.item_ids)):
            if i == query_index:
                continue  # skip the item itself

            sim = float(similarities[i])
            blended = self._blend(sim, i)
            results.append((i, blended, sim))

        # Sort by blended score, highest first
        results.sort(key=lambda r: r[1], reverse=True)

        top_results = results[:top_k]
        return [
            {
                "item_id": self.item_ids[i],
                "score": round(score, 4),
                "similarity": round(sim, 4),
            }
            for (i, score, sim) in top_results
        ]

    def _blend(self, similarity, index):
        """Mix the similarity with the sentiment score (if we have one)."""
        if self.sentiment_scores is None:
            return similarity

        sentiment = self.sentiment_scores[index]
        w = self.sentiment_weight
        return (1 - w) * similarity + w * sentiment
