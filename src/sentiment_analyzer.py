"""
Sentiment Analyzer using TF-IDF + Logistic Regression.

I picked Logistic Regression because it was the first classifier that actually
made sense to me when I was learning scikit-learn. TF-IDF turns the text into
numbers and Logistic Regression draws the line between positive and negative.
"""

import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


class SentimentAnalyzer:
    """A simple sentiment classifier (positive vs negative)."""

    def __init__(self, max_features=5000):
        """
        Build the pipeline.

        max_features: how big the vocabulary can get. 5000 felt like a
        reasonable starting point for small datasets.
        """
        # I learned that putting the vectorizer and classifier into a single
        # Pipeline makes life way easier - no more manually transforming the
        # text every time I want to predict something.
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),       # unigrams + bigrams worked better
                stop_words="english",
                lowercase=True,
            )),
            ("classifier", LogisticRegression(max_iter=1000)),
        ])
        self.is_trained = False

    def train(self, texts, labels, test_size=0.2):
        """
        Train the model on a list of texts and labels.

        Returns a dict with the test accuracy and a classification report
        so I can see how well it actually did.
        """
        if len(texts) != len(labels):
            raise ValueError("texts and labels must be the same length")

        # Split into train/test so we don't cheat by testing on training data
        X_train, X_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=42,
            stratify=labels,
        )

        self.pipeline.fit(X_train, y_train)
        self.is_trained = True

        predictions = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, zero_division=0)

        return {
            "accuracy": accuracy,
            "report": report,
        }

    def predict(self, texts):
        """Predict labels for a list of texts."""
        if not self.is_trained:
            raise RuntimeError("You need to train the model first!")
        return list(self.pipeline.predict(texts))

    def predict_score(self, text):
        """
        Get the probability that a single text is positive.

        Useful later for the recommender - I can boost items that have
        more positive reviews.
        """
        if not self.is_trained:
            raise RuntimeError("You need to train the model first!")

        # predict_proba gives probabilities for each class.
        # I had to look up which column was "positive" - it depends on
        # the order in classes_.
        classes = list(self.pipeline.named_steps["classifier"].classes_)
        positive_index = classes.index("positive")
        probabilities = self.pipeline.predict_proba([text])[0]
        return float(probabilities[positive_index])

    def save(self, path):
        """Save the trained model to disk so we don't retrain every time."""
        if not self.is_trained:
            raise RuntimeError("Nothing to save - model isn't trained.")

        # make the folder if it doesn't exist
        folder = os.path.dirname(path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)

        joblib.dump(self.pipeline, path)

    def load(self, path):
        """Load a previously trained model."""
        self.pipeline = joblib.load(path)
        self.is_trained = True
