# NLP Sentiment Analysis + Content Recommender

A small Python project I built while learning NLP and recommendation systems
with scikit-learn. It has two pieces:

1. **`SentimentAnalyzer`** – a TF-IDF + Logistic Regression classifier that
   labels text as positive or negative.
2. **`ContentRecommender`** – a TF-IDF cosine-similarity recommender that
   can optionally blend in sentiment scores so highly-rated items get a
   small ranking boost.

The two pieces work together in `main.py`: the analyzer scores each item's
review, and the recommender uses those scores to bias its rankings.

## Migration Note

This repository contains consolidated research and algorithmic logic from
my 2024-2025 AI/ML focus period. It has been moved to this public repo to
serve as a technical portfolio for on-device and backend intelligence
patterns.

## Project Structure

```
.
├── data/                       # sample CSV datasets
│   ├── sample_reviews.csv
│   └── sample_items.csv
├── models/                     # trained models get saved here
├── src/                        # the actual classes
│   ├── __init__.py
│   ├── sentiment_analyzer.py
│   └── content_recommender.py
├── tests/                      # pytest tests for both classes
│   ├── __init__.py
│   ├── test_sentiment_analyzer.py
│   └── test_content_recommender.py
├── main.py                     # entry point that ties everything together
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

This will:
1. Train the sentiment model on `data/sample_reviews.csv`
2. Save the trained model to `models/sentiment_model.joblib`
3. Score each item in `data/sample_items.csv` with the trained model
4. Build the recommender and print a few example recommendations

## Running the Tests

```bash
pytest -v
```

## Things I Learned Building This

- Wrapping the vectorizer and classifier in a single `Pipeline` removes a
  whole class of bugs where I forgot to transform the input the same way
  twice.
- `predict_proba` returns columns in the order of `classifier.classes_`,
  which is alphabetical by default – not the order I passed labels in.
  I had to look up the index of `"positive"` rather than assuming it was 0 or 1.
- Cosine similarity on TF-IDF vectors is a surprisingly strong baseline
  for content-based recommendations on small catalogs.
- Blending sentiment with similarity is a heuristic, not a real model –
  but it's a useful one when you don't have user interaction data yet.

## Notes / Caveats

- The included datasets are tiny and synthetic. They're enough to
  demonstrate the pipeline but not enough to produce a meaningful
  accuracy number.
- This is a learning project, so the focus is on clarity over performance.
