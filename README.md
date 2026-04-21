# Core ML Image Classifier

A small, modular Python project demonstrating a mobile-first image
classification pipeline built around MobileNetV2:

- **`ImageClassifier`** — a MobileNetV2 classifier that returns top-k
  ImageNet predictions with confidence scores and provides a one-call
  export to Core ML for on-device iOS/macOS deployment.
- **`ImageIndexer`** — a content-based image retrieval index that
  reuses the MobileNetV2 backbone as a feature extractor and ranks
  candidates by cosine similarity over 1280-dimensional embeddings.

## Migration Note

This repository contains consolidated research and algorithmic logic from my
2024-2025 AI/ML focus period. It has been moved to this public repo to serve
as a technical portfolio for on-device and backend intelligence patterns.

## Project Structure

```
core_ml_image_classifier/
├── data/                     # Sample catalogue and generated demo images
│   ├── catalogue.csv
│   └── images/
├── models/                   # Persisted weights and Core ML bundles (runtime)
├── src/                      # Library code
│   ├── __init__.py
│   ├── image_classifier.py
│   └── image_indexer.py
├── tests/                    # Pytest suite
│   ├── test_image_classifier.py
│   └── test_image_indexer.py
├── main.py                   # Runnable demo entry point
├── requirements.txt
└── README.md
```

## Getting Started

```bash
cd core_ml_image_classifier
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Demo

```bash
python main.py
```

This generates a small set of synthetic colour-tile images under
`data/images/`, classifies each one with MobileNetV2, saves the PyTorch
weights to `models/mobilenet_v2.pt`, then fits an `ImageIndexer` over
`data/catalogue.csv` and prints nearest-neighbour recommendations for a
seed item and a query image.

## Running the Tests

```bash
pytest
```

## Programmatic Usage

```python
from src import ImageClassifier, ImageIndexer

classifier = ImageClassifier()
classifier.classify("cat.jpg", top_k=3)
# [("tabby, tabby cat", 0.78), ("Egyptian cat", 0.12), ("tiger cat", 0.05)]

classifier.export_to_coreml("models/mobilenet_v2.mlpackage")

indexer = ImageIndexer(classifier=classifier)
indexer.fit(
    ["tabby", "labrador"],
    ["images/cat.jpg", "images/dog.jpg"],
)
indexer.recommend_from_image("images/query.jpg")
```

## Design Notes

- Classes follow a `fit` / `predict`-style API that mirrors
  scikit-learn so they compose cleanly with the rest of the ML
  ecosystem.
- Preprocessing, model weights, and inference are kept inside the
  classifier, while I/O boundaries (image loading, CSV parsing, model
  persistence) stay at the edges so the core classes remain small and
  easy to test.
- `ImageClassifier.extract_features` exposes the 1280-D penultimate
  embedding so the indexer can reuse the same forward pass without
  duplicating the backbone.
- `ImageClassifier.export_to_coreml` bakes the ImageNet preprocessing
  into the exported model so the Swift call site can hand a
  `CVPixelBuffer` straight through, keeping the on-device integration
  minimal and the inference loop free of per-frame Python-equivalent
  work.
- The similarity index is a readable baseline rather than a production
  visual-search system; it is a useful starting point for richer
  retrieval strategies such as ANN indexes or learned re-rankers.
