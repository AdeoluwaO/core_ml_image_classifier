"""Entry point demonstrating the ImageClassifier and ImageIndexer."""

import os

from PIL import Image

from src.image_classifier import ImageClassifier
from src.image_indexer import ImageIndexer

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
CATALOGUE_PATH = os.path.join(DATA_DIR, "catalogue.csv")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "models", "mobilenet_v2.pt")

SAMPLE_IMAGES = {
    "red_tile.jpg": (220, 30, 30),
    "blue_tile.jpg": (30, 30, 220),
    "green_tile.jpg": (30, 220, 30),
    "yellow_tile.jpg": (220, 220, 30),
    "query.jpg": (230, 40, 40),
}


def ensure_sample_images() -> None:
    """Create the demo colour-tile images on first run."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    for name, color in SAMPLE_IMAGES.items():
        path = os.path.join(IMAGES_DIR, name)
        if not os.path.exists(path):
            Image.new("RGB", (224, 224), color).save(path)


def run_classifier_demo(classifier: ImageClassifier) -> None:
    """Classify the demo images and print the top predictions for each."""
    print("=== Image Classification ===")
    samples = ["red_tile.jpg", "blue_tile.jpg", "green_tile.jpg"]
    for name in samples:
        path = os.path.join(IMAGES_DIR, name)
        print(f"\n{name}")
        for label, confidence in classifier.classify(path, top_k=3):
            print(f"  {confidence:.3f}  {label}")

    classifier.save(WEIGHTS_PATH)
    print(f"\nSaved weights to {WEIGHTS_PATH}")


def run_indexer_demo(classifier: ImageClassifier) -> None:
    """Build the indexer over the demo catalogue and print recommendations."""
    print("\n=== Image Retrieval ===")
    items, relative_paths = ImageIndexer.from_csv(
        CATALOGUE_PATH, "item", "image"
    )
    image_paths = [os.path.join(DATA_DIR, path) for path in relative_paths]

    indexer = ImageIndexer(classifier=classifier)
    indexer.fit(items, image_paths)

    seed = items[0]
    print(f"Items visually similar to '{seed}':")
    for item, score in indexer.recommend(seed, top_k=3):
        print(f"  {score:.3f}  {item}")

    query_path = os.path.join(IMAGES_DIR, "query.jpg")
    print(f"\nRecommendations for query image 'query.jpg':")
    for item, score in indexer.recommend_from_image(query_path, top_k=3):
        print(f"  {score:.3f}  {item}")


def main() -> None:
    """Run both demos end to end."""
    ensure_sample_images()
    classifier = ImageClassifier()
    run_classifier_demo(classifier)
    run_indexer_demo(classifier)


if __name__ == "__main__":
    main()
