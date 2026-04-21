"""Content-based image retrieval using MobileNetV2 feature embeddings."""

from typing import Iterable, List, Optional, Tuple, Union

import pandas as pd
import torch
from PIL import Image

from src.image_classifier import ImageClassifier

ImageInput = Union[str, Image.Image]


class ImageIndexer:
    """Rank catalogue images by visual similarity to a query.

    The indexer extracts MobileNetV2 feature embeddings for every
    catalogue image and scores candidates against a query using cosine
    similarity. It is intentionally simple: a readable baseline for
    visual search, not a production retrieval stack.
    """

    def __init__(self, classifier: Optional[ImageClassifier] = None) -> None:
        """Initialize the indexer.

        Args:
            classifier: An ``ImageClassifier`` whose backbone is reused
                for feature extraction. If omitted, a fresh classifier
                is loaded — pass an existing instance to share weights
                when both components run in the same process.
        """
        self._classifier = classifier if classifier is not None else ImageClassifier()
        self._items: List[str] = []
        self._image_paths: List[str] = []
        self._features: Optional[torch.Tensor] = None

    def fit(
        self,
        items: Iterable[str],
        image_paths: Iterable[str],
    ) -> None:
        """Index a catalogue of images by their feature embeddings.

        Args:
            items: Unique item identifiers (for example, SKU names).
            image_paths: Paths to the image files, aligned with ``items``.
        """
        self._items = list(items)
        self._image_paths = list(image_paths)

        if len(self._items) != len(self._image_paths):
            raise ValueError("items and image_paths must have matching length.")
        if not self._items:
            raise ValueError("At least one item is required to fit the indexer.")

        features = self._classifier.extract_features(self._image_paths)
        self._features = torch.nn.functional.normalize(features, dim=1)

    def recommend(self, item: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Return the most visually similar items to a catalogue entry.

        Args:
            item: Identifier of the seed item (must have been passed to
                ``fit``).
            top_k: Number of recommendations to return.

        Returns:
            List of (item, similarity) tuples sorted from most to least
            similar, excluding the seed item itself.
        """
        self._require_fitted()
        if item not in self._items:
            raise KeyError(f"Item '{item}' is not in the catalogue.")

        index = self._items.index(item)
        similarities = torch.mv(self._features, self._features[index])
        return self._rank(similarities.tolist(), exclude_index=index, top_k=top_k)

    def recommend_from_image(
        self,
        image: ImageInput,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Return items most similar to an ad-hoc query image.

        Useful for cold-start visual search where the user uploads a
        reference image rather than selecting an existing catalogue entry.

        Args:
            image: PIL image or path to the query image.
            top_k: Number of recommendations to return.

        Returns:
            List of (item, similarity) tuples sorted by similarity.
        """
        self._require_fitted()
        features = self._classifier.extract_features([image])
        query = torch.nn.functional.normalize(features, dim=1).squeeze(0)
        similarities = torch.mv(self._features, query)
        return self._rank(similarities.tolist(), exclude_index=None, top_k=top_k)

    @classmethod
    def from_csv(
        cls,
        path: str,
        item_column: str,
        image_column: str,
    ) -> Tuple[List[str], List[str]]:
        """Load a catalogue from a CSV file.

        Args:
            path: Path to the CSV file.
            item_column: Column holding item identifiers.
            image_column: Column holding paths to image files.

        Returns:
            Tuple of (items, image_paths) lists.
        """
        frame = pd.read_csv(path)
        return (
            frame[item_column].astype(str).tolist(),
            frame[image_column].astype(str).tolist(),
        )

    def _rank(
        self,
        similarities,
        exclude_index,
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """Sort similarity scores and return the top-k items."""
        scored = [
            (self._items[i], float(score))
            for i, score in enumerate(similarities)
            if i != exclude_index
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def _require_fitted(self) -> None:
        """Raise a clear error if the indexer has not been fitted yet."""
        if self._features is None:
            raise RuntimeError("ImageIndexer must be fitted before use.")
