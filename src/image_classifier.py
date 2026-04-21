"""Image classification using a pre-trained MobileNetV2 backbone."""

import os
from typing import Iterable, List, Tuple, Union

import torch
from PIL import Image
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

ImageInput = Union[str, Image.Image]


class ImageClassifier:
    """Classify images with a pre-trained MobileNetV2 network.

    The class wraps a torchvision MobileNetV2 model and exposes a small,
    focused API: classify, classify_batch, extract_features, and
    export_to_coreml. Predictions are returned as ``(label, confidence)``
    tuples sorted from most to least likely, which mirrors the output
    shape of an equivalent Core ML classifier on-device.
    """

    def __init__(self, weights_preset: str = "IMAGENET1K_V2") -> None:
        """Initialize the classifier with pre-trained MobileNetV2 weights.

        Args:
            weights_preset: Name of a ``torchvision.models.MobileNet_V2_Weights``
                enum member. The default is the second-generation ImageNet
                checkpoint, which Apple also uses as the baseline MobileNetV2
                in their Core ML model zoo.
        """
        self._weights = MobileNet_V2_Weights[weights_preset]
        self._model = mobilenet_v2(weights=self._weights)
        self._model.eval()
        self._preprocess = self._weights.transforms()
        self._labels: List[str] = list(self._weights.meta["categories"])

    @property
    def labels(self) -> List[str]:
        """Return the ordered list of class labels."""
        return list(self._labels)

    def classify(
        self,
        image: ImageInput,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Return top-k predictions for a single image.

        Args:
            image: PIL image or path to an image file.
            top_k: Number of predictions to return.

        Returns:
            List of (label, confidence) tuples sorted from highest to
            lowest confidence.
        """
        return self.classify_batch([image], top_k=top_k)[0]

    def classify_batch(
        self,
        images: Iterable[ImageInput],
        top_k: int = 5,
    ) -> List[List[Tuple[str, float]]]:
        """Return top-k predictions for a batch of images.

        Args:
            images: Iterable of PIL images or paths to image files.
            top_k: Number of predictions to return per image.

        Returns:
            List aligned with ``images`` where each element is a list of
            (label, confidence) tuples.
        """
        tensor = self._preprocess_batch(list(images))
        with torch.no_grad():
            logits = self._model(tensor)
            probabilities = torch.softmax(logits, dim=1)
        top_values, top_indices = torch.topk(probabilities, k=top_k, dim=1)
        return [
            [
                (self._labels[int(idx)], float(value))
                for idx, value in zip(indices, values)
            ]
            for indices, values in zip(top_indices, top_values)
        ]

    def extract_features(self, images: Iterable[ImageInput]) -> torch.Tensor:
        """Return pooled feature embeddings from the MobileNetV2 backbone.

        The returned vectors sit just before the classifier head, which
        makes them a convenient embedding for similarity search. The
        ``ImageIndexer`` class consumes this method directly.

        Args:
            images: Iterable of PIL images or paths to image files.

        Returns:
            Tensor of shape ``(num_images, 1280)`` containing the output
            of the feature block after global average pooling.
        """
        tensor = self._preprocess_batch(list(images))
        with torch.no_grad():
            features = self._model.features(tensor)
            pooled = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        return torch.flatten(pooled, start_dim=1)

    def export_to_coreml(
        self,
        path: str,
        image_size: Tuple[int, int] = (224, 224),
    ) -> None:
        """Convert the classifier to a Core ML model bundle.

        Preprocessing (ImageNet mean/std normalisation over 0-255 pixel
        values) is baked into the exported model so Swift callers can pass
        a ``CVPixelBuffer`` straight through without reimplementing the
        transform.

        Args:
            path: Destination path for the exported ``.mlmodel`` /
                ``.mlpackage`` bundle.
            image_size: ``(height, width)`` for the fixed input size the
                Core ML model will accept.
        """
        import coremltools as ct

        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        example = torch.randn(1, 3, *image_size)
        traced = torch.jit.trace(self._model, example)

        mean = self._preprocess.mean
        std = self._preprocess.std
        image_input = ct.ImageType(
            name="image",
            shape=(1, 3, *image_size),
            scale=1.0 / 255.0,
            bias=[0.0, 0.0, 0.0],
            color_layout=ct.colorlayout.RGB,
        )
        coreml_model = ct.convert(
            traced,
            inputs=[image_input],
            classifier_config=ct.ClassifierConfig(self._labels),
            convert_to="mlprogram",
        )
        coreml_model.user_defined_metadata["preprocessing.mean"] = str(list(mean))
        coreml_model.user_defined_metadata["preprocessing.std"] = str(list(std))
        coreml_model.save(path)

    def save(self, path: str) -> None:
        """Persist the classifier weights to disk.

        Args:
            path: Destination file path (typically ending in ``.pt``).
        """
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        torch.save(self._model.state_dict(), path)

    def load(self, path: str) -> None:
        """Restore classifier weights from a previously saved checkpoint.

        Args:
            path: Source file path for the saved state dict.
        """
        state = torch.load(path, map_location="cpu")
        self._model.load_state_dict(state)
        self._model.eval()

    def _preprocess_batch(self, images: List[ImageInput]) -> torch.Tensor:
        """Load, normalise, and stack a batch of images into a tensor."""
        if not images:
            raise ValueError("At least one image is required.")
        tensors = [self._preprocess(self._load(image)) for image in images]
        return torch.stack(tensors)

    @staticmethod
    def _load(image: ImageInput) -> Image.Image:
        """Return a RGB PIL image for either a path or an existing image."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.open(image).convert("RGB")
