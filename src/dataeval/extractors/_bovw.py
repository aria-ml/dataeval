"""
Bag of Visual Words feature extractor for rotation-invariant image embeddings.
"""

__all__ = []

from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import MiniBatchKMeans

from dataeval.config import get_seed
from dataeval.protocols import Array
from dataeval.utils.arrays import as_numpy
from dataeval.utils.preprocessing import rescale, to_canonical_grayscale


class BoVWExtractor:
    """
    Bag of Visual Words (BoVW) feature extractor for images.

    This class implements the :class:`~dataeval.protocols.FeatureExtractor` protocol
    for use with drift detectors and duplicate detection. It extracts SIFT keypoints
    from images and quantizes them into a visual vocabulary, producing rotation and
    scale invariant histogram embeddings.

    The BoVW approach works by:

    1. Extracting local SIFT descriptors from each image
    2. Clustering all descriptors to form a "visual vocabulary" (codebook)
    3. Representing each image as a histogram of visual word occurrences

    This produces embeddings that are invariant to image rotation, scale changes,
    and minor viewpoint variations, making it effective for finding near-duplicate
    images even when they have been transformed.

    Parameters
    ----------
    vocab_size : int, default 2048
        Number of visual words (clusters) in the vocabulary. Larger vocabularies
        capture finer visual distinctions but require more training data and memory.
        Common values range from 256 to 4096. The actual vocabulary size may be
        smaller if the training data contains fewer total SIFT descriptors than
        the requested size.

    Attributes
    ----------
    vocab_size : int
        The configured vocabulary size.
    kmeans : MiniBatchKMeans or None
        The fitted k-means clustering model. None before :meth:`fit` is called.
    sift : cv2.SIFT
        The SIFT feature detector/descriptor extractor.

    Example
    -------
    Basic usage with fit/transform pattern

    >>> import numpy as np
    >>> from dataeval.extractors import BoVWExtractor
    >>>
    >>> # Create sample images (C, H, W format)
    >>> rng = np.random.default_rng(42)
    >>> images = [rng.integers(0, 256, (3, 64, 64), dtype=np.uint8) for _ in range(10)]
    >>>
    >>> # Create extractor and fit vocabulary
    >>> extractor = BoVWExtractor(vocab_size=64)
    >>> extractor.fit(images)
    >>>
    >>> # Transform images to embeddings
    >>> embeddings = extractor.transform(images)
    >>> embeddings.shape
    (10, 64)

    Using with duplicate detection

    >>> from dataeval.quality import Duplicates
    >>>
    >>> # Fit extractor on reference dataset
    >>> extractor = BoVWExtractor(vocab_size=128)
    >>> extractor.fit(reference_data)
    >>>
    >>> # Use embeddings for duplicate detection
    >>> embeddings = extractor.transform(unlabeled_data)

    One-shot fit and transform

    >>> # Convenience method that fits and transforms in one call
    >>> extractor = BoVWExtractor(vocab_size=64)
    >>> embeddings = extractor(images)
    >>> embeddings.shape
    (10, 64)

    Notes
    -----
    **Vocabulary Training**: The vocabulary should be trained on a representative
    sample of images. Once fitted, the same extractor can transform new images
    into comparable embeddings. Calling :meth:`fit` again will replace the
    existing vocabulary.

    **Image Format**: Images should be in (C, H, W) channel-first format, which
    is standard for PyTorch datasets. Both RGB (3 channels) and grayscale
    (1 channel) images are supported. Images are automatically converted to
    uint8 if needed.

    **Empty Features**: Images with no detected SIFT features (e.g., uniform
    color images) will have zero-valued histogram embeddings.

    **Reproducibility**: The k-means clustering uses a random seed from
    DataEval's global configuration via :func:`~dataeval.config.get_seed`.
    Set a seed with :func:`~dataeval.config.set_seed` for reproducible results.

    See Also
    --------
    dataeval.quality.Duplicates : Duplicate detection using embeddings
    dataeval.extractors.UncertaintyFeatureExtractor : Uncertainty-based feature extraction
    """

    def __init__(self, vocab_size: int = 2048) -> None:
        try:
            import cv2
        except ImportError as e:
            raise ImportError(
                "BoVWExtractor requires 'opencv-python' or related package. "
                "Please install it via 'pip install opencv-python' or using the extra `dataeval[opencv]`."
            ) from e

        if vocab_size < 1:
            raise ValueError("vocab_size must be at least 1")
        self.vocab_size = vocab_size
        self._kmeans: MiniBatchKMeans | None = None
        self._sift = cv2.SIFT.create()

    def _preprocess_image(self, img: Any) -> NDArray[np.uint8]:
        """
        Convert an image to grayscale uint8 format for SIFT processing.

        Parameters
        ----------
        img : Array
            Input image in (C, H, W) format. Supports torch tensors and numpy arrays.

        Returns
        -------
        NDArray[np.uint8]
            Grayscale image in (H, W) format as uint8.
        """
        img = as_numpy(img[0] if isinstance(img, tuple) else img)
        if img.dtype != np.uint8:
            img = rescale(img, depth=8)
        return to_canonical_grayscale(img)

    def _extract_descriptors(self, data: Any) -> Iterator[NDArray[np.float32]]:
        """
        Extract SIFT descriptors from images.

        Yields
        ------
        NDArray[np.float32]
            SIFT descriptors for each image. May be empty (shape (0, 128))
            for images with no detected features.
        """
        for img in data:
            img = self._preprocess_image(img)
            _, des = self._sift.detectAndCompute(img, None)
            yield np.zeros((0, 128), dtype=np.float32) if des is None else des.astype(np.float32)

    def fit(self, data: Any) -> None:
        """
        Train the visual vocabulary on the provided images.

        Extracts SIFT descriptors from all images and clusters them using
        MiniBatchKMeans to form the visual vocabulary (codebook).

        Parameters
        ----------
        data : Any
            Iterable of images in (C, H, W) format. Supports RGB (3 channels)
            and grayscale (1 channel) images. Also accepts (image, label) tuples
            as returned by PyTorch datasets.

        Returns
        -------
        BoVWExtractor
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If no SIFT features are found in any image. This typically occurs
            when all images are uniform (e.g., solid color).
        """
        all_descriptors = [des for des in self._extract_descriptors(data) if len(des) > 0]

        if not all_descriptors:
            raise ValueError("No SIFT features found in any image. Cannot build vocabulary.")

        train_data = np.vstack(all_descriptors).astype(np.float64)

        # Ensure vocab_size doesn't exceed the number of descriptors
        n_clusters = min(self.vocab_size, len(train_data))

        # Use MiniBatchKMeans for speed on large datasets
        seed = get_seed()
        self._kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init="auto", random_state=seed)
        self._kmeans.fit(train_data)

    def transform(self, data: Any) -> Array:
        """
        Transform images into BoVW histogram embeddings.

        Uses the fitted vocabulary to convert images into normalized histograms
        of visual word occurrences.

        Parameters
        ----------
        data : Any
            Iterable of images in (C, H, W) format. Supports RGB (3 channels)
            and grayscale (1 channel) images. Also accepts (image, label) tuples
            as returned by PyTorch datasets.

        Returns
        -------
        Array
            Embeddings array of shape (n_images, vocab_size). Each row is an
            L2-normalized histogram of visual word occurrences. Images with no
            detected features have zero-valued histograms.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self._kmeans is None:
            raise RuntimeError("Extractor has not been fitted. Call fit() first.")

        n_clusters: int = int(self._kmeans.n_clusters)  # type: ignore

        embeddings: list[NDArray[np.float32]] = []
        for des in self._extract_descriptors(data):
            histogram = np.zeros(n_clusters, dtype=np.float32)

            if len(des) > 0:
                # Assign each keypoint to the nearest visual word
                words = self._kmeans.predict(des.astype(np.float64))

                # Count occurrences
                unique, counts = np.unique(words, return_counts=True)
                histogram[unique] = counts

                # L2 normalize - crucial for comparison
                norm = np.linalg.norm(histogram)
                if norm > 0:
                    histogram /= norm

            embeddings.append(histogram)

        return np.array(embeddings)

    def __call__(self, data: Any) -> Array:
        """
        Fit vocabulary and transform images in one step.

        Convenience method that combines :meth:`fit` and :meth:`transform`.
        Equivalent to calling ``extractor.fit(data).transform(data)``.

        Parameters
        ----------
        data : Any
            Iterable of images in (C, H, W) format.

        Returns
        -------
        Array
            Embeddings array of shape (n_images, vocab_size).

        Raises
        ------
        ValueError
            If no SIFT features are found in any image.

        Note
        ----
        This method retrains the vocabulary on the input data. If you want to
        use a pre-trained vocabulary, call :meth:`fit` once on training data
        and then use :meth:`transform` for new images.
        """
        self.fit(data)
        return self.transform(data)

    def __repr__(self) -> str:
        """Return string representation of the extractor."""
        fitted = self._kmeans is not None
        return f"{self.__class__.__name__}(vocab_size={self.vocab_size}, fitted={fitted})"
