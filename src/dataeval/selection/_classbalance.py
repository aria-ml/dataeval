__all__ = []

from collections.abc import Iterable, Iterator, Mapping, Sequence
from itertools import tee
from typing import Any, Literal

import numpy as np

from dataeval.config import get_seed
from dataeval.core._label_stats import label_stats
from dataeval.protocols import AnnotatedDataset, Array, ObjectDetectionTarget, SegmentationTarget
from dataeval.selection._select import Select, Selection, SelectionStage
from dataeval.utils.arrays import as_numpy


class ClassBalance(Selection[Any]):
    """
    Select a balanced subset of images based on class distribution.

    This selection strategy balances class representation in datasets for classification,
    object detection, and segmentation tasks. It supports two balancing methods: global
    (weighted sampling by inverse class frequency) and interclass (equal samples per class).

    Parameters
    ----------
    method : {'global', 'interclass'}
        Balancing strategy to use:
        - 'global': Sample images with probability proportional to inverse square root
        of class frequencies, giving higher weight to rare classes
        - 'interclass': Sample equal number of images from each class
    num_samples : int or None, optional
        Total number of samples to select. If None, returns dataset size worth of samples.
    background_class : int or str or None, optional
        Class label to treat as background. For 'global' method, background gets frequency 1.0.
        For 'interclass' method, background is excluded from sampling.
    num_empty : int or float or None, optional
        Number of empty images (no targets) to include. If float, treated as proportion
        of dataset size. If None, no special handling for empty images.
    aggregation_func : {'mean', 'max'}, default='mean'
        How to aggregate repeat factors when image contains multiple classes.
        Only used in 'global' method.
    oversample_factor : float, default=1.0
        Scaling factor for class repeat factors in 'global' method. Higher values
        increase oversampling of rare classes.
    minimize_duplicates : bool, default=False
        If True, use probability scoring to reduce duplicate selections in 'interclass'
        method when sampling with replacement.

    Notes
    -----
    - Empty images (those with no detection/segmentation targets) are tracked separately from class labels
    - The selection may contain duplicate indices depending on method and parameters
    - Uses numpy random number generator seeded from dataeval config
    """

    stage = SelectionStage.FILTER

    def __init__(
        self,
        method: Literal["global", "interclass"],
        num_samples: int | None = None,
        background_class: int | str | None = None,
        num_empty: int | float | None = None,
        aggregation_func: Literal["mean", "max"] = "mean",
        oversample_factor: float = 1.0,
        minimize_duplicates: bool = False,
    ) -> None:
        self.method = method
        self.num_samples = num_samples
        self.background_class = background_class
        self.num_empty = num_empty
        self.aggregation_func = aggregation_func
        self.oversample_factor = oversample_factor
        self.minimize_duplicates = minimize_duplicates
        self._rng = np.random.default_rng(get_seed())
        self._images_per_class: Mapping[int, Sequence[int]]
        self._classes: Sequence[int]

    def _yield_labels(self, dataset: Select[Any]) -> Iterator[tuple[int, int]]:
        """
        Yield (label, image_index) pairs from dataset targets.

        Parameters
        ----------
        dataset : Select[Any]
            Dataset to analyze, containing (input, target) pairs.

        Yields
        ------
        tuple[int, int]
            Pairs of (class_label, image_index) for each label in the dataset.
        """
        for img_idx, datum in enumerate(dataset):
            target = datum[1] if isinstance(datum, tuple) else None
            if isinstance(target, Array):
                if len(target) > 0:
                    yield (int(np.argmax(as_numpy(target))), img_idx)
            elif isinstance(target, ObjectDetectionTarget | SegmentationTarget):
                labels_raw = target.labels if isinstance(target.labels, Iterable) else [target.labels]
                for lbl in labels_raw:
                    yield (int(as_numpy(lbl)), img_idx)

    def _compute_label_stats(
        self, dataset: Select[Any]
    ) -> tuple[dict[int, list[int]], dict[int, list[int]], dict[int, float], list[int]]:
        """
        Compute label statistics for the dataset using core label_stats.

        Analyzes the dataset to extract relationships between images and their class labels,
        supporting classification, object detection, and segmentation tasks.

        Parameters
        ----------
        dataset : Select[Any]
            Dataset to analyze, containing (input, target) pairs.

        Returns
        -------
        images_per_class : dict[int, list[int]]
            Mapping from class label to list of image indices containing that class.
        classes_per_image : dict[int, list[int]]
            Mapping from image index to list of class labels in that image.
        class_freqs : dict[int, float]
            Mapping from class label to its frequency (proportion of images containing it).
        empty_image_indices : list[int]
            List of indices for images with no targets.

        Notes
        -----
        For classification tasks, each image has exactly one label.
        For detection/segmentation, images may have multiple labels or be empty.
        Empty images are tracked separately, not as a class.
        """
        # Get index2label mapping if available
        index2label: dict[int, str] | None = (
            dataset.metadata.get("index2label", None) if isinstance(dataset, AnnotatedDataset) else None
        )

        def _unzip(source: Iterator[tuple[int, int]]) -> tuple[Iterator[int], Iterator[int]]:
            it1, it2 = tee(source)
            return (x for x, _ in it1), (y for _, y in it2)

        # Unzip flat labels and item indices from iterator
        class_labels, item_indices = _unzip(self._yield_labels(dataset))

        # Use core label_stats function
        stats = label_stats(class_labels, item_indices, index2label, image_count=self._num_images)

        # Convert image_indices_per_class to the format we need
        images_per_class = {k: list(v) for k, v in stats["image_indices_per_class"].items()}

        # Get classes_per_image directly from stats - convert to dict for backwards compatibility
        classes_per_image = {i: list(classes) for i, classes in enumerate(stats["classes_per_image"]) if classes}

        # Compute class frequencies from image counts
        class_freqs = {k: v / self._num_images for k, v in stats["image_counts_per_class"].items()}

        # Get empty image indices
        empty_image_indices = list(stats["empty_image_indices"])

        return images_per_class, classes_per_image, class_freqs, empty_image_indices

    def _get_empty_images(self) -> list[int]:
        """
        Sample indices of images with no target annotations.

        Samples the specified number of empty images (those with no detection/segmentation
        targets). Uses sampling without replacement if enough empty images exist, otherwise
        samples with replacement.

        Returns
        -------
        list[int]
            List of image indices for empty images, of length self._empty.

        Notes
        -----
        Empty images are tracked separately in self._empty_image_indices.
        """
        empty_indices = self._empty_image_indices
        empty_count = len(empty_indices)
        if empty_count >= self._empty:
            # Sample empty images without repeat
            sampled_positions = self._rng.permutation(empty_count)[: self._empty]
            empty_imgs = [empty_indices[i] for i in sampled_positions]
        else:
            # Sample empty images with repeat
            sampled_positions = self._rng.integers(empty_count, size=self._empty)
            empty_imgs = [empty_indices[i] for i in sampled_positions]
        return empty_imgs

    def _global_balance(self) -> list[int]:
        """
        Perform global class balancing using repeat factors.

        Samples images with probability proportional to the inverse square root of class
        frequencies, giving higher weight to images containing rare classes. Each image's
        weight is computed by aggregating (mean or max) the repeat factors of all classes
        it contains.

        Returns
        -------
        list[int]
            List of sampled image indices, possibly with duplicates. Length equals the
            number of non-empty images to sample (num_samples - _empty or _num_images - _empty).

        Notes
        -----
        - Background class (if specified) is treated as having frequency 1.0
        - Empty images (if any) are excluded and handled separately
        - Samples with replacement based on computed probabilities
        """
        if self.background_class in self._cls_frq:
            self._cls_frq[self.background_class] = 1.0
        class_repeat_factor = {
            k: max(1.0, (self.oversample_factor / self._cls_frq[k]) ** (1 / 2)) for k in self._cls_frq
        }
        img_prob = []
        empty_image_set = set(self._empty_image_indices)
        if self._empty > 0 and len(self._empty_image_indices) > 0:
            imgs_to_get = (
                self.num_samples - self._empty if self.num_samples is not None else self._num_images - self._empty
            )
            for i in range(self._num_images):
                img_prob.append(
                    0
                    if i in empty_image_set
                    else (
                        np.mean([class_repeat_factor[lbl] for lbl in self._cls_per_img[i]])
                        if self.aggregation_func == "mean"
                        else np.max([class_repeat_factor[lbl] for lbl in self._cls_per_img[i]])
                    )
                )
        else:
            imgs_to_get = self._num_images if self.num_samples is not None else self._num_images
            img_prob = (
                [np.mean([class_repeat_factor[lbl] for lbl in self._cls_per_img[i]]) for i in range(self._num_images)]
                if self.aggregation_func == "mean"
                else [
                    np.max([class_repeat_factor[lbl] for lbl in self._cls_per_img[i]]) for i in range(self._num_images)
                ]
            )

        # Normalize probabilities to sum to 1.0 - use uniform distribution if all are 0
        total_prob = sum(img_prob)
        img_prob = [p / total_prob for p in img_prob] if total_prob > 0 else [1.0 / self._num_images] * self._num_images

        return self._rng.choice(self._num_images, size=imgs_to_get, replace=True, p=img_prob).tolist()

    def _calculate_selection_probability(
        self, imgs: Sequence[int], cls: int, current_list: Sequence[int]
    ) -> list[float]:
        """
        Calculate normalized selection probabilities for images to minimize duplicates.

        Computes a score for each candidate image that penalizes images already selected
        multiple times and images containing labels other than the target class.

        Parameters
        ----------
        imgs : list[int]
            List of candidate image indices to score.
        cls : int
            Target class label being sampled.
        current_list : list[int]
            List of image indices already selected.

        Returns
        -------
        list[float]
            List of normalized probability scores, one per image in imgs, summing to 1.0.
            Higher scores indicate better candidates (fewer repeats, fewer wrong labels).

        Notes
        -----
        Score formula: 1 / (1 + num_repeats + 2 * wrong_labels)
        Wrong labels are weighted twice as heavily as repeats.
        Scores are normalized to sum to 1.0 for use with np.random.choice.
        """
        scores = []
        for idx in imgs:
            num_repeats = np.nonzero(as_numpy(current_list) == idx)[0].sum()
            wrong_labels = np.nonzero(as_numpy(self._cls_per_img[idx]) != cls)[0].sum()
            score = 1 / (1 + num_repeats + 2 * wrong_labels)
            scores.append(score)

        # Normalize scores to sum to 1.0 - use uniform distribution if all are 0
        total = sum(scores)
        return [s / total for s in scores] if total > 0 else [1.0 / len(imgs)] * len(imgs)

    def _inter_balance(self) -> list[int]:
        """
        Perform interclass balancing by sampling equally from each class.

        Distributes the target number of samples evenly across all non-background classes,
        with any remainder distributed randomly. For each class, samples images containing
        that class with or without replacement as needed.

        Returns
        -------
        list[int]
            List of sampled image indices, possibly with duplicates. Length equals the
            number of non-empty images to sample (num_samples - _empty or _num_images - _empty).

        Notes
        -----
        - Background class (if specified) is excluded from sampling
        - Empty images (if any) are excluded and handled separately
        - If minimize_duplicates is True, uses probability scoring to reduce duplicate selections
        - Samples without replacement when possible; uses replacement only when class has
        fewer images than required
        """
        if self._empty > 0 and len(self._empty_image_indices) > 0:
            class_keys = [k for k in self._images_per_class if k != self.background_class]
            imgs_to_get = (
                self.num_samples - self._empty if self.num_samples is not None else self._num_images - self._empty
            )
        else:
            # Always filter out background_class from balancing
            class_keys = [k for k in self._images_per_class if k != self.background_class]
            imgs_to_get = self.num_samples if self.num_samples is not None else self._num_images

        samples = []
        n_class = len(class_keys)

        # Handle edge case where there are no classes to balance
        if n_class == 0:
            return samples

        samples_per_class, leftover = divmod(imgs_to_get, n_class)
        gets_extra_sample = self._rng.permutation(class_keys)[:leftover].tolist()
        for i, cls in enumerate(class_keys):
            class_imgs = self._images_per_class[cls]
            n_imgs = samples_per_class
            n_imgs += 1 if i in gets_extra_sample else 0
            replace = len(class_imgs) < n_imgs
            if replace and self.minimize_duplicates:
                samples.extend(class_imgs)
                probs = self._calculate_selection_probability(class_imgs, cls, samples)
                num_replace = n_imgs - len(class_imgs)
                additional = self._rng.choice(class_imgs, size=num_replace, replace=True, p=probs)
                samples.extend(additional)
            else:
                probs = self._calculate_selection_probability(class_imgs, cls, samples)
                additional = self._rng.choice(class_imgs, size=n_imgs, replace=replace, p=probs, shuffle=False)
                samples.extend(additional)

        return samples

    def __call__(self, dataset: Select[Any]) -> None:
        selection = []
        self._num_images = len(dataset)
        self._empty = (
            int(self._num_images * self.num_empty)
            if isinstance(self.num_empty, float)
            else (0 if self.num_empty is None else self.num_empty)
        )
        (
            self._images_per_class,
            self._cls_per_img,
            self._cls_frq,
            self._empty_image_indices,
        ) = self._compute_label_stats(dataset)
        self._classes = list(self._images_per_class.keys())
        if self._empty is not None and len(self._empty_image_indices) > 0:
            selection.extend(self._get_empty_images())
        selection.extend(self._global_balance() if self.method == "global" else self._inter_balance())
        selection.sort()
        dataset._selection = selection
