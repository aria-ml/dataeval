from __future__ import annotations

__all__ = []

from typing import Any, Callable, Sequence, TypeVar, Union, cast

import numpy as np

from dataeval.typing import Array, ObjectDetectionTarget, SegmentationTarget
from dataeval.utils._array import as_numpy
from dataeval.utils.data._selection import Select, Selection, SelectionStage, _TargetWrapper

TDatum = TypeVar("TDatum")

TargetTransform = Callable[[Union[ObjectDetectionTarget , SegmentationTarget]], _TargetWrapper]
MetadataTransform = Callable[[dict[str, Any], Any], dict[str, Any]]

# Transform functions moved to module level
def create_target_transform(
        filtered_indices: list[int]) -> TargetTransform:
    """Create a function that transforms detection targets by filtering to specified indices."""
    def transform_target(target: ObjectDetectionTarget | SegmentationTarget) -> _TargetWrapper:
        wrapper = _TargetWrapper()
        
        if isinstance(target, ObjectDetectionTarget):
            if isinstance(target.boxes , Array):
                typed_boxes = cast(Array, target.boxes) 
                wrapper.boxes = typed_boxes[filtered_indices]
            else: 
                wrapper.boxes = [target.boxes[i] for i in filtered_indices]

        elif isinstance(target, SegmentationTarget):
            if isinstance(target.mask, Array):
                typed_mask = cast(Array, target.mask)
                wrapper.mask = typed_mask[filtered_indices]
            else:
                wrapper.mask = [target.mask[i] for i in filtered_indices]

        if hasattr(target, 'labels'):
            if isinstance(target.labels, Array):
                typed_labels = cast(Array, target.labels)
                wrapper.labels = typed_labels[filtered_indices]
            else:
                wrapper.labels = [target.labels[i] for i in filtered_indices]

        if hasattr(target, 'scores') and target.scores is not None:
            if isinstance(target.scores, Array):
                typed_scores = cast(Array, target.scores)
                wrapper.scores = typed_scores[filtered_indices]
            else:
                wrapper.scores = [target.scores[i] for i in filtered_indices]
        
        return wrapper
    
    return transform_target


def create_metadata_transform(filtered_indices: list[int], 
                             original_target: Any) -> MetadataTransform:
    """Create a function that transforms metadata by filtering array items to specified indices. We 
    need to do this when we have metadata for multiple detections per image."""

    def transform_metadata(metadata: dict[str, Any], transformed_target: Any) -> dict[str, Any]:
        
        result_metadata: dict[str, Any] = {}
        
        for key, value in metadata.items():
            if isinstance(value, (list, tuple, np.ndarray)):
                # Check if this matches the number of detections
                if hasattr(original_target, 'labels') and len(value) == len(original_target.labels):
                    # This is per-detection metadata, filter it
                    result_metadata[key] = [value[i] for i in filtered_indices]
                else:
                    # This is some other array metadata, keep as is
                    result_metadata[key] = value
            else:
                # This is per-image metadata, keep as is
                result_metadata[key] = value
        
        return result_metadata
    
    return transform_metadata


def calculate_balance_score(
    idx: int,
    class_indices: dict[int, list[int]],
    current_counts: dict[int, int],
    selected_indices: set[int],
    target_classes: list[int]
) -> float:
    """
    Calculate how good an image is for class balance.
    
    Args:
        idx: Index of the image to evaluate
        class_indices: Dictionary mapping classes to image indices
        current_counts: Current count of each class
        selected_indices: Already selected indices
        target_classes: List of classes to balance
        
    Returns:
        Score indicating how good this image is for balance (lower is better)
    """
    # Lower score is better (we want to minimize imbalance)
    if idx in selected_indices:
        return float('inf')  # Already selected
    
    # Calculate how this image affects balance
    image_classes = [cls for cls in target_classes if idx in class_indices.get(cls, [])]
    
    if not image_classes:
        return float('inf')  # No relevant classes
    
    # Find the most underrepresented class
    min_class = min(current_counts.keys(), key=lambda k: current_counts[k])
    min_count = current_counts[min_class]
    
    # Prioritize images with the most under-represented class
    help_to_min = 1 if min_class in image_classes else 0
    
    # Penalize adding to already well-represented classes
    harm_to_balance = sum(
        1 for cls in image_classes if current_counts[cls] > min_count
    )
    
    # Balance score: prioritize helping min class, penalize increasing imbalance
    if help_to_min > 0:
        return -help_to_min + 0.1 * harm_to_balance
    else:
        return harm_to_balance


class ClassFilter(Selection[TDatum]):
    """
    Filter and optionally balance the dataset by class.

    Parameters
    ----------
    classes : Sequence[int] or None, default None
        The classes to filter by. If None, all classes are included.
    balance : bool, default False
        Whether to balance the classes.
    filter_detections : bool, default True
        If True, remove detections that are not in the set of specified classes.
        If False, keep all detections for images that have any detection of a specified class.
        Only applies to ObjectDetectionTargets and SegmentationTargets.

    Note
    ----
    If `balance` is True, the total number of instances of each class will
    be equalized. This may result in a lower total number of instances.
    
    For ObjectDetectionTargets and SegmentationTargets, an image is included if at least 
    one detection/mask has a class in the specified classes. If filter_detections is True,
    detections/masks of other classes will be removed from the target.
    """

    stage = SelectionStage.FILTER

    def __init__(
        self, 
        classes: Sequence[int] | None = None, 
        balance: bool = False,
        filter_detections: bool = True
    ) -> None:
        self.classes = classes
        self.balance = balance
        self.filter_detections = filter_detections

    def _apply_advanced_balancing(
        self, 
        dataset: Select[TDatum], 
        class_indices: dict[int, list[int]]
    ) -> list[int]:
        """
        Apply advanced class balancing using a greedy approach.
        
        Args:
            dataset: The dataset to balance
            class_indices: Dictionary mapping classes to image indices
            
        Returns:
            List of indices after balancing
        """
        if not class_indices:
            return []
            
        # Determine target classes
        target_classes = list(class_indices.keys())
        
        # Determine minimum count per class (will aim for this many of each class)
        min_count = min(len(indices) for indices in class_indices.values())
        
        # Initialize selection
        selected_indices = set()
        current_counts = dict.fromkeys(target_classes, 0)
        
        # Greedy selection algorithm
        all_indices = list(set().union(*class_indices.values()))
        
        while min(current_counts.values()) < min_count:
            # Check if we've reached the dataset size limit
            if len(selected_indices) >= dataset._size_limit:
                break

            # Find image that most improves balance
            best_idx = min(
                all_indices,
                key=lambda idx: calculate_balance_score(
                    idx, 
                    class_indices, 
                    current_counts,
                    selected_indices,
                    target_classes
                )
            )
            
            # Check if best score is infinity (can't improve further)
            score = calculate_balance_score(
                best_idx, 
                class_indices, 
                current_counts,
                selected_indices,
                target_classes
            )
            
            if score == float('inf'):
                break
                
            # Add this index and update counts
            selected_indices.add(best_idx)
            for cls in target_classes:
                if best_idx in class_indices.get(cls, []):
                    current_counts[cls] += 1
        
        return sorted(selected_indices)

    def __call__(self, dataset: Select[TDatum]) -> None:
        if self.classes is None and not self.balance:
            return

        class_indices: dict[int, list[int]] = {} if self.classes is None else {k: [] for k in self.classes}      
        for i, idx in enumerate(dataset._selection):
            # Get the item but don't modify it directly
            _, target, _ = cast(tuple[Any, Any, dict[str, Any]], dataset._dataset[idx])
           
            
            if isinstance(target, Array):
                # Handle classification targets
                label = int(np.argmax(as_numpy(target)))
                if not self.classes or label in self.classes:
                    class_indices.setdefault(label, []).append(i)
            
            elif isinstance(target, (ObjectDetectionTarget, SegmentationTarget)):
                # Handle both object detection and segmentation targets
                target_classes: set[int] = set()
                filtered_indices: list[int] = []
                
                # Get all labels and scores
                labels = as_numpy(target.labels)
                scores = as_numpy(target.scores) if hasattr(target, 'scores') and target.scores is not None else None
                
                # Process each item (detection or segment)
                for idx_item in range(len(labels)):
                    label = int(labels[idx_item])
                    
                    # Apply argmax if needed for multi-class scores
                    if scores is not None and scores.ndim > 1:
                        label = int(np.argmax(scores[idx_item]))
                    
                    # Check if this item should be kept
                    if not self.classes or label in self.classes:
                        target_classes.add(label)
                        filtered_indices.append(idx_item)
                
                # Keep image if it has any item with classes we want
                if target_classes:
                    # Add this image to each class's list
                    for cls in target_classes:
                        if not self.classes or cls in self.classes:
                            class_indices.setdefault(cls, []).append(i)
                    
                    # Define transformation functions for Select to apply during __getitem__
                    if self.filter_detections and filtered_indices and len(filtered_indices) < len(labels):
                        # Create and store transform functions
                        dataset._target_transforms[idx] = create_target_transform(filtered_indices)
                        dataset._metadata_transforms[idx] = create_metadata_transform(filtered_indices, target)
            
            else:
                raise TypeError("ClassFilter only supports classification targets as an array of confidence scores, "
                            "ObjectDetectionTargets, or SegmentationTargets.")
        
        # Apply balancing if requested
        if self.balance and class_indices:
            # Use the advanced balancing algorithm
            subselection = self._apply_advanced_balancing(dataset, class_indices)
        else:
            # Traditional approach - just take all images with classes we want
            # For each class, take all available examples up to dataset size limit
            per_class_limit = dataset._size_limit
            subselection = [i for v in class_indices.values() for i in v[:per_class_limit]]
            # Remove duplicates (when an image contains multiple classes)
            subselection = np.unique(subselection).tolist()
            
        # Update the dataset selection
        dataset._selection = [dataset._selection[i] for i in subselection]