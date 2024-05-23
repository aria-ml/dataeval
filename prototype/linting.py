from collections import Counter
from typing import Any, Dict, Optional

import numpy as np
import scipy as sp
from PIL import Image, ImageFilter

from daml._internal.metrics.base import EvaluateMixin

# Comments from John regarding implementation
"""
images is constrained to be uniform w/h.  Probably a reasonable assumption for now, but
there are datasets with non-uniform image sizes.

Flattening all images out together makes sense for what's there.
I wonder if there will be value in per-image statistics down the road, too.

May want more than 10 bins on the histogram given typical image sizes

size = height * width : size might be an ambiguous variable name.
Maybe pixel_area or something?

in _evaluate_boxes, does NxPx4 imply the same number of objects in each image?
If so, we might need to rethink how we store them.

on boxes_data = np.concatenate(...(line 97), would expand_dims solve this more cleanly?
Or would it be easier to just catch the case where the ndim of the input data is 2 or 4?
"""


"""
Checklist of things to include:
Basic Stats - (x is implemented)
    x means 
    x medians
    - std devs??
    x percentiles
    x number of nan or 0 values
    x skew
    x kurtosis
    x histogram
    - quantiles histogram??

Cleanvision stuff -
    x brightness
    x aspect ratio
    x entropy? (amount of information in picture)
    x blurriness
    x color space
    x image size
"""

# Main questions/concerns:
# Should this class accept all images, a batch of images or a single image?
# Is importing a new package pillow okay? - has fast c routines for basic image stats
# Need to decide if channels is first or last and do we force a 1 if it is grayscale?
# How do we support datasets that cannot all be stored in ram
# -> how do we know to iterate over sets vs a completely different set altogether?


#########
# All of these functions work on a single image assuming that channels is first
# Assuming image is an ndarray


# Define a function that slices an array to only keep the last 3 dimensions
def _slice_to_3_dimensions(array):
    # Calculate how many dimensions need to be sliced
    num_slices_needed = array.ndim - 3

    # Generate a slicing tuple to keep the last three dimensions
    # and take the first element of the rest
    slice_tuple = (0,) * num_slices_needed + (slice(None),) * 3

    # Apply the slicing tuple to the array
    sliced_array = array[slice_tuple]

    return sliced_array


# Define a function to apply edge detection using PIL libraries
def _use_PIL(image):
    # Convert a numpy array to a PIL Image object
    im = Image.fromarray(np.moveaxis(image, 0, -1))
    # Convert the image to grayscale
    gray = im.convert("L")
    # Apply an edge detection filter
    ed = gray.filter(ImageFilter.FIND_EDGES)
    # Return a numpy array of the edges
    return np.array(ed)


# Define a function to manually perform edge detection using a kernel
def _edge_filter(image):
    # Define offset and kernel for edge detection
    offset = 0.5
    kernel = np.ones((3, 3), np.uint8) * -1
    kernel[1, 1] = 8

    # Create a "grayscale" image by summing the channels and convert to float for processing
    img = np.sum(image, axis=0).astype(np.float32)
    edges = np.zeros_like(img, dtype=np.float32)

    # Apply convolution with the kernel manually
    for y in range(1, img.shape[0] - 1):
        for x in range(1, img.shape[1] - 1):
            region = img[y - 1 : y + 2, x - 1 : x + 2]
            edges[y, x] = np.sum(region * kernel) + offset

    return edges.astype(np.uint8)


# Class to encapsulate image statistics calculations
class ImageStats:
    def __init__(self, image: np.ndarray):
        self.image = image
        # Potentially need to add a check to make sure the image contains values
        # Initialize image processing steps
        self.get_channels()
        self.get_size_and_aspect_ratio()
        self.get_image_range()
        self.get_missing_and_zero()
        self.get_basic_stats_per_band()
        self.get_histogram()
        self.get_brightness()
        self.get_entropy()
        self.get_blurriness()
        # Reset image to free memory
        self.image = np.array([])

    # Determine the number of channels and process dimensions accordingly
    def get_channels(self):
        dim = self.image.ndim
        if dim == 2:
            self.bands = 1
            self.image = np.expand_dims(self.image, axis=0)
        elif dim == 3:
            self.bands = self.image.shape[0]
        elif dim > 3:
            print(
                "Image has more than 3 dimensions. \
                  This is for single images, not batches or videos. \
                  Selecting the first index in the beginning dimensions \
                  for continued processing."
            )
            self.bands = self.image.shape[-3]
            self.image = _slice_to_3_dimensions(self.image)
        else:
            raise ValueError("You provided a 1-D array, not an image.")

    # Extract size and aspect ratio of the image
    def get_size_and_aspect_ratio(self):
        self.height = self.image.shape[-2]
        self.width = self.image.shape[-1]

        self.size = self.height * self.width
        self.aspect_ratio = min(self.width / self.height, self.height / self.width)

    # Determine the range of the image values based on its max and min values
    def get_image_range(self):
        max_val = np.max(self.image)
        min_val = np.min(self.image)

        self.rescale = True
        if min_val < 0:
            self.val_range = (min_val, max_val)
        elif max_val <= 1:
            self.val_range = (0, 1)
            self.rescale = False
        elif max_val < 2**8:
            self.val_range = (0, 2**8 - 1)
        elif max_val < 2**12:
            self.val_range = (0, 2**12 - 1)
        elif max_val < 2**16:
            self.val_range = (0, 2**16 - 1)
        else:
            self.val_range = (0, 2**32 - 1)

    # Calculate the number of missing values and zeros in the image
    def get_missing_and_zero(self):
        self.missing = np.sum(np.isnan(self.image))
        self.zero = self.size - np.count_nonzero(self.image, axis=(1, 2))

    # Calculate basic statistical measures for each image band
    def get_basic_stats_per_band(self):
        self.mean = np.mean(self.image, axis=(1, 2))
        self.var = np.var(self.image, axis=(1, 2))
        self.skew = sp.stats.skew(self.image, axis=(1, 2))
        self.kurtosis = sp.stats.kurtosis(self.image, axis=(1, 2))
        # self.range = np.hstack([
        #     np.min(self.image, axis=(1,2)).T,
        #     np.max(self.image, axis=(1,2)).T
        # ])
        # self.median = np.median(self.image, axis=(1,2))
        # Below code also implements the above range and median
        self.percentiles = np.percentile(
            self.image, q=[0, 25, 50, 75, 100], axis=(1, 2)
        ).T  # this gives back array (bands, # of percentiles)
        if self.bands == 1:
            self.percentiles = self.percentiles[np.newaxis, :]

    # Compute image histogram per channel
    def get_histogram(self):
        self.histogram = np.vstack(
            [
                np.histogram(
                    self.image[i, :, :],
                    bins=256,
                    range=self.val_range,
                )[0]
                for i in range(self.bands)
            ]
        )

    # Calculate the overall brightness of the image
    def get_brightness(self):
        # Parameters to translate RGB to grayscale
        luma = np.array([0.2126, 0.7152, 0.0722])
        # self.rescale - Flag to adjust image values to be [0,1]
        # if 3 channels, treat them as RGB, else take the mean of the channels
        if self.rescale and self.bands == 3:
            self.avg_brightness = np.sum(luma * ((self.mean + self.val_range[0]) / self.val_range[1]) ** 2)
            adj_image = (self.image + self.val_range[0]) / self.val_range[1]
            self.brightness = np.sum(luma[:, np.newaxis, np.newaxis] * adj_image**2) / self.size
        elif self.rescale:
            self.avg_brightness = np.mean((self.mean + self.val_range[0]) / self.val_range[1])
            self.brightness = np.mean(
                np.sum((self.image + self.val_range[0]) / self.val_range[1], axis=(1, 2)) / self.size
            )
        elif self.bands == 3:
            self.avg_brightness = np.repeat(np.sum(luma * self.mean**2), 3)
            self.brightness = np.repeat(np.sum(luma[:, np.newaxis, np.newaxis] * self.image**2) / self.size, 3)
        else:
            self.avg_brightness = np.mean(self.mean)
            self.brightness = np.mean(np.sum(self.image, axis=(1, 2)) / self.size)

    # Compute the overall entropy for the image
    # Way to determine approximate how much information is in the image
    def get_entropy(self):
        # get the average histogram from the per channel histogram
        flat_hist = np.mean(self.histogram, axis=0)
        flat_sum = flat_hist.sum()
        # verify its not an empty histogram
        if flat_sum == 0:
            return 0
        # Translate the histogram into probabilities
        probabilities = flat_hist / flat_sum
        probabilities = probabilities[probabilities > 0]
        # Get the entropy of the image -> could also use sp.entropy(probabilities)
        self.entropy = -np.sum(probabilities * np.log2(probabilities))

    # Assess the blurriness of the image using the edge detection results
    def get_blurriness(self):
        # edges = _use_PIL(self.image) if self.bands == 1 or self.bands == 3 else _edge_filter(self.image)
        edges = _edge_filter(self.image)
        # Using the standard deviation to determine how sharp the edge is
        # Blurry images will have a higher standard deviation
        self.blurry = np.std(edges)


# Pulls in a dataset and then gets the individual image stats
# then runs group stats
# Class to encapsulate dataset statistics calculations
class DatasetStats:
    def __init__(
        self,
        images,
        labels: Optional[np.ndarray] = None,
        boxes: Optional[np.ndarray] = None,
    ) -> None:
        # Initialization of DatasetStats with datasets images, optional labels, and optional bounding boxes.
        self.images = images
        self.labels = labels
        self.boxes = boxes
        self.num_images = self.images.shape[0]

        # Initialize arrays to hold various statistics for each image.
        self.img_height = np.zeros(self.num_images)
        self.img_width = np.zeros(self.num_images)
        self.img_size = np.zeros(self.num_images)
        self.img_aspect_ratio = np.zeros(self.num_images)
        self.img_channels = np.zeros(self.num_images)
        self.img_missing = np.zeros(self.num_images)
        self.img_brightness = np.zeros(self.num_images)
        self.img_entropy = np.zeros(self.num_images)
        self.img_avg_brightness = np.zeros(self.num_images)
        self.img_blurriness = np.zeros(self.num_images)
        self.img_range = {}  # Dictionary to hold unique value ranges across images.
        self.image_stats = []  # List to hold ImageStats objects for each image.

        # Call methods to process images and compute various statistics.
        self.process_images()
        self.process_channel_stats()
        self.dataset_stats()

    # Process each image in the dataset and calculate various statistics.
    def process_images(self):
        for i, image in enumerate(self.images):
            stats = ImageStats(image)
            self.image_stats.append(stats)

            # Extract per-image statistics from the ImageStats object.
            self.img_height[i] = stats.height
            self.img_width[i] = stats.width
            self.img_size[i] = stats.size
            self.img_aspect_ratio[i] = stats.aspect_ratio
            self.img_channels[i] = stats.bands
            if stats.missing:
                self.img_missing[i] = 1  # Record if an image has missing data.
            if stats.val_range not in self.img_range:
                self.img_range[stats.val_range] = 1  # Track unique value ranges.
            else:
                self.img_range[stats.val_range] += 1
            self.img_avg_brightness[i] = stats.avg_brightness
            self.img_brightness[i] = stats.brightness
            self.img_entropy[i] = stats.entropy
            self.img_blurriness[i] = stats.blurry

    # Calculate channel-specific statistics for each image
    def process_channel_stats(self):
        max_channels = int(self.img_channels.max())
        # Initialize arrays to hold channel-specific statistics.
        self.img_zeros = np.empty((self.images.shape[0], max_channels))
        self.img_mean = np.empty((self.images.shape[0], max_channels))
        self.img_var = np.empty((self.images.shape[0], max_channels))
        self.img_skew = np.empty((self.images.shape[0], max_channels))
        self.img_kurtosis = np.empty((self.images.shape[0], max_channels))
        self.img_percentile = np.empty((self.images.shape[0], max_channels, 5))
        self.img_histogram = np.empty((self.images.shape[0], max_channels, 256))

        # Iterate over images and fill the statistics arrays, handling channels properly.
        for i, stat in enumerate(self.image_stats):
            # Fill statistics for present channels and mark absent ones as NaN.
            if self.img_channels[i] < max_channels:
                self.img_zeros[i, : self.img_channels[i]] = stat.zero
                self.img_zeros[i, self.img_channels[i] :] = np.nan
                self.img_mean[i, : self.img_channels[i]] = stat.mean
                self.img_mean[i, self.img_channels[i] :] = np.nan
                self.img_var[i, : self.img_channels[i]] = stat.var
                self.img_var[i, self.img_channels[i] :] = np.nan
                self.img_skew[i, : self.img_channels[i]] = stat.skew
                self.img_skew[i, self.img_channels[i] :] = np.nan
                self.img_kurtosis[i, : self.img_channels[i]] = stat.kurtosis
                self.img_kurtosis[i, self.img_channels[i] :] = np.nan
                self.img_percentile[i, : self.img_channels[i], :] = stat.percentiles
                self.img_percentile[i, self.img_channels[i] :, :] = np.nan
                self.img_histogram[i, : self.img_channels[i], :] = stat.histogram
                self.img_histogram[i, self.img_channels[i] :, :] = np.nan
            else:
                self.img_zeros[i, :] = stat.zero
                self.img_mean[i, :] = stat.mean
                self.img_var[i, :] = stat.var
                self.img_skew[i, :] = stat.skew
                self.img_kurtosis[i, :] = stat.kurtosis
                self.img_percentile[i, :, :] = stat.percentiles
                self.img_histogram[i, :, :] = stat.histogram

    # Compute aggregate statistics across the dataset.
    def dataset_stats(self):
        # Calculating aggregate statistics - min, mean, max
        # These stats are listed in the form of (min, mean, max)
        self.height = (
            self.img_height.min(),
            self.img_height.mean(),
            self.img_height.max(),
        )
        self.width = (
            self.img_width.min(),
            self.img_width.mean(),
            self.img_width.max(),
        )
        self.size = (
            self.img_size.min(),
            self.img_size.mean(),
            self.img_size.max(),
        )
        self.aspect_ratio = (
            self.img_aspect_ratio.min(),
            self.img_aspect_ratio.mean(),
            self.img_aspect_ratio.max(),
        )
        self.zeros = (
            self.img_zeros.min(),
            self.img_zeros.mean(),
            self.img_zeros.max(),
        )
        self.avg_brightness = (
            self.img_avg_brightness.min(),
            self.img_avg_brightness.mean(),
            self.img_avg_brightness.max(),
        )
        self.brightness = (
            self.img_brightness.min(),
            self.img_brightness.mean(),
            self.img_brightness.max(),
        )
        self.entropy = (
            self.img_entropy.min(),
            self.img_entropy.mean(),
            self.img_entropy.max(),
        )
        self.blurriness = (
            self.img_blurriness.min(),
            self.img_blurriness.mean(),
            self.img_blurriness.max(),
        )

        # Compute dataset-wide variance, skewness, and kurtosis.
        if self.images.ndim == 3:
            self.images = np.expand_dims(self.images, axis=1)
        self.dataset_var = np.var(self.images, axis=(1, 2, 3))
        self.dataset_skew = sp.stats.skew(self.images)
        self.dataset_kurtosis = sp.stats.kurtosis(self.images)

        # Count missing images and value range occurrences.
        self.missing = np.sum(self.img_missing)
        self.value_range = self.img_range

        # Calculate aggregate statistics per channel - min/channel, mean/channel, max/channel
        # for example 3 channels would give ([1,2,3] - min, [2,3,4] - mean, [3,4,5] - max)
        self.mean = (
            np.nanmin(self.img_mean, axis=0),
            np.nanmean(self.img_mean, axis=0),
            np.nanmax(self.img_mean, axis=0),
        )
        self.var = (
            np.nanmin(self.img_var, axis=0),
            np.nanmean(self.img_var, axis=0),
            np.nanmax(self.img_var, axis=0),
        )
        self.skew = (
            np.nanmin(self.img_skew, axis=0),
            np.nanmean(self.img_skew, axis=0),
            np.nanmax(self.img_skew, axis=0),
        )
        self.kurtosis = (
            np.nanmin(self.img_kurtosis, axis=0),
            np.nanmean(self.img_kurtosis, axis=0),
            np.nanmax(self.img_kurtosis, axis=0),
        )
        self.percentile = (
            np.nanmin(self.img_percentile, axis=0),
            np.nanmean(self.img_percentile, axis=0),
            np.nanmax(self.img_percentile, axis=0),
        )
        self.histogram = (
            np.nanmin(self.img_histogram, axis=0),
            np.nanmean(self.img_histogram, axis=0),
            np.nanmax(self.img_histogram, axis=0),
        )

    # Need to determine still what the output is for this class.


# Old code
class Linting(EvaluateMixin):
    """
    Basic Image and Label Statistics

    Parameters
    ----------
    images : np.ndarray
        A numpy array of n_samples of images either (H, W) or (C, H, W).

    labels : np.ndarray
        A numpy array of n_samples of class labels with M unique classes.

    boxes : np.ndarray
        A numpy array of n_samples of object boxes with P objects per image
        (n_samples, P, H, W)
    """

    def __init__(
        self,
        images: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        boxes: Optional[np.ndarray] = None,
    ) -> None:
        self.images = images
        self.labels = labels
        self.boxes = boxes

    def evaluate(self) -> Dict[str, float]:
        """
        Returns
        -------
        Dict[str, float]

        """
        results: Dict[str, Any] = {}
        if self.images:
            img_stats = self._evaluate_images()
            results["images"] = img_stats
        if self.labels:
            label_stats = self._evaluate_labels()
            results["labels"] = label_stats
        if self.boxes:
            box_stats = self._evaluate_boxes()
            results["boxes"] = box_stats

        return results

    def _evaluate_images_or_boxes(self, data: np.ndarray) -> Dict[str, Any]:
        if data.ndim == 3:  # Assuming (C, H, W)
            data = np.expand_dims(data, axis=0)  # Convert to (N, C, H, W) for consistency
        n_samples, channels, height, width = data.shape

        pixel_values = data.reshape(-1, channels)
        size = height * width
        aspect_ratio = min(width / height, height / width)

        stats = {
            "size": size,
            "aspect_ratio": aspect_ratio,
            "num_channels": channels,
            "pixel_value_mean": np.mean(pixel_values, axis=0),
            "pixel_value_median": np.median(pixel_values, axis=0),
            "pixel_value_max": np.max(pixel_values, axis=0),
            "pixel_value_min": np.min(pixel_values, axis=0),
            "pixel_value_variance": np.var(pixel_values, axis=0),
            "pixel_value_skew": sp.stats.skew(pixel_values, axis=0),
            "pixel_value_kurtosis": sp.stats.kurtosis(pixel_values, axis=0),
            "pixel_value_histogram": [np.histogram(pixel_values[:, i], bins=10)[0].tolist() for i in range(channels)],
        }

        return stats

    def _evaluate_images(self) -> Dict[str, Any]:
        if self.images is None:
            return {}

        return self._evaluate_images_or_boxes(self.images)

    def _evaluate_boxes(self) -> Dict[str, Any]:
        if self.boxes is None:
            return {}

        # Assuming boxes is a (N, P, 4) array where each box is defined by
        # (x_min, y_min, x_max, y_max)

        # Convert boxes to an array of dimensions for the sake of statistical analysis
        box_dimensions = self.boxes[:, :, 2:] - self.boxes[:, :, :2]  # (N, P, 2) where 2 corresponds to (width, height)
        box_dimensions = box_dimensions.reshape(-1, 2)  # Flatten to (N*P, 2)
        boxes_data = np.concatenate(
            [box_dimensions, np.ones((len(box_dimensions), 1))], axis=-1
        )  # Add a dummy channel dimension
        return self._evaluate_images_or_boxes(boxes_data)

    def _evaluate_labels(self) -> Dict[str, Any]:
        if self.labels is None:
            return {}

        label_counts = Counter(self.labels.flatten())
        return {"label_counts": dict(label_counts)}
