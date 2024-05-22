from dataclasses import dataclass
from typing import Generic, NamedTuple, Optional, TypeVar, Union

import numpy as np
from scipy.signal import convolve2d
from scipy.stats import entropy, kurtosis, skew

EDGE_KERNEL = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.int8)
BIT_DEPTH = (1, 8, 12, 16, 32)

TStatValue = TypeVar("TStatValue", np.number, np.ndarray)


class BitDepth(NamedTuple):
    depth: int
    pmin: Union[float, int]
    pmax: Union[float, int]


def get_bitdepth(image: np.ndarray) -> BitDepth:
    """
    Approximates the bit depth of the image using the
    min and max pixel values.
    """
    pmin, pmax = np.min(image), np.max(image)
    if pmin < 0:
        return BitDepth(0, pmin, pmax)
    else:
        depth = ([x for x in BIT_DEPTH if 2**x > pmax] or [max(BIT_DEPTH)])[0]
        return BitDepth(depth, 0, 2**depth - 1)


def rescale(image: np.ndarray, depth: int = 1) -> np.ndarray:
    """
    Rescales the image using the bit depth provided.
    """
    bitdepth = get_bitdepth(image)
    if bitdepth.depth == depth:
        return image
    else:
        normalized = (image + bitdepth.pmin) / (bitdepth.pmax - bitdepth.pmin)
        return normalized * (2**depth - 1)


def normalize_image_shape(image: np.ndarray) -> np.ndarray:
    """
    Normalizes the image shape into (C,H,W).
    """
    ndim = image.ndim
    if ndim == 2:
        return np.expand_dims(image, axis=0)
    elif ndim == 3:
        return image
    elif ndim > 3:
        # Slice all but the last 3 dimensions
        return image[(0,) * (ndim - 3)]
    else:
        raise ValueError("Images must have 2 or more dimensions.")


def edge_filter(image: np.ndarray, offset: float = 0.5) -> np.ndarray:
    """
    Returns the image filtered using a 3x3 edge detection kernel:
    [[ -1, -1, -1 ],
     [ -1,  8, -1 ],
     [ -1, -1, -1 ]]
    """
    edges = convolve2d(image, EDGE_KERNEL, mode="same", boundary="symm") + offset
    return np.clip(edges, 0, 255)


@dataclass
class ImageStats(Generic[TStatValue]):
    """
    Dataclass containing annotations for all of the supported
    metrics for image statistics.
    """

    height: TStatValue
    width: TStatValue
    size: TStatValue
    aspect_ratio: TStatValue
    depth: TStatValue
    channels: TStatValue
    missing: TStatValue
    brightness: TStatValue
    blurriness: TStatValue
    mean: TStatValue
    zero: TStatValue
    var: TStatValue
    skew: TStatValue
    kurtosis: TStatValue
    percentiles: np.ndarray
    histogram: np.ndarray
    entropy: TStatValue


@dataclass
class ChannelStats:
    """
    Dataclass containing annotations for all of the supported
    metrics for channel specific statistics.
    """

    ch_mean: np.ndarray
    ch_var: np.ndarray
    ch_skew: np.ndarray
    ch_kurtosis: np.ndarray
    ch_percentiles: np.ndarray
    ch_histogram: np.ndarray


class SingleImageStats(ImageStats[np.number], ChannelStats):
    """
    Calculates image and channel statistics.
    """

    def __init__(self, image: np.ndarray):
        # Normalize the image dimensions
        image = normalize_image_shape(image)
        bitdepth = get_bitdepth(image)
        scaled = rescale(image)

        # Image stats
        self.channels = np.int32(image.shape[0])
        self.height = np.int32(image.shape[-2])
        self.width = np.int32(image.shape[-1])
        self.size = self.height * self.width
        self.aspect_ratio = self.width / self.height
        self.depth = np.int32(bitdepth.depth)
        self.blurriness = np.std(edge_filter(np.mean(image, axis=0)))
        self.missing = np.sum(np.isnan(image))
        self.zero = np.int32(np.count_nonzero(image == 0))

        self.mean = np.mean(scaled)
        self.brightness = self.mean  # TODO: replace with a better calculation
        self.var = np.var(scaled)
        self.skew = np.float32(skew(scaled.ravel()))
        self.kurtosis = np.float32(kurtosis(scaled.ravel()))
        self.percentiles = np.percentile(scaled, q=[0, 25, 50, 75, 100])

        # Channel stats
        ch_flat = scaled.reshape(self.channels, -1)
        self.ch_mean = np.mean(scaled, axis=(1, 2))
        self.ch_var = np.var(scaled, axis=(1, 2))
        self.ch_skew = skew(ch_flat, axis=1)
        self.ch_kurtosis = kurtosis(ch_flat, axis=1)
        self.ch_percentiles = np.percentile(scaled, q=[0, 25, 50, 75, 100], axis=(1, 2)).T
        self.ch_histogram = np.apply_along_axis(lambda x: np.histogram(x, bins=256, range=(0, 1))[0], 1, ch_flat)

        self.histogram = np.sum(self.ch_histogram, axis=0)
        self.entropy = np.float32(entropy(self.histogram))


# Pulls in a dataset and then gets the individual image stats
# then runs group stats
# Class to encapsulate dataset statistics calculations
class DatasetStats(ImageStats[np.ndarray], ChannelStats):
    def __init__(
        self,
        images,
        labels: Optional[np.ndarray] = None,
        boxes: Optional[np.ndarray] = None,
    ) -> None:
        # Initialization of DatasetStats with datasets images, optional labels, and optional bounding boxes.
        # self.images = images
        # self.labels = labels
        # self.boxes = boxes
        self.length = len(images)
        self.ch_map = np.empty((0, 3), dtype=np.uint32)  # [index, channels, channel_index]
        channel_stats = {}

        # Iterate through images
        for i, image in enumerate(images):
            stats = SingleImageStats(image)

            # Aggregate the image statistics
            for stat in ImageStats.__annotations__:
                image_stat = getattr(stats, stat)
                aggregated_stat = getattr(self, stat, None)
                if aggregated_stat is None:
                    shape = () if np.isscalar(image_stat) else image_stat.shape
                    aggregated_stat = np.empty((self.length,) + shape)
                    setattr(self, stat, aggregated_stat)
                aggregated_stat[i] = image_stat

            # Build the image and channel mapping for each channel
            c = int(stats.channels)
            image_ch_map = np.linspace((i, c, 0), (i, c, c - 1), c, dtype=np.uint32)
            self.ch_map = np.concatenate((self.ch_map, image_ch_map))

            # Aggregate the channel statistics
            for stat in ChannelStats.__annotations__:
                channel_stat = getattr(stats, stat)
                if stat not in channel_stats:
                    channel_stats[stat] = [channel_stat]
                else:
                    channel_stats[stat].append(channel_stat)

        # Concatenate all channel_stats into aggregatged stats
        for stat, channel_stat in channel_stats.items():
            aggregated_channel_stat = np.concatenate(channel_stat)
            setattr(self, stat, aggregated_channel_stat)

    def get_channel_mask(self, channels: int, channel: Optional[int] = None) -> np.ndarray:
        """
        Returns a mask for channel stats based on desired channel count and channel
        """
        return (
            self.ch_map[:, 1] == channels
            if channel is None
            else (self.ch_map[:, (1, 2)] == (channels, channel)).all(axis=1)
        )
