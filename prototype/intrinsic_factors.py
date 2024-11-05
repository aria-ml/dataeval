from typing import Dict, Tuple

import numpy as np


def compute_hwa_xyxy(xyxy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute width, height, size/area, and :term:`aspect ratio<Aspect Ratio>` for bounding boxes.
    Compute in one function to reuse quantities.
    Catch divide by zero error in aspect ratio calculation with epsilon
    Computed together for reuse.

    Parameters
    ----------
    xyxy: np.ndarray[num_boxes, 4]
        bounding boxes in [x1,y1,x2,y2] format.  Size

    Returns
    -------
    height: np.ndarray
        box height (second dimension)
    width: np.ndarray
        box width (first dimension)
    area: np.ndarray
        box area (width * height)
    aspect_ratio: np.ndarray
        maximum of w/h and h/w, catch divide by zero


    """
    height = xyxy[:, 2] - xyxy[:, 0]
    width = xyxy[:, 3] - xyxy[:, 1]
    area = width * height
    aspect_ratio = np.maximum(width / (height + 1e-7), height / (width + 1e-7))
    return height, width, area, aspect_ratio


def box_location_features(xyxy: np.ndarray, img_szs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute distance to center and distance to edge of image.  One might be more
    meaningful than the other depending on size of the image.

    Parameters
    ----------
        xyxy: np.ndarray[num_boxes, 4]
            bounding boxes in [x1,y1,x2,y2] format

        img_szs: np.ndarray[num_boxes, 2]
            Image sizes corresponding to boxes in xyxy

    Returns
    -------
    d2c: np.ndarray
        Distance from box center to image center
    d2e: np.ndarray
        Minimum distance from box edge to image edge.
    """

    d2c, d2e = [], []
    box_ctr = xyxy[:, 2:4] - xyxy[:, :2]
    img_ctr = img_szs / 2
    d2c = np.sqrt(np.sum((box_ctr - img_ctr).astype(float) ** 2, axis=1))
    d2e = np.max(
        np.array(
            [
                xyxy[:, 0],  # box min close to top edge
                xyxy[:, 1],  # box min close to left edge
                np.abs(xyxy[:, 2] - img_szs[:, 0]),  # bottom edge of box near bottom of image
                np.abs(xyxy[:, 3] - img_szs[:, 1]),  # box max near right edge of image
            ]
        ),
        axis=0,
    )
    return d2c, d2e


def box2center_xywh(xywh: np.ndarray, img_szs: np.ndarray) -> np.ndarray:
    """
    Compute distance to center and distance to edge of image
        one might be more meaningful than the other depending on size of the image

    Parameters
    ----------
    xyxy: np.ndarray[num_boxes, 4]
        bounding boxes in [x1,y1,x2,y2] format

    img_szs: np.ndarray[num_boxes, 2]
        Image sizes corresponding to boxes in xyxy

    Returns
    -------
    d2c: np.ndarray
        Distance from box center to image center
    """

    box_ctr = xywh[:, :1]
    img_ctr = img_szs / 2
    return np.sqrt(np.sum((box_ctr - img_ctr).astype(float) ** 2, axis=1))


def box2edge_xywh(xywh, img_szs):
    """
    Compute distance to center and distance to edge of image
        one might be more meaningful than the other depending on size of the image

    Parameters
    ----------
    xyxy: np.ndarray[num_boxes, 4]
        bounding boxes in [x1,y1,x2,y2] format

    img_szs: np.ndarray[num_boxes, 2]
        Image sizes corresponding to boxes in xyxy

    Returns
    -------
    d2e: np.ndarray
        Minimum distance from box edge to image edge
    """

    d2e = []
    box_ctr = xywh[:, :1]
    box_min = box_ctr - xywh[:, 2:]
    box_max = box_ctr + xywh[:, 2:]
    d2e = np.max(
        np.array(
            [
                box_min[:, 0],  # box min close to top edge
                box_min[:, 1],  # box min close to left edge
                np.abs(box_max[:, 0] - img_szs[:, 0]),  # bottom edge of box near bottom of image
                np.abs(box_max[:, 1] - img_szs[:, 1]),  # box max near right edge of image
            ]
        ),
        axis=0,
    )
    return d2e


def intrinsic_factors_xywh(xywh: np.ndarray, img_sizes: np.ndarray) -> Tuple[Dict, Dict]:
    """
    Collect intrinsic metadata factors into a dictionary for boxes with xywh
    format

    Parameters
    ----------
    xywh: np.ndarray
        Bounding boxes in [x,y,w,h] format
    img_sizes: np.ndarray
        Image sizes for each bounding box

    Returns
    -------
    prop: Dict[np.ndarray]
        Dictionary where keys are intrinsic metadata factors, values are :term:`NumPy`
        arrays of values.
    is_categorical: Dict[np.ndarray]
        Dictionary specifying whether each key in prop is a categorical variable
        or not.

    """
    prop = {}
    prop["box_width"] = xywh[:, 2]
    prop["box_height"] = xywh[:, 3]
    prop["box_area"] = xywh[:, 2] * xywh[:, 3]
    prop["box_aspect_ratio"] = np.minimum(xywh[:, 2] / (xywh[:, 3] + 1e-6), xywh[:, 3] / (xywh[:, 2] + 1e-6))

    prop["dist_to_center"] = box2center_xywh(xywh, img_sizes)
    prop["dist_to_edge"] = box2edge_xywh(xywh, img_sizes)

    is_categorical = {key: False for key in prop}
    return prop, is_categorical
