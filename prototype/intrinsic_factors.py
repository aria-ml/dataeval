import numpy as np


def compute_hwa_xyxy(xyxy):
    """
    Compute width, height, size/area, and aspect ratio for bounding boxes.
    Compute in one function to reuse quantities.
    Catch divide by zero error in aspect ratio calculation with epsilon

    Inputs:
        xyxy: num_boxes x 4 array of bounding boxes
    """
    height = xyxy[:, 2] - xyxy[:, 0]
    width = xyxy[:, 3] - xyxy[:, 1]
    area = width * height
    ar = np.minimum(width / (height + 1e-7), height / (width + 1e-7))
    return height, width, area, ar


# def box_to_img_center(xyxy, img_sizes):


def box_location_features(xyxy, img_szs):
    """
    Compute distance to center and distance to edge of image
        one might be more meaningful than the other depending on size of the image

    Inputs
        xyxy: xyxy (pixels)
        img_szs: numpy array of image sizes corresponding to boxes in xyxy
    """

    d2c, d2e = [], []
    box_ctr = xyxy[:, 2:4] - xyxy[:, :2]
    # img_szs = np.array([sz for sz in sizes])
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


def box2center_xywh(xywh, img_szs):
    """
    Compute distance to center and distance to edge of image
        one might be more meaningful than the other depending on size of the image

    Inputs
        xyxy: xyxy (pixels)
        img_szs: numpy array of image sizes corresponding to boxes in xyxy
    """

    # d2c, d2e = [], []
    box_ctr = xywh[:, :1]
    img_ctr = img_szs / 2
    return np.sqrt(np.sum((box_ctr - img_ctr).astype(float) ** 2, axis=1))


def box2edge_xywh(xywh, img_szs):
    """
    Compute distance to center and distance to edge of image
        one might be more meaningful than the other depending on size of the image

    Inputs
        xyxy: xyxy (pixels)
        img_szs: numpy array of image sizes corresponding to boxes in xyxy
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


def intrinsic_factors_xywh(xywh, img_sizes):
    # changing this for FMOW in xywh format
    prop = {}
    prop["box_width"] = xywh[:, 2]
    prop["box_height"] = xywh[:, 3]
    prop["box_area"] = xywh[:, 2] * xywh[:, 3]
    prop["box_aspect_ratio"] = np.minimum(xywh[:, 2] / (xywh[:, 3] + 1e-6), xywh[:, 3] / (xywh[:, 2] + 1e-6))

    prop["dist_to_center"] = box2center_xywh(xywh, img_sizes)
    prop["dist_to_edge"] = box2edge_xywh(xywh, img_sizes)

    is_categorical = {key: False for key in prop}
    return prop, is_categorical
