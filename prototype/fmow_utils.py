from datetime import date, datetime
from pathlib import Path

import numpy as np
from PIL import Image


def get_image_sizes(pth):
    fns = Path(pth).glob("*.tif")
    sz = {}
    for fn in fns:
        if fn.name[0] == ".":
            continue
        im = Image.open(str(fn))
        sz[fn.name] = im.size
    return sz


def get_datetime(df):
    return df.timestamp.map(datetime.fromisoformat)


def get_month(df):
    dt = get_datetime(df)
    return np.array([t.month for t in dt])


def get_day_of_year(df):
    dt = get_datetime(df)
    return np.array([t.timetuple().tm_yday for t in dt])


def get_season(df):
    dts = get_datetime(df)
    Y = 2000  # dummy leap year to allow input X-02-29 (leap day)
    seasons = [
        (0, (date(Y, 1, 1), date(Y, 3, 20))),
        (1, (date(Y, 3, 21), date(Y, 6, 20))),
        (2, (date(Y, 6, 21), date(Y, 9, 22))),
        (3, (date(Y, 9, 23), date(Y, 12, 20))),
        (0, (date(Y, 12, 21), date(Y, 12, 31))),
    ]

    def get_season_(dt):
        if isinstance(dt, datetime):
            dt = dt.date()
        dt = dt.replace(year=Y)
        return next(season for season, (start, end) in seasons if start <= dt <= end)

    # raw seasons (northern hemisphere)
    ssn = np.array([get_season_(d) for d in dts])
    # adjust for southern hemisphere (zones A through M)
    is_southern_hemisphere = np.array([code[-1].upper() < "N" for code in df.utm]).astype(int)
    # add two seasons mod 4
    return (ssn + 2 * is_southern_hemisphere) % 4


def get_fmow_boxes(tab):
    # numpy array of
    bbox = [bb[0]["box"] for bb in tab.bounding_boxes]
    return np.array(bbox)


def extrinsic_factors_fmow(df):
    ext = {}
    ext["month"] = get_month(df)
    ext["day_of_year"] = get_day_of_year(df)
    ext["season"] = get_season(df)

    _vars = [
        "utm",
        "country_code",
        "cloud_cover",
        "target_azimuth_dbl",
        "sun_azimuth_dbl",
        "sun_elevation_dbl",
        "off_nadir_angle_dbl",
    ]
    for v in _vars:
        ext[v] = df[v].to_numpy()

    # map variable names to boolean is_categorical
    is_categorical = {
        "utm": True,
        "country_code": True,
        "cloud_cover": True,
        "gsd": True,
        "target_azimuth_dbl": False,
        "sun_azimuth_dbl": False,
        "sun_elevation_dbl": False,
        "off_nadir_angle_dbl": False,
        "month": True,
        "season": True,
        "day_of_year": True,
    }

    assert all(k in is_categorical for k in ext)

    return ext, is_categorical


def validate_dict(d):
    # assert that length of all arrays are equal -- could be expanded to
    # checking values, fraction of unique values, etc.
    lengths = []
    for arr in d.values():
        lengths.append(arr.shape)
    assert lengths[1:] == lengths[:-1]


def cat_to_int(x):
    """
    map categorical variables to numbers that mutual_information can accommodate
    """
    ux, mapped_vals = np.unique(x, return_inverse=True)
    return np.array(mapped_vals), ux
