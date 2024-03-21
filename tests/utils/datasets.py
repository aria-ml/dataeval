import hashlib
import os
import typing
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve


def _validate_file(fpath, file_hash, chunk_size=65535):
    hasher = hashlib.sha256()
    with open(fpath, "rb") as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b""):
            hasher.update(chunk)

    return str(hasher.hexdigest()) == str(file_hash)


def _get_file(
    fname: str,
    origin: str,
    file_hash: typing.Optional[str] = None,
):
    cache_dir = os.path.join(os.path.expanduser("~"), ".keras")
    datadir_base = os.path.expanduser(cache_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join("/tmp", ".keras")
    datadir = os.path.join(datadir_base, "datasets")
    os.makedirs(datadir, exist_ok=True)

    fname = os.fspath(fname) if isinstance(fname, os.PathLike) else fname
    fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath):
        if file_hash is not None and not _validate_file(fpath, file_hash):
            download = True
    else:
        download = True

    if download:
        try:
            error_msg = "URL fetch failure on {}: {} -- {}"
            try:
                urlretrieve(origin, fpath)
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg)) from e
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason)) from e
        except (Exception, KeyboardInterrupt):
            if os.path.exists(fpath):
                os.remove(fpath)
            raise

        if (
            os.path.exists(fpath)
            and file_hash is not None
            and not _validate_file(fpath, file_hash)
        ):
            raise ValueError(
                "Incomplete or corrupted file detected. "
                f"The sha256 file hash does not match the provided value "
                f"of {file_hash}.",
            )
    return fpath


def download_mnist() -> str:
    """Code to download mnist originates from keras/datasets:

    https://github.com/keras-team/keras/blob/v2.15.0/keras/datasets/mnist.py#L25-L86
    """
    origin_folder = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    path = _get_file(
        "mnist.npz",
        origin=origin_folder + "mnist.npz",
        file_hash=("731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"),
    )

    return path
