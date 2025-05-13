from __future__ import annotations

__all__ = []

import hashlib
import tarfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ARCHIVE_ENDINGS = [".zip", ".tar", ".tgz"]
COMPRESS_ENDINGS = [".gz", ".bz2"]


def _print(text: str, verbose: bool) -> None:
    if verbose:
        print(text)


def _validate_file(fpath: Path | str, file_md5: str, md5: bool = False, chunk_size: int = 65535) -> bool:
    hasher = hashlib.md5(usedforsecurity=False) if md5 else hashlib.sha256()
    with open(fpath, "rb") as fpath_file:
        while chunk := fpath_file.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest() == file_md5


def _download_dataset(url: str, file_path: Path, timeout: int = 60, verbose: bool = False) -> None:
    """Download a single resource from its URL to the `data_folder`."""
    error_msg = "URL fetch failure on {}: {} -- {}"
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"{error_msg.format(url, e.response.status_code, e.response.reason)}") from e
    except requests.exceptions.RequestException as e:
        raise ValueError(f"{error_msg.format(url, 'Unknown error', str(e))}") from e

    total_size = int(response.headers.get("content-length", 0))
    block_size = 8192  # 8 KB
    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True, disable=not verbose)

    with open(file_path, "wb") as f:
        for chunk in response.iter_content(block_size):
            f.write(chunk)
            progress_bar.update(len(chunk))
    progress_bar.close()


def _extract_zip_archive(file_path: Path, extract_to: Path) -> None:
    """Extracts the zip file to the given directory."""
    try:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)  # noqa: S202
            file_path.unlink()
    except zipfile.BadZipFile:
        raise FileNotFoundError(f"{file_path.name} is not a valid zip file, skipping extraction.")


def _extract_tar_archive(file_path: Path, extract_to: Path) -> None:
    """Extracts a tar file (or compressed tar) to the specified directory."""
    try:
        with tarfile.open(file_path, "r:*") as tar_ref:
            tar_ref.extractall(extract_to)  # noqa: S202
            file_path.unlink()
    except tarfile.TarError:
        raise FileNotFoundError(f"{file_path.name} is not a valid tar file, skipping extraction.")


def _extract_archive(
    file_ext: str, file_path: Path, directory: Path, compression: bool = False, verbose: bool = False
) -> None:
    """
    Single function to extract and then flatten if necessary.
    Recursively extracts nested zip files as well.
    Extracts and flattens all folders to the base directory.
    """
    if file_ext != ".zip" or compression:
        _extract_tar_archive(file_path, directory)
    else:
        _extract_zip_archive(file_path, directory)
    # Look for nested zip files in the extraction directory and extract them recursively.
    # Does NOT extract in place - extracts everything to directory
    for child in directory.iterdir():
        if child.suffix == ".zip":
            _print(f"Extracting nested zip: {child} to {directory}", verbose)
            _extract_zip_archive(child, directory)


def _ensure_exists(
    url: str,
    filename: str,
    md5: bool,
    checksum: str,
    directory: Path,
    root: Path,
    download: bool = True,
    verbose: bool = False,
) -> None:
    """
    For each resource, download it if it doesn't exist in the dataset_dir.
    If the resource is a zip file, extract it (including recursively extracting nested zips).
    """
    file_path = directory / str(filename)
    alternate_path = root / str(filename)
    _, file_ext = file_path.stem, file_path.suffix
    compression = False
    if file_ext in COMPRESS_ENDINGS:
        file_ext = file_path.suffixes[0]
        compression = True

    check_path = alternate_path if alternate_path.exists() and not file_path.exists() else file_path

    # Download file if it doesn't exist.
    if not check_path.exists() and download:
        _print(f"Downloading {filename} from {url}", verbose)
        _download_dataset(url, check_path, verbose=verbose)

        if not _validate_file(check_path, checksum, md5):
            raise Exception("File checksum mismatch. Remove current file and retry download.")

        # If the file is a zip, tar or tgz extract it into the designated folder.
        if file_ext in ARCHIVE_ENDINGS:
            _print(f"Extracting {filename}...", verbose)
            _extract_archive(file_ext, check_path, directory, compression, verbose)

    elif not check_path.exists() and not download:
        raise FileNotFoundError(
            "Data could not be loaded with the provided root directory, ",
            f"the file path to the file {filename} does not exist, ",
            "and the download parameter is set to False.",
        )
    else:
        if not _validate_file(check_path, checksum, md5):
            raise Exception("File checksum mismatch. Remove current file and retry download.")
        _print(f"{filename} already exists, skipping download.", verbose)

        if file_ext in ARCHIVE_ENDINGS:
            _print(f"Extracting {filename}...", verbose)
            _extract_archive(file_ext, check_path, directory, compression, verbose)
