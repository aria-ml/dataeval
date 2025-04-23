import hashlib
from pathlib import Path

import pytest
import requests
from requests import RequestException, Response

from dataeval.utils.datasets._fileio import (
    _archive_extraction,
    _download_dataset,
    _ensure_exists,
    _extract_tar_archive,
    _extract_zip_archive,
    _flatten_extraction,
    _validate_file,
)

TEMP_MD5 = "d149274109b50d5147c09d6fc7e80c71"
TEMP_SHA256 = "2b749913055289cb3a5c602a17196b5437dc59bba50e986ea449012a303f7201"


def get_tmp_hash(fpath, chunk_size=65535):
    hasher = hashlib.md5()
    with open(fpath, "rb") as fpath_file:
        while chunk := fpath_file.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


class MockHTTPError(Response):
    def __init__(self):
        super().__init__()
        self.reason = "MockError"
        self.status_code = 404


class MockRequestException(Response):
    def __init__(self):
        self.reason = "MockError"
        self.status_code = 404

    def raise_for_status(self):
        raise RequestException


@pytest.mark.optional
class TestHelperFunctionsBaseDataset:
    @pytest.mark.parametrize("verbose", [True, False])
    def test_ensure_exists_no_zip(self, capsys, dataset_no_zip, verbose):
        resource = ("fakeurl", "stuff.txt", True, TEMP_MD5)
        _ensure_exists(*resource, dataset_no_zip.parent, dataset_no_zip.parent, True, verbose)
        if verbose:
            captured = capsys.readouterr()
            assert captured.out == "stuff.txt already exists, skipping download.\n"

    @pytest.mark.parametrize("verbose", [True, False])
    def test_ensure_exists_single_zip(self, capsys, dataset_single_zip, verbose):
        checksum = get_tmp_hash(dataset_single_zip)
        resource = ("fakeurl", "testing.zip", True, checksum)
        _ensure_exists(*resource, dataset_single_zip.parent, dataset_single_zip.parent, True, verbose)
        if verbose:
            captured = capsys.readouterr()
            assert "Extracting testing.zip..." in captured.out

    def test_ensure_exists_file_exists_bad_checksum(self, dataset_no_zip):
        resource = ("fakeurl", "stuff.txt", True, TEMP_SHA256)
        err_msg = "File checksum mismatch. Remove current file and retry download."
        with pytest.raises(Exception) as e:
            _ensure_exists(*resource, dataset_no_zip.parent, dataset_no_zip.parent, False)
        assert err_msg in str(e.value)

    def test_ensure_exists_download_non_zip(self, capsys, mnist_folder):
        resource = (
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
            "mnist.npz",
            False,
            "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1",
        )
        _ensure_exists(*resource, mnist_folder, mnist_folder.parent, True, True)
        captured = capsys.readouterr()
        assert (
            "Downloading mnist.npz from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
            in captured.out
        )

    def test_ensure_exists_download_bad_checksum(self, mnist_folder):
        resource = (
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
            "mnist.npz",
            False,
            "abc",
        )
        err_msg = "File checksum mismatch. Remove current file and retry download."
        with pytest.raises(Exception) as e:
            _ensure_exists(*resource, mnist_folder, mnist_folder.parent, True, False)
        assert err_msg in str(e.value)

    @pytest.mark.parametrize("verbose", [True, False])
    def test_ensure_exists_download_zip(self, capsys, mnist_folder, verbose):
        resource = (
            "https://figshare.com/ndownloader/files/43168999",
            "2021.zip",
            True,
            "b84749b21fa95a4a4c7de3741db78bc7",
        )
        _ensure_exists(*resource, mnist_folder, mnist_folder.parent, True, verbose)
        if verbose:
            captured = capsys.readouterr()
            assert f"Extracting {resource[1]}..." in captured.out

    def test_ensure_exists_error(self, dataset_no_zip):
        resource = ("fakeurl", "something.zip", True, "")
        err_msg = "Data could not be loaded with the provided root directory,"
        with pytest.raises(FileNotFoundError) as e:
            _ensure_exists(*resource, dataset_no_zip.parent, dataset_no_zip.parent, False)
        assert err_msg in str(e.value)

    def test_download_dataset_http_error(self, monkeypatch):
        def mock_get(*args, **kwargs):
            return MockHTTPError()

        monkeypatch.setattr(requests, "get", mock_get)
        with pytest.raises(RuntimeError):
            _download_dataset(url="http://mock/", file_path=Path("fake/path"))

    def test_download_dataset_request_error(self, monkeypatch):
        def mock_get(*args, **kwargs):
            return MockRequestException()

        monkeypatch.setattr(requests, "get", mock_get)
        with pytest.raises(ValueError):
            _download_dataset(url="http://mock/", file_path=Path("fake/path"))

    @pytest.mark.parametrize("use_md5, hash_value", [(True, TEMP_MD5), (False, TEMP_SHA256)])
    def test_validate_file(self, dataset_no_zip, use_md5, hash_value):
        assert _validate_file(dataset_no_zip, hash_value, use_md5)

    @pytest.mark.parametrize("verbose", [True, False])
    def test_zip_extraction_nested_zip(self, capsys, dataset_nested_zip, verbose):
        _archive_extraction(dataset_nested_zip.suffix, dataset_nested_zip, dataset_nested_zip.parent, False, verbose)
        if verbose:
            captured = capsys.readouterr()
            assert "Extracting nested zip" in captured.out

    def test_extract_archive_bad_zip(self, dataset_no_zip):
        err_msg = f"{dataset_no_zip.name} is not a valid zip file, skipping extraction."
        with pytest.raises(FileNotFoundError) as e:
            _extract_zip_archive(dataset_no_zip, dataset_no_zip.parent)
        assert err_msg in str(e.value)

    @pytest.mark.parametrize("verbose", [True, False])
    def test_flatten_extraction(self, capsys, dataset_nested_folder, verbose):
        base = dataset_nested_folder.parent
        _extract_zip_archive(dataset_nested_folder, base)
        _flatten_extraction(
            base,
            verbose,
        )
        if verbose:
            captured = capsys.readouterr()
            assert "Moving translate to /tmp" in captured.out
            assert "Removing empty folder mnist_c" in captured.out
        assert (base / "translate").exists()

    def test_tarfile_error(self, dataset_single_zip):
        err_msg = f"{dataset_single_zip.name} is not a valid tar file"
        with pytest.raises(FileNotFoundError) as e:
            _extract_tar_archive(file_path=dataset_single_zip, extract_to=dataset_single_zip.parent)
        assert err_msg in str(e.value)
