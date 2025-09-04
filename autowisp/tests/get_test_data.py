"""Utility to download and uncompress the test data from Zenodo."""

from os import path
import shutil
from zipfile import ZipFile
from tempfile import TemporaryDirectory

import requests


def download_zip(destination):
    """Download the test data zip file from Zenodo."""

    print("Starting download")
    result = path.join(destination, "test_data.zip")
    if path.exists("test_data.zip"):
        shutil.copyfile("test_data.zip", result)
        print("Re-using existing 'test_data.zip'")
        return result
    req = requests.get(
        "https://zenodo.org/records/17058754/files/test_data.zip",
        timeout=60,
    )
    if not req.ok:
        raise RuntimeError(
            f"Failed to download test data: {req.status_code} {req.reason}"
        )
    with open(result, "wb") as download:
        download.write(req.content)
    return result


def get_test_data(destination):
    """Uncompress the zip file to the specified destination."""

    print(f"Downloading test data to {destination!r} ...")
    with ZipFile(download_zip(destination), "r") as zip_ref:
        print(f"Unzipping all files to {destination!r} ...")
        zip_ref.extractall(destination)


if __name__ == "__main__":
    with TemporaryDirectory() as temp_dir:
        get_test_data(temp_dir)
