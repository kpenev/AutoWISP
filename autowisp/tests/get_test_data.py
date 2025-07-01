"""Utility to download and uncompress the test data from Zenodo."""

from os import path
from zipfile import ZipFile
from tempfile import TemporaryDirectory

import requests


def download_zip(destination):
    """Download the test data zip file from Zenodo."""

    req = requests.get(
        "https://zenodo.org/records/15786531/files/test_data.zip",
        timeout=60,
    )
    if not req.ok:
        raise RuntimeError(
            f"Failed to download test data: {req.status_code} {req.reason}"
        )
    result = path.join(destination, "test.zip")
    with open(result, "wb") as download:
        download.write(req.content)
    return result


def get_test_data(destination):
    """Uncompress the zip file to the specified destination."""

    with ZipFile(download_zip(destination), "r") as zip_ref:
        zip_ref.extractall(destination)


if __name__ == "__main__":
    with TemporaryDirectory() as temp_dir:
        print(f"Downloading test data to {temp_dir!r} ...")
        get_test_data(temp_dir)
