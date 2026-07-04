"""Utility to download and uncompress the test data from Zenodo."""

import os
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
        "https://zenodo.org/records/21192512/files/test_data.zip",
        timeout=60,
    )
    if not req.ok:
        raise RuntimeError(
            f"Failed to download test data: {req.status_code} {req.reason}"
        )
    with open(result, "wb") as download:
        download.write(req.content)
    return result


def get_test_data(destination, local_source=None):
    """Populate ``destination`` with the test data.

    If ``local_source`` is given, it must point to either a zip file
    (extracted into ``destination``) or a directory whose contents are
    copied into ``destination``. Otherwise, the test data zip is
    downloaded from Zenodo and extracted.
    """

    if local_source is not None:
        local_source = path.abspath(local_source)
        if path.isdir(local_source):
            print(
                f"Copying local test data from {local_source!r} to "
                f"{destination!r} ..."
            )
            for entry in os.listdir(local_source):
                src = path.join(local_source, entry)
                dst = path.join(destination, entry)
                if path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
            return
        if path.isfile(local_source):
            print(
                f"Unzipping local test data {local_source!r} to "
                f"{destination!r} ..."
            )
            with ZipFile(local_source, "r") as zip_ref:
                zip_ref.extractall(destination)
            return
        raise FileNotFoundError(
            f"Local test data source {local_source!r} is neither a file "
            "nor a directory."
        )

    print(f"Downloading test data to {destination!r} ...")
    with ZipFile(download_zip(destination), "r") as zip_ref:
        print(f"Unzipping all files to {destination!r} ...")
        zip_ref.extractall(destination)


if __name__ == "__main__":
    with TemporaryDirectory() as temp_dir:
        get_test_data(temp_dir)
