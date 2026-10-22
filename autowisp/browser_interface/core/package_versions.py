"""Read the package versions installed in the BUI's Python environment."""

from importlib.metadata import PackageNotFoundError, version


def get_installed_versions():
    """Return installed versions, without consulting working-tree code."""

    versions = {}
    for package in ("autowisp", "astrowisp"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "Not installed"
    return versions
