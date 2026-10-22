"""Check the BUI reports each installed distribution independently."""

from importlib.metadata import PackageNotFoundError
import unittest
from unittest.mock import call, patch

from autowisp.browser_interface.core.package_versions import get_installed_versions


class TestInstalledVersions(unittest.TestCase):
    """Version display uses distribution metadata and handles missing packages."""

    @patch("autowisp.browser_interface.core.package_versions.version")
    def test_separate_installed_versions(self, metadata_version):
        """The two packages can have different release versions."""
        metadata_version.side_effect = ["1.2.3", "2.4.0"]
        self.assertEqual(
            get_installed_versions(),
            {"autowisp": "1.2.3", "astrowisp": "2.4.0"},
        )
        self.assertEqual(
            metadata_version.call_args_list,
            [call("autowisp"), call("astrowisp")],
        )

    @patch("autowisp.browser_interface.core.package_versions.version")
    def test_missing_autowisp(self, metadata_version):
        """A source checkout must not stand in for an installed release."""
        metadata_version.side_effect = [PackageNotFoundError("autowisp"), "2.4.0"]
        self.assertEqual(
            get_installed_versions(),
            {"autowisp": "Not installed", "astrowisp": "2.4.0"},
        )

    @patch("autowisp.browser_interface.core.package_versions.version")
    def test_missing_astrowisp(self, metadata_version):
        """Missing AstroWISP metadata does not hide AutoWISP's version."""
        metadata_version.side_effect = ["1.2.3", PackageNotFoundError("astrowisp")]
        self.assertEqual(
            get_installed_versions(),
            {"autowisp": "1.2.3", "astrowisp": "Not installed"},
        )

    @patch("autowisp.browser_interface.core.package_versions.version")
    def test_versions_are_refreshed(self, metadata_version):
        """A new page request reads metadata again after an upgrade."""
        metadata_version.side_effect = ["1.2.3", "2.4.0", "1.2.4", "2.4.1"]
        get_installed_versions()
        self.assertEqual(
            get_installed_versions(),
            {"autowisp": "1.2.4", "astrowisp": "2.4.1"},
        )


if __name__ == "__main__":
    unittest.main()
