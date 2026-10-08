"""Tests for restricting light curves to a subset of the DR sources.

Two mechanisms are exercised (see ``lc_filter``):

1. A brighter magnitude limit for the light-curve catalog than for the
   photometry catalog (:class:`TestLCFilter.test_bright_mag_limit`).
2. An explicit list of GAIA source IDs to keep
   (:class:`TestLCFilter.test_manual_source_list`), and the two combined
   (:class:`TestLCFilter.test_combined`).

The processing branches are *subset self-consistency* checks: the restricted
run must reproduce, bit for bit, the light curves and EPD statistics of the
full-catalog fixtures -- only for the sources that survive the restriction.
EPD is per-source independent, so kept sources are identical; TFA would not be,
which is why only EPD is checked.

``TestCatalogSourceListFilter`` is a hermetic unit test of the read-time
``source_id`` filtering added to :func:`autowisp.catalog.read_catalog_file`.
"""

from glob import glob
from os import path
import os

import pandas
import h5py
from astropy.table import Table

from autowisp.catalog import (
    read_catalog_file,
    read_source_id_list,
    get_catalog_config,
    get_catalog_info,
)
from autowisp.processing_steps import create_lightcurves
from autowisp.data_reduction.data_reduction_file import DataReductionFile
from autowisp.tests import AutoWISPTestCase
from autowisp.tests.h5_test_case import H5TestCase


class TestCatalogSourceListFilter(AutoWISPTestCase):
    """Read-time ``source_id`` filtering in ``read_catalog_file``."""

    def _a_catalog(self):
        """Return the path to a staged Gaia catalog fixture."""

        catalogs = sorted(
            glob(
                path.join(
                    self.processing_directory, "MASTERS", "Gaia", "*.fits"
                )
            )
        )
        self.assertTrue(catalogs, "No staged Gaia catalog to test against.")
        return catalogs[0]

    def test_source_id_filter(self):
        """Only the requested source IDs are returned, others ignored."""

        catalog_fname = self._a_catalog()
        full = read_catalog_file(catalog_fname)
        all_ids = [str(source_id) for source_id in full.index]
        self.assertGreater(len(all_ids), 4, "Catalog fixture too small.")

        # Keep every other source (a deterministic ~half subset).
        keep = set(sorted(all_ids)[::2])

        filtered = read_catalog_file(catalog_fname, source_id_filter=keep)
        self.assertEqual(
            {str(source_id) for source_id in filtered.index},
            keep,
            "Filtered catalog does not contain exactly the requested IDs.",
        )

        # IDs absent from the catalog are silently ignored.
        with_bogus = keep | {"1", "999999999999999999"}
        filtered_bogus = read_catalog_file(
            catalog_fname, source_id_filter=with_bogus
        )
        self.assertEqual(
            {str(source_id) for source_id in filtered_bogus.index},
            keep,
            "Source IDs not present in the catalog should be ignored.",
        )

        # An empty filter keeps nothing.
        self.assertEqual(
            len(read_catalog_file(catalog_fname, source_id_filter=set())),
            0,
            "An empty source_id_filter should return no rows.",
        )

    def test_read_source_id_list(self):
        """``read_source_id_list`` reads one ID per line, ignoring blanks."""

        list_fname = path.join(self.processing_directory, "ids.txt")
        with open(list_fname, "w", encoding="utf-8") as id_file:
            id_file.write("808860337475327872\n\n  809305880203284608  \n")
        self.assertEqual(
            read_source_id_list(list_fname),
            {"808860337475327872", "809305880203284608"},
        )


class TestLCFilter(H5TestCase):
    """End-to-end subset tests for restricted light-curve creation."""

    _epd_groups = [
        f"AperturePhotometry/Aperture{ap_ind:03d}/EPD" for ap_ind in range(4)
    ]

    @staticmethod
    def _is_postprocessing(lc_path):
        """True for EPD/TFA groups (excluded from the create-LC compare)."""

        return (
            "/EPD/" in lc_path
            or "/TFA/" in lc_path
            or lc_path.endswith("/TFA")
            or lc_path.endswith("/EPD")
        )

    @staticmethod
    def _epd_ignore(name):
        """Ignore the per-run EPD filter list when comparing EPD groups."""

        return name.endswith("/EPD/FitProperties/Filter")

    def _lc_ids(self, lc_dir):
        """Return the set of GAIA source IDs (str) with an LC in ``lc_dir``."""

        return {
            path.basename(fname)[len("GDR3_") : -len(".h5")]
            for fname in glob(path.join(lc_dir, "*.h5"))
        }

    def _full_lc_ids(self):
        """The source IDs that got a light curve in the full-catalog run."""

        return self._lc_ids(path.join(self.test_directory, "LC"))

    def _compare_lc(self, source_id, groups, ignore):
        """Assert generated LC for ``source_id`` matches the full fixture."""

        generated = path.join(
            self.processing_directory, "LC", f"GDR3_{source_id}.h5"
        )
        expected = path.join(self.test_directory, "LC", f"GDR3_{source_id}.h5")
        for group in groups:
            self.assert_groups_match(generated, expected, group, ignore)
            self.assert_groups_match(expected, generated, group, ignore)

    def _run_restricted_lc(self, kept_ids, extra_args):
        """Run create_lightcurves -> epd -> stats and check the kept subset.

        Args:
            kept_ids(set of str):    The source IDs expected to get a light
                curve under the restriction.

            extra_args(list):    Extra CLI arguments for wisp-create-lightcurves
                that impose the restriction.
        """

        kept_ids = set(kept_ids)
        full_ids = self._full_lc_ids()
        self.assertTrue(
            kept_ids < full_ids,
            "Test misconfigured: kept IDs are not a strict subset of the "
            "full-run light-curve sources.",
        )

        # 1. create_lightcurves with the restriction imposed. DR files are
        # staged by the caller (before any catalog staging that needs them).
        self.run_step(
            ["wisp-create-lightcurves", "-c", "test.cfg"]
            + extra_args
            + [path.join(self.processing_directory, "DR")]
        )
        generated_ids = self._lc_ids(path.join(self.processing_directory, "LC"))
        self.assertEqual(
            generated_ids,
            kept_ids,
            "create_lightcurves produced the wrong set of light curves.",
        )
        for source_id in kept_ids:
            self._compare_lc(source_id, ["/"], self._is_postprocessing)

        # 2. epd on the restricted set of light curves.
        self.run_step(
            [
                "wisp-epd",
                "-c",
                "test.cfg",
                path.join(self.processing_directory, "LC"),
            ]
        )
        for source_id in kept_ids:
            self._compare_lc(source_id, self._epd_groups, self._epd_ignore)

        # 3. EPD statistics -- must be the kept subset of the full statistics.
        self.run_step(["wisp-generate-epd-statistics", "-c", "test.cfg", "LC"])
        generated_stats, expected_stats = (
            pandas.read_csv(
                path.join(dirname, "MASTERS", "epd_statistics.txt"),
                sep=r"\s+",
                index_col="ID",
            ).sort_index()
            for dirname in [self.processing_directory, self.test_directory]
        )
        kept_int = sorted(int(source_id) for source_id in kept_ids)
        self.assertEqual(
            set(generated_stats.index),
            set(kept_int),
            "EPD statistics cover the wrong set of sources.",
        )
        self.assertApproxPandas(
            expected_stats.loc[kept_int],
            generated_stats.loc[kept_int],
            "EPD statistics subset",
        )

    def test_manual_source_list(self):
        """A manual source list keeps only the listed sources."""

        # All fixture GAIA IDs are multiples of 128 (they were rounded through
        # float64, whose 52-bit mantissa cannot hold a ~2^59 GAIA ID), so a
        # literal even/odd split keeps everything. Use a deterministic half
        # instead: every other source by sorted ID.
        self.get_inputs(["DR"])
        full_ids = sorted(self._full_lc_ids())
        kept_ids = set(full_ids[::2])

        list_fname = path.join(self.processing_directory, "keep_sources.txt")
        with open(list_fname, "w", encoding="utf-8") as id_file:
            id_file.write("\n".join(sorted(kept_ids)) + "\n")

        self._run_restricted_lc(
            kept_ids, ["--lc-catalog-source-list", list_fname]
        )

    def _astrometried_dr_files(self):
        """DR files that carry a Version000 sky-to-frame solution.

        ``get_catalog_info`` reads the astrometry of every DR passed to it to
        size the catalog; the outlier frame without a solution must be excluded,
        exactly as the pipeline's DR selection does.
        """

        result = []
        for dr_fname in sorted(
            glob(path.join(self.processing_directory, "DR", "*.h5"))
        ):
            with h5py.File(dr_fname, "r") as dr_file:
                if "SkyToFrameTransformation/Version000" in dr_file:
                    result.append(dr_fname)
        return result

    def _stage_mag_limited_lc_catalog(self, mag_limit):
        """Stage a magnitude-limited LC Gaia catalog offline; return kept IDs.

        The LC catalog cache filename is a checksum of the query (including the
        magnitude limit), so a brighter limit resolves to a name absent from the
        staged cache and would trigger a live query. Compute that name via
        ``get_catalog_info`` (as ``collect_light_curves`` does) and write a
        magnitude-subset of the staged mag<=12 catalog there.

        Returns the source IDs (str) expected to get a light curve: the full-run
        LC sources with ``phot_g_mean_mag <= mag_limit``.
        """

        def catalog_fname(magnitude_limit):
            config = create_lightcurves.parse_command_line(
                [
                    "-c",
                    "test.cfg",
                    "--lc-catalog-max-magnitude",
                    str(magnitude_limit),
                ]
            )
            with DataReductionFile(
                config["single_photref_dr_fname"], "r"
            ) as sphotref:
                header = sphotref.get_frame_header()
            return get_catalog_info(
                dr_files=self._astrometried_dr_files(),
                header=header,
                configuration=get_catalog_config(config, "lc"),
                skytoframe_version=config["skytoframe_version"],
            )[0]["fname"]

        old_cwd = os.getcwd()
        os.chdir(self.processing_directory)
        try:
            full_catalog = Table.read(catalog_fname(12.0))
            limited = full_catalog[full_catalog["phot_g_mean_mag"] <= mag_limit]
            limited.meta["MAGMAX"] = mag_limit
            limited.write(
                catalog_fname(mag_limit), format="fits", overwrite=True
            )
            catalog_ids = {str(int(sid)) for sid in limited["source_id"]}
        finally:
            os.chdir(old_cwd)

        return self._full_lc_ids() & catalog_ids

    def test_bright_mag_limit(self):
        """A brighter LC catalog magnitude limit keeps only bright sources."""

        self.get_inputs(["DR"])
        kept_ids = self._stage_mag_limited_lc_catalog(11.0)
        self._run_restricted_lc(
            kept_ids, ["--lc-catalog-max-magnitude", "11.0"]
        )

    def test_combined(self):
        """A magnitude limit and a manual source list combine (intersection)."""

        self.get_inputs(["DR"])
        bright_ids = self._stage_mag_limited_lc_catalog(11.0)

        listed = set(sorted(self._full_lc_ids())[::2])
        kept_ids = bright_ids & listed

        list_fname = path.join(self.processing_directory, "keep_sources.txt")
        with open(list_fname, "w", encoding="utf-8") as id_file:
            id_file.write("\n".join(sorted(listed)) + "\n")

        self._run_restricted_lc(
            kept_ids,
            [
                "--lc-catalog-max-magnitude",
                "11.0",
                "--lc-catalog-source-list",
                list_fname,
            ],
        )
