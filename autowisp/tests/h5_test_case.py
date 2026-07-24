"""Define class to compare groups in DR files."""

from os import path
from glob import glob

import numpy
import h5py
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from autowisp.data_reduction.data_reduction_file import DataReductionFile


from autowisp.tests import AutoWISPTestCase


class H5TestCase(AutoWISPTestCase):
    """Add assert for comparing groups in HDF5 files."""

    # Datasets (keyed by leaf name) whose stored value has a large, arbitrary
    # zero-point, so a relative tolerance is meaningless: rtol=1e-8 on a
    # ~2.4e6 BJD is a ~2000 s absolute slop, which silently hides real timing
    # errors. Compare these with a purely absolute tolerance (in the dataset's
    # own units) instead. BJD is in days, so 1e-6 day (~0.086 s) catches real
    # timing errors while tolerating cross-platform numerical noise.
    _absolute_tolerance_datasets = {"BJD": 1e-6}

    def _project_relative(self, dr_fname, value):
        """Return ``value`` as a path relative to whichever project_home
        ``dr_fname`` lives in (``test_directory`` or ``processing_directory``).

        ``value`` may be ``bytes`` (h5py string attrs) or ``str``; the
        result is returned as ``str``. If ``dr_fname`` does not lie
        under either project_home the value is returned unchanged.
        """

        if isinstance(value, bytes):
            value = value.decode()
        if not path.isabs(value):
            return value
        dr_norm = path.normpath(dr_fname)
        # ``processing_directory`` is nested inside ``test_directory``, so
        # match the more specific one first.
        for home in (self.processing_directory, self.test_directory):
            home_norm = path.normpath(home)
            if dr_norm == home_norm or dr_norm.startswith(home_norm + path.sep):
                return path.relpath(path.normpath(value), home_norm)
        return value

    def assert_groups_match(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self, dr_fname1, dr_fname2, group_name, ignore, reorder=None
    ):
        """Check if two DR files have the same groups.

        Args:
            reorder(numpy.ndarray or None):    Optional integer permutation
                applied to dr_fname2 datasets along axis 0 before comparison.
                Only applied to datasets whose leading dimension equals
                ``len(reorder)``.
        """

        def reordered(dset2):
            """Return dset2 data with ``reorder`` applied along axis 0."""

            data = dset2[:]
            if (
                reorder is not None
                and data.ndim >= 1
                and data.shape[0] == len(reorder)
            ):
                return data[reorder]
            return data

        def assert_dset_match(dset1, dset2):
            """Assert that the two datasets contain the same data."""

            self.assertEqual(
                dset1.shape,
                dset2.shape,
                f"Datasets {dr_fname1!r}/{dset1.name!r} and "
                f"{dr_fname2!r}/{dset2.name!r} have different shapes.",
            )
            data1 = dset1[:]
            data2 = reordered(dset2)
            if dset1.name.endswith(
                "/MagnitudeFitting/Configuration/SinglePhotometricReference"
            ):
                data1 = numpy.array(
                    [self._project_relative(dr_fname1, v) for v in data1]
                )
                data2 = numpy.array(
                    [self._project_relative(dr_fname2, v) for v in data2]
                )
            if dset1.dtype.kind == "f":
                abs_atol = self._absolute_tolerance_datasets.get(
                    dset1.name.rsplit("/", 1)[-1]
                )
                rtol, atol = (
                    (0.0, abs_atol) if abs_atol is not None else (1e-8, 1e-8)
                )
                differ = numpy.logical_not(
                    numpy.isclose(
                        data1, data2, rtol=rtol, atol=atol, equal_nan=True
                    )
                )
                if differ.any():
                    self.fail(
                        f"Data in datasets {dr_fname1!r}/{dset1.name!r} and "
                        f"{dr_fname2!r}/{dset2.name!r} do not match (different "
                        f"elements: {numpy.nonzero(differ)}."
                        "\n\tMax abs difference: "
                        + str(numpy.abs(data1 - data2).max())
                        + "\n\tMax rel difference: "
                        + str(
                            numpy.abs(
                                (data1 - data2) / numpy.maximum(data1, data2)
                            ).max()
                        )
                        + f"\n{dr_fname1!r}/{dset1.name!r}"
                        + f"\n\t{data1[differ]}"
                        + f"\n\t{data2[differ]}"
                        + f"\n\tdiff: {data1[differ] - data2[differ]}\n\t"
                    )
            elif dset1.dtype.kind == "O":
                # Variable-length (ragged) rows -- each entry is itself an
                # array (e.g. TFA TemplateStarIDs). Compare row by row.
                mismatched = [
                    index
                    for index in range(len(data1))
                    if not numpy.array_equal(data1[index], data2[index])
                ]
                if mismatched:
                    self.fail(
                        f"Data in datasets {dr_fname1!r}/{dset1.name!r} and "
                        f"{dr_fname2!r}/{dset2.name!r} do not match "
                        f"(different rows: {mismatched})."
                        f"\n{dr_fname1!r}/{dset1.name!r}"
                        f"\n\t{[data1[index] for index in mismatched]}"
                        f"\n\t{[data2[index] for index in mismatched]}"
                    )
            else:
                differ = data1 != data2
                if numpy.any(differ):
                    self.fail(
                        f"Data in datasets {dr_fname1!r}/{dset1.name!r} and "
                        f"{dr_fname2!r}/{dset2.name!r} do not match "
                        f"(different elements: {numpy.nonzero(differ)})."
                        f"\n{dr_fname1!r}/{dset1.name!r}"
                        f"\n\t{data1[differ]}"
                        f"\n\t{data2[differ]}"
                    )

        with h5py.File(dr_fname1, "r") as dr1, h5py.File(dr_fname2, "r") as dr2:
            if group_name not in dr1:
                self.assertTrue(
                    group_name not in dr2,
                    f"Group {group_name!r} not found in {dr_fname1}.",
                )
                return
            self.assertTrue(
                group_name in dr2,
                f"Group {group_name!r} not found in {dr_fname2}.",
            )

            def assert_obj_match(_, obj1):
                """Assert the two datasets or groups contain the same data."""

                if ignore is not None and ignore(obj1.name):
                    return
                obj2 = dr2[obj1.name]
                self.assertEqual(
                    set(obj1.attrs.keys()),
                    set(obj2.attrs.keys()),
                    f"Attributes in {dr_fname1!r}/{obj1.name!r} and "
                    f"{dr_fname2!r}/{obj2.name!r} do not match.",
                )
                for key, value in obj1.attrs.items():
                    other = obj2.attrs[key]
                    if key == "SinglePhotometricReference":
                        value = self._project_relative(dr_fname1, value)
                        other = self._project_relative(dr_fname2, other)
                    msg = (
                        f"Attribute {dr_fname1!r}/{obj1.name!r}.{key} does "
                        f"not match {dr_fname2!r}/{obj1.name!r}.{key}: "
                        f"{value!r} vs {other!r}."
                    )
                    if numpy.atleast_1d(value).dtype.kind == "f":
                        self.assertTrue(
                            numpy.allclose(
                                other,
                                value,
                                rtol=1e-8,
                                atol=1e-8,
                                equal_nan=True,
                            ),
                            msg,
                        )
                    elif numpy.atleast_1d(value).size > 1:
                        self.assertTrue(numpy.array_equal(other, value), msg)
                    else:
                        self.assertEqual(other, value, msg)

                if isinstance(obj1, h5py.Dataset):
                    self.assertTrue(
                        isinstance(obj2, h5py.Dataset),
                        f"Object {dr_fname2!r}/{obj2.name!r} is not a dataset!",
                    )
                    if obj1.name == "/FITSHeader":
                        with DataReductionFile(
                            dr_fname1, "r"
                        ) as dr1_file, DataReductionFile(
                            dr_fname2, "r"
                        ) as dr2_file:
                            self._compare_headers(
                                dr_fname1,
                                dr_fname2,
                                dr1_file.get_frame_header(),
                                dr2_file.get_frame_header(),
                            )
                    elif not obj1.name.endswith("/MaxSources"):
                        assert_dset_match(obj1, obj2)

            if isinstance(dr1[group_name], h5py.Dataset):
                assert_obj_match(None, dr1[group_name])
            else:
                dr1[group_name].visititems(assert_obj_match)

    # pylint: disable=too-many-arguments
    def run_step_test(
        self, step_name, inputs, compare, *, ignore=None, output_type="DR"
    ):
        """
        Run a test of a single step that updates the DR files.

        Args:
            step_name(str):    The name of the step being tested

            inputs([]):    List of the directories or files needed by the step.
                The first entry (with full path added) is passed as input to the
                step.

            compare([]):    List of the HDF5 groups to compare in order to
                ensure the step produced correct results,

            ignore(callable):    Function that returns true on any dataset or
                group in the HDF5 file that should not be compared when it is
                under the groups specified in ``compare``

            tput_type(str):    The type of output files produced by the step
                (i.e. whic files should be compared),
        """

        if isinstance(inputs, str):
            inputs = [inputs]
        self.get_inputs(inputs)
        for fname in glob(
            path.join(self.processing_directory, output_type, "*.h5")
        ):
            with h5py.File(fname, "a") as h5_file:
                for group in compare:
                    if group in h5_file:
                        del h5_file[group]

        self.run_step(
            [
                f"wisp-{step_name.replace('_', '-')}",
                "-c",
                "test.cfg",
                path.join(self.processing_directory, inputs[0]),
            ]
        )

        generated = sorted(
            glob(path.join(self.processing_directory, output_type, "*.h5"))
        )
        expected = sorted(
            glob(path.join(self.test_directory, output_type, "*.h5"))
        )
        self.assertTrue(
            [path.basename(fname) for fname in generated]
            == [path.basename(fname) for fname in expected],
            "Generated files do not match expected files!",
        )
        for gen_fname, exp_fname in zip(generated, expected):
            for group in compare:
                self.assert_groups_match(gen_fname, exp_fname, group, ignore)
                self.assert_groups_match(exp_fname, gen_fname, group, ignore)

        self.successful_test = True

    # pylint: enable=too-many-arguments


class DRTestCase(H5TestCase):
    """H5TestCase aware of row-order ambiguity in SourceExtraction groups."""

    _srcext_match_columns = ("x", "y", "flux")

    def _find_sources_group(self, root, group_name):
        """Return path of subgroup of ``group_name`` holding source columns."""

        cols = self._srcext_match_columns
        sources_path = [None]

        def visit(_, obj):
            if isinstance(obj, h5py.Group) and all(c in obj for c in cols):
                sources_path[0] = obj.name
                return True
            return None

        node = root[group_name]
        if isinstance(node, h5py.Group):
            if all(c in node for c in cols):
                return node.name
            node.visititems(visit)
        return sources_path[0]

    def _build_srcextract_reorder(self, dr_fname1, dr_fname2, group_name):
        """Return permutation aligning dr_fname2 sources with dr_fname1."""

        with h5py.File(dr_fname1, "r") as dr1, h5py.File(dr_fname2, "r") as dr2:
            sources_path = self._find_sources_group(dr1, group_name)
            if sources_path is None or sources_path not in dr2:
                return None
            cols1 = numpy.column_stack(
                [
                    dr1[sources_path][col][:]
                    for col in self._srcext_match_columns
                ]
            )
            cols2 = numpy.column_stack(
                [
                    dr2[sources_path][col][:]
                    for col in self._srcext_match_columns
                ]
            )
        if cols1.shape != cols2.shape:
            return None
        scale = numpy.std(numpy.concatenate([cols1, cols2], axis=0), axis=0)
        scale[scale == 0] = 1.0
        cost = cdist(cols1 / scale, cols2 / scale)
        row_ind, col_ind = linear_sum_assignment(cost)
        perm = numpy.empty(cols1.shape[0], dtype=numpy.intp)
        perm[row_ind] = col_ind
        return perm

    def assert_groups_match(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self, dr_fname1, dr_fname2, group_name, ignore, reorder=None
    ):
        """Compute SourceExtraction reorder, then delegate to H5TestCase."""

        # is_srcextract = group_name.strip("/").startswith("SourceExtraction")
        # if reorder is None and is_srcextract:
        #    reorder = self._build_srcextract_reorder(
        #        dr_fname1, dr_fname2, group_name
        #    )

        # if is_srcextract:
        #    user_ignore = ignore

        #    def ignore(name):
        #        if name.rsplit("/", 1)[-1] == "id":
        #            return True
        #        return user_ignore is not None and user_ignore(name)

        super().assert_groups_match(
            dr_fname1, dr_fname2, group_name, ignore, reorder=reorder
        )
