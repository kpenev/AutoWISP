"""Define class to compare groups in DR files."""

from os import path
from glob import glob

import numpy
import h5py


from autowisp.tests import AutoWISPTestCase


class H5TestCase(AutoWISPTestCase):
    """Add assert for comparing groups in HDF5 files."""

    def assert_groups_match(self, dr_fname1, dr_fname2, group_name, ignore):
        """Check if two DR files have the same groups."""

        def assert_dset_match(dset1, dset2):
            """Assert that the two datasets contain the same data."""

            if dset1.dtype.kind == "f":
                differ = numpy.logical_not(
                    numpy.isclose(
                        dset1[:], dset2[:], rtol=1e-8, atol=1e-8, equal_nan=True
                    )
                )
                self.assertFalse(
                    differ.any(),
                    f"Data in datasets {dr_fname1!r}/{dset1.name!r} and "
                    f"{dr_fname2!r}/{dset2.name!r} do not match (differint ."
                    f"elements: {numpy.nonzero(differ)}",
                )
            else:
                self.assertTrue(
                    numpy.array_equal(dset1[:], dset2[:]),
                    f"Data in datasets {dr_fname1!r}/{dset1.name!r} and "
                    f"{dr_fname2!r}/{dset2.name!r} do not match.",
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
                    msg = (
                        f"Attribute {dr_fname1!r}/{obj1.name!r}.{key} does not "
                        f"match {dr_fname1!r}/{obj1.name!r}.{key}."
                    )
                    if numpy.atleast_1d(value).dtype.kind == "f":
                        self.assertTrue(
                            numpy.allclose(
                                obj2.attrs[key],
                                value,
                                rtol=1e-8,
                                atol=1e-8,
                                equal_nan=True,
                            ),
                            msg,
                        )
                    elif numpy.atleast_1d(value).size > 1:
                        self.assertTrue(
                            numpy.array_equal(obj2.attrs[key], value), msg
                        )
                    else:
                        self.assertEqual(obj2.attrs[key], value, msg)

                if isinstance(obj1, h5py.Dataset):
                    self.assertTrue(
                        isinstance(obj2, h5py.Dataset),
                        f"Object {dr_fname2!r}/{obj2.name!r} is not a dataset!",
                    )
                    if not obj1.name.endswith("/MaxSources"):
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
                '--num-parallel-processes', '1',
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
