"""Define class to compare groups in DR files."""

from os import path
from glob import glob
from shutil import copytree, rmtree

import numpy
import h5py


from autowisp.tests import steps_dir
from autowisp.tests import AutoWISPTestCase


class DRTestCase(AutoWISPTestCase):
    """Add assert for comparing groups in DR files."""

    def assert_groups_match(self, dr_fname1, dr_fname2, group_name):
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
                    numpy.array_equal(dset1[:], dset2[:], equal_nan=True),
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
                    else:
                        self.assertEqual(obj2.attrs[key], value, msg)

                if isinstance(obj1, h5py.Dataset):
                    self.assertTrue(
                        isinstance(obj2, h5py.Dataset),
                        f"Object {dr_fname2!r}/{obj2.name!r} is not a dataset!",
                    )
                    assert_dset_match(obj1, obj2)

            dr1[group_name].visititems(assert_obj_match)

    def run_step_test(self, step_name, input_type, compare_groups):
        """Run a test of a single step that updates the DR files."""

        if input_type == "CAL":
            input_dir = path.join(self.test_directory, input_type, "object")
        else:
            assert input_type == "DR"
            input_dir = path.join(self.processing_directory, "DR")
        copytree(
            path.join(self.test_directory, "DR"),
            path.join(self.processing_directory, "DR"),
        )
        for fname in glob(path.join(self.processing_directory, "DR", "*.h5")):
            with h5py.File(fname, "a") as dr_file:
                for group in compare_groups:
                    if group in dr_file:
                        del dr_file[group]

        self.run_calib_step(
            [
                "python3",
                path.join(steps_dir, step_name + ".py"),
                "-c",
                "test.cfg",
                input_dir,
            ]
        )

        generated = sorted(
            glob(path.join(self.processing_directory, "DR", "*.h5"))
        )
        expected = sorted(glob(path.join(self.test_directory, "DR", "*.h5")))
        self.assertTrue(
            [path.basename(fname) for fname in generated]
            == [path.basename(fname) for fname in expected],
            "Generated files do not match expected files!",
        )
        for gen_fname, exp_fname in zip(generated, expected):
            for group in compare_groups:
                self.assert_groups_match(gen_fname, exp_fname, group)
        rmtree(path.join(self.processing_directory, "DR"))
