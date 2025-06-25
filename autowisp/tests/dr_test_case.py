"""Define class to compare groups in DR files."""

import numpy
import h5py

from autowisp.tests import AutoWISPTestCase
from autowisp import DataReductionFile


class DRTestCase(AutoWISPTestCase):
    """Add assert for comparing groups in DR files."""

    def assert_groups_match(self, dr_fname1, dr_fname2, group_name):
        """Check if two DR files have the same groups."""

        def assert_dset_match(dset1, dset2):
            """Assert that the two datasets contain the same data."""

            print(
                f"Comparing  {dr_fname1!r}/{dset1.name!r} to "
                f"{dr_fname1!r}/{dset2.name!r}"
            )
            if dset1.dtype.kind == "f":
                self.assertTrue(
                    numpy.allclose(dset1[:], dset2[:], rtol=1e-8, atol=1e-8),
                    f"Data in datasets {dr_fname1!r}/{dset1.name!r} and "
                    "{dr_fname2!r}/{dset2.name!r} do not match.",
                )
            else:
                self.assertTrue(
                    numpy.array_equal(dset1[:], dset2[:], equal_nan=True),
                    f"Data in datasets {dr_fname1!r}/{dset1.name!r} and "
                    "{dr_fname2!r}/{dset2.name!r} do not match.",
                )

        with DataReductionFile(dr_fname1, "r") as dr1, DataReductionFile(
            dr_fname2, "r"
        ) as dr2:
            self.assertTrue(
                group_name in dr1,
                f"Group {group_name!r} not found in {dr_fname1}.",
            )
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
                    self.assertTrue(
                        obj2.attrs[key] == value
                        or (
                            numpy.isnan(value) and numpy.isnan(obj2.attrs[key])
                        ),
                        f"Attribute {dr_fname1!r}/{obj1.name!r}.{key} does not "
                        f"match {dr_fname1!r}/{obj1.name!r}.{key}.",
                    )

                if isinstance(obj1, h5py.Dataset):
                    self.assertTrue(
                        isinstance(obj2, h5py.Dataset),
                        f"Object {dr_fname2!r}/{obj2.name!r} is not a dataset!",
                    )
                    assert_dset_match(obj1, obj2)

            dr1[group_name].visititems(assert_obj_match)
