"""Test cases for master stacking."""

from os import path, remove

from autowisp.tests.fits_test_case import FITSTestCase


class TestStackToMaster(FITSTestCase):
    """Test cases for master stacking steps."""

    def _test_stack_to_master(self, master_type):
        """Perform a stacking step and test outputs match expectations."""

        input_dir = path.join(self.test_directory, "CAL", master_type)
        self.run_step(
            [
                "wisp-stack-to-master"
                + ("-flat" if master_type == "flat" else ""),
                "-c",
                path.join(self.processing_directory, "test.cfg"),
                input_dir,
            ]
        )
        generated_master = path.join(
            self.processing_directory, "MASTERS", master_type + "_R.fits.fz"
        )
        self.assert_fits_match(
            path.join(
                self.test_directory, "MASTERS", master_type + "_R.fits.fz"
            ),
            generated_master,
        )
        remove(generated_master)
        self.successful_test = True

    def test_stack_master_bias(self):
        """Check if creating master bias works as expected."""

        self._test_stack_to_master("zero")

    def test_stack_master_dark(self):
        """Check if creating master bias works as expected."""

        self._test_stack_to_master("dark")

    def test_stack_master_flat(self):
        """Check if creating master bias works as expected."""

        self._test_stack_to_master("flat")
