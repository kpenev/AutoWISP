"""Test cases for master stacking."""

from os import path
from subprocess import run, PIPE, STDOUT

from autowisp.tests import autowisp_dir, FITSTestCase


class TestStackToMaster(FITSTestCase):
    """Test cases for master stacking steps."""

    def _test_stack_to_master(self, master_type):
        """Perform a stacking step and test outputs match expectations."""

        input_dir = path.join(self.test_directory, "CAL", master_type)
        command = [
            "python3",
            path.join(
                autowisp_dir,
                "processing_steps",
                (
                    "stack_to_master"
                    + ("_flat" if master_type == "flat" else "")
                    + ".py"
                ),
            ),
            "-c",
            path.join(self.processing_directory, "test.cfg"),
            input_dir,
        ]
        stack_process = run(
            command,
            cwd=self.processing_directory,
            check=False,
            stdout=PIPE,
            stderr=STDOUT,
        )
        self.assertTrue(
            stack_process.returncode == 0,
            f"Master stacking command:\n{command!r} "
            f"failed:\n{stack_process.stdout.decode('utf-8')}",
        )
        self.assert_fits_match(
            path.join(
                self.test_directory, "MASTERS", master_type + "_R.fits.fz"
            ),
            path.join(
                self.processing_directory, "MASTERS", master_type + "_R.fits.fz"
            ),
        )

    def test_stack_master_bias(self):
        """Check if creating master bias works as expected."""

        self._test_stack_to_master("zero")

    def test_stack_master_dark(self):
        """Check if creating master bias works as expected."""

        self._test_stack_to_master("dark")

    def test_stack_master_flat(self):
        """Check if creating master bias works as expected."""

        self._test_stack_to_master("flat")
