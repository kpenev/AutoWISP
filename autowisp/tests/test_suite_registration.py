"""Check that the suite runs every test the package defines.

``python -m autowisp.tests`` collects from the names imported into
``autowisp.tests.__main__``, so a test class nobody imports there is simply
never run.  Nothing reports that: the suite passes, faster than it should,
and the tests that were supposed to be guarding something are not.  Running
one module directly -- ``python -m unittest autowisp.tests.test_x`` -- hides
it too, since that bypasses ``__main__`` entirely and the tests do run.

Both ways of getting it wrong have happened: a class added and never
registered, a class moved between modules and left importing from the old
one (which at least fails loudly), and two classes of the same name in
different modules, where importing both puts one in the namespace and
silently drops the other.

So this compares what the runner reaches with what the modules define,
which is the one check that catches all three.
"""

import importlib
import pathlib
import unittest


def collect_ids(suite):
    """
    Return ``{module.Class.method}`` for every test in *suite*.

    Qualified by module because two modules may define classes of the same
    name, which is one of the ways a test goes missing: the bare name would
    make the two look like one and hide exactly that case.
    """

    found = set()
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            found |= collect_ids(item)
        else:
            case = type(item)
            found.add(
                f"{case.__module__}.{case.__name__}."
                # pylint: disable=protected-access
                f"{item._testMethodName}"
                # pylint: enable=protected-access
            )

    return found


class TestSuiteRegistration(unittest.TestCase):
    """The runner reaches every test in the package."""

    def test_every_test_is_registered(self):
        """Nothing defined in a test module is missing from ``__main__``."""

        loader = unittest.TestLoader()
        here = pathlib.Path(__file__).parent

        defined = set()
        for module_file in sorted(here.glob("test_*.py")):
            defined |= collect_ids(
                loader.loadTestsFromModule(
                    importlib.import_module(
                        f"autowisp.tests.{module_file.stem}"
                    )
                )
            )

        # By dotted name rather than ``sys.modules["__main__"]``, so that
        # this looks at the runner whichever way the suite was started --
        # ``python -m autowisp.tests``, pytest, or one module on its own.
        reachable = collect_ids(
            loader.loadTestsFromModule(
                importlib.import_module("autowisp.tests.__main__")
            )
        )

        missing = sorted(defined - reachable)
        self.assertEqual(
            missing,
            [],
            "these tests exist but the suite never runs them; import each "
            "into autowisp/tests/__main__.py, and where the name is already "
            "taken there by another module's class, rename one of the two "
            f"rather than letting the import shadow it: {missing}",
        )
