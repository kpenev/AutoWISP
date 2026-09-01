"""Tests for the symbol table every AutoWISP expression is evaluated in.

``Evaluator`` and ``LightCurveEvaluator`` are siblings deriving from
``EvaluatorBase``, and roughly two dozen modules build one. What they offer
is therefore worth pinning in one place: the names AutoWISP adds, the names
it takes away, and what happens to a bad expression.
"""

import unittest
from types import SimpleNamespace

import numpy

from autowisp.evaluator import (
    Evaluator,
    EvaluatorBase,
    LightCurveEvaluator,
)


def _lightcurve_evaluator():
    """Return a ``LightCurveEvaluator`` over a lightcurve with no datasets.

    Only ``elements["dataset"]`` is consulted during construction, so a
    stand-in avoids needing a lightcurve file to check the symbol table.
    """

    return LightCurveEvaluator(SimpleNamespace(elements={"dataset": []}))


class TestNanAggregates(unittest.TestCase):
    """The NaN-ignoring aggregates are available and are numpy's."""

    def test_present_in_both_evaluators(self):
        """Checked in both, since they used to differ.

        ``LightCurveEvaluator`` defined two of these and ``Evaluator`` none,
        so which aggregates worked depended on which one you were in.
        """

        for evaluator in (Evaluator(), _lightcurve_evaluator()):
            for name in EvaluatorBase.nan_aggregates:
                with self.subTest(
                    evaluator=type(evaluator).__name__, name=name
                ):
                    self.assertIn(name, evaluator.symtable)

    def test_bound_to_the_numpy_functions(self):
        """A name resolving to something else would be worse than absent."""

        symtable = Evaluator().symtable
        for name in EvaluatorBase.nan_aggregates:
            with self.subTest(name=name):
                self.assertIs(symtable[name], getattr(numpy, name))

    def test_a_missing_numpy_name_would_fail_loudly(self):
        """The list is explicit so that numpy dropping one is an error."""

        for name in EvaluatorBase.nan_aggregates:
            with self.subTest(name=name):
                self.assertTrue(hasattr(numpy, name))

    def test_data_shadows_an_aggregate_of_the_same_name(self):
        """The data is what the user is asking about.

        Binding happens before the data is loaded precisely so that this
        works; the reverse order would make a diagnostic unreachable.
        """

        self.assertEqual(Evaluator({"nanmedian": 42}).symtable["nanmedian"], 42)


class TestRemovedNames(unittest.TestCase):
    """Names asteval offers that AutoWISP takes away."""

    def test_absent_from_both_evaluators(self):
        """Removed on the shared base, so neither can reach them."""

        for evaluator in (Evaluator(), _lightcurve_evaluator()):
            for name in EvaluatorBase.removed_names:
                with self.subTest(
                    evaluator=type(evaluator).__name__, name=name
                ):
                    self.assertNotIn(name, evaluator.symtable)

    def test_filesystem_access_raises(self):
        """Asteval permits reading files; AutoWISP does not.

        Its ``open`` already refuses every mode but reading, so this is
        about reading: expressions travel between installations in export
        files, and a shared one must not be able to read arbitrary files.
        """

        with self.assertRaises(NameError):
            Evaluator()("open('/etc/passwd').read()")

    def test_printing_raises(self):
        """A side effect rather than a value."""

        with self.assertRaises(NameError):
            Evaluator()("print(1)")

    def test_the_mathematical_names_survive(self):
        """The removals must not cost anything anyone would write."""

        numpy.testing.assert_allclose(
            Evaluator({"x": numpy.array([1.0, 4.0])})("sqrt(x) + abs(-x)"),
            [2.0, 6.0],
        )


class TestErrorHandling(unittest.TestCase):
    """A bad expression raises rather than evaluating to ``None``."""

    def test_raises_by_default_in_both_evaluators(self):
        """Asteval's default returns ``None``, relocating the failure."""

        for evaluator in (Evaluator(), _lightcurve_evaluator()):
            with self.subTest(evaluator=type(evaluator).__name__):
                with self.assertRaises(NameError):
                    evaluator("no_such_name + 1")

    def test_permissive_behaviour_remains_available(self):
        """Callers that want the old behaviour can still ask for it."""

        self.assertIsNone(
            Evaluator()(
                "no_such_name + 1", raise_errors=False, show_errors=False
            )
        )


if __name__ == "__main__":
    unittest.main()
