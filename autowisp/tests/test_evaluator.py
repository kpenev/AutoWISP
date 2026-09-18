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


class TestIterativeRejection(unittest.TestCase):
    """The iterative-rejection fits cope with NaN-padded input.

    Diagnostics are NaN-padded onto the full image list, so a missing value
    is the usual case rather than an exception. A NaN in the dependent
    variable is left out of the fit but still gets a prediction; a NaN in
    the independent one has nothing to predict at.
    """

    def setUp(self):
        """A straight line with one outlier and some missing values."""

        self.x = numpy.arange(20.0)
        self.expected = 2.0 * self.x + 1.0
        self.y = self.expected.copy()
        self.y[4] = 100.0
        self.missing_y = [7, 12]
        self.y[self.missing_y] = numpy.nan
        self.missing_x = 15
        self.x[self.missing_x] = numpy.nan
        self.expected[self.missing_x] = numpy.nan

    def test_average_ignores_nan_and_rejects_outlier(self):
        """With ``nanmean``, only rejection keeps the outlier out."""

        y = numpy.full(20, 10.0)
        y[3] = 100.0
        y[[5, 11]] = numpy.nan
        self.assertEqual(
            Evaluator({"y": y})(
                "iterative_rejection_average(y, 3, average_func=nanmean)"
            ),
            10.0,
        )

    def test_polynomial_fit(self):
        """Predicted where only y is missing, NaN where x is."""

        numpy.testing.assert_allclose(
            Evaluator({"x": self.x, "y": self.y})(
                "iterative_rej_polynomial_fit(x, y, 1, 3)"
            ),
            self.expected,
        )

    def test_threshold_pair(self):
        """A signed pair rejects above and below separately, in any order.

        The outlier above the line (at 4) is rejected; the one below it (at
        9) is well within the negative threshold, so it stays and pulls the
        fit. Two thresholds of the same sign are refused rather than
        guessed at.
        """

        y = self.y.copy()
        y[9] -= 60.0
        keep = numpy.isfinite(self.x) & numpy.isfinite(y)
        keep[4] = False
        expected = numpy.polynomial.polynomial.polyval(
            self.x,
            numpy.polynomial.polynomial.polyfit(self.x[keep], y[keep], 1),
        )
        evaluator = Evaluator({"x": self.x, "y": y})
        for threshold in ("(3, -10)", "(-10, 3)"):
            with self.subTest(threshold=threshold):
                numpy.testing.assert_allclose(
                    evaluator(
                        f"iterative_rej_polynomial_fit(x, y, 1, {threshold})"
                    ),
                    expected,
                )
        with self.assertRaises(ValueError):
            evaluator("iterative_rej_polynomial_fit(x, y, 1, (3, 10))")

    def test_polynomial_fit_with_too_few_points_is_nan(self):
        """Fewer finite points than coefficients gives no fit at all.

        Rather than the minimum-norm solution ``lstsq`` would return, which
        would plot as a line through nothing.
        """

        for num_finite in (0, 1):
            with self.subTest(num_finite=num_finite):
                y = numpy.full(self.x.shape, numpy.nan)
                y[:num_finite] = 1.0
                self.assertTrue(
                    numpy.isnan(
                        Evaluator({"x": self.x, "y": y})(
                            "iterative_rej_polynomial_fit(x, y, 1, 3)"
                        )
                    ).all()
                )

    def test_smoothing_spline(self):
        """Predicted where only y is missing, NaN where x is."""

        y = self.expected.copy()
        y[self.missing_y] = numpy.nan
        numpy.testing.assert_allclose(
            Evaluator({"x": self.x, "y": y})(
                "iterative_rej_smoothing_spline(x, y, 5, s=1e-6)"
            ),
            self.expected,
            atol=1e-3,
        )

    def _noisy_sine(self):
        """Return x, y, and the truth: noisy with outliers and NaNs.

        The noise (0.05) comes from ``RandomState`` because its stream is
        frozen, while a ``Generator``'s may change between numpy versions.
        The outliers are 200 times the noise so that failing to reject them
        is unmistakable.
        """

        x = numpy.linspace(0.0, 10.0, 101)
        truth = numpy.sin(x)
        y = truth + numpy.random.RandomState(0).normal(scale=0.05, size=101)
        y[[20, 40, 60]] += 10.0
        y[[5, 70]] = numpy.nan
        return x, y, truth

    def test_smoothing_spline_rejects_outliers(self):
        """The result does not depend on how loose the initial ``s`` is.

        It only has to be loose enough for the spline not to bend to the
        outliers. Over 200 noise seeds, the largest error is at most 0.12,
        against at least 1.0 without rejection (see the next test).
        """

        x, y, truth = self._noisy_sine()
        for initial_smoothing in (300.0, 3000.0):
            with self.subTest(initial_smoothing=initial_smoothing):
                smoothed = Evaluator({"x": x, "y": y})(
                    "iterative_rej_smoothing_spline(x, y, 5, "
                    f"s={initial_smoothing})"
                )
                self.assertLess(numpy.abs(smoothed - truth).max(), 0.15)

    def test_smoothing_spline_without_iterations(self):
        """``max_iterations=0`` is a single fit, rejecting nothing.

        Smoothing enough to leave the outliers alone flattens the sine.
        """

        x, y, truth = self._noisy_sine()
        smoothed = Evaluator({"x": x, "y": y})(
            "iterative_rej_smoothing_spline(x, y, 5, max_iterations=0, "
            "s=300.0)"
        )
        self.assertGreater(numpy.abs(smoothed - truth).max(), 0.5)


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
