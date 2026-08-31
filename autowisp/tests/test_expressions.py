"""Tests for the database-free half of diagnostic expressions.

Everything here runs against a library and values passed in as arguments,
so there is no database, no Django and no fixture beyond a dictionary --
which is the point of keeping this tier free of both.
"""

import unittest

import numpy

from autowisp.diagnostics.expressions import (
    check_expression,
    evaluate_expressions,
    get_bare_aggregates,
    get_expression_dependents,
    get_expression_names,
    order_expressions,
)
from autowisp.exceptions import PipelineError

#: A small library with a diamond in it: two expressions share ``rel``.
_library = {
    "rel": "astrom_residual / diagonal_fov",
    "scaled": "rel / nanmedian(rel)",
    "offset": "rel + bg_center",
}

#: What the open project is taken to record.
_known = frozenset({"astrom_residual", "diagonal_fov", "bg_center", "jd"})


class TestReferencedNames(unittest.TestCase):
    """Reading the names out of an expression."""

    def test_names_include_functions(self):
        """Splitting them apart is the caller's job, not this one's."""

        self.assertEqual(
            get_expression_names("rel / nanmedian(rel)"), {"rel", "nanmedian"}
        )

    def test_statements_are_rejected(self):
        """``mode="eval"`` is the guard on imported expressions."""

        for text in ("import os", "y = 1", "for i in x: pass"):
            with self.subTest(text=text):
                with self.assertRaises(SyntaxError):
                    get_expression_names(text)


class TestBareAggregates(unittest.TestCase):
    """Spotting the ``median``-where-``nanmedian``-was-meant mistake."""

    def test_flags_bare_and_ignores_nan_forms(self):
        """Both halves matter: a false positive would nag on good input."""

        self.assertEqual(
            get_bare_aggregates("median(a) + nanmedian(b) + std(c)"),
            {"median", "std"},
        )

    def test_ignores_names_that_are_not_calls(self):
        """A diagnostic called ``sum`` is not an aggregate call."""

        self.assertEqual(get_bare_aggregates("mean + 1"), set())


class TestDependents(unittest.TestCase):
    """The delete guard."""

    def test_finds_every_dependent(self):
        """Both members of the diamond, not just the first."""

        self.assertEqual(
            get_expression_dependents("rel", _library), {"scaled", "offset"}
        )

    def test_unreferenced_expression_has_none(self):
        """Deleting this one would break nothing."""

        self.assertEqual(get_expression_dependents("scaled", _library), set())


class TestOrdering(unittest.TestCase):
    """What has to be evaluated, and in what order."""

    def test_chain_is_ordered(self):
        """A reference comes before the expression using it."""

        order, _ = order_expressions(["scaled"], _library, _known)
        self.assertEqual(order, ["rel", "scaled"])

    def test_diamond_places_the_shared_expression_once(self):
        """The property that makes evaluation non-redundant."""

        order, _ = order_expressions(["scaled", "offset"], _library, _known)
        self.assertEqual(order.count("rel"), 1)
        self.assertLess(order.index("rel"), order.index("scaled"))
        self.assertLess(order.index("rel"), order.index("offset"))

    def test_only_the_targets_subtree_is_ordered(self):
        """Asking for one expression does not drag in the library."""

        order, _ = order_expressions(["offset"], _library, _known)
        self.assertNotIn("scaled", order)

    def test_plain_diagnostic_needs_no_evaluation(self):
        """A target that is simply recorded orders nothing."""

        order, needed = order_expressions(["bg_center"], _library, _known)
        self.assertEqual(order, [])
        self.assertEqual(needed, {"bg_center"})

    def test_needed_diagnostics_are_transitive(self):
        """Reported through the chain, not just one level down."""

        _, needed = order_expressions(["scaled"], _library, _known)
        self.assertEqual(needed, {"astrom_residual", "diagonal_fov"})

    def test_cycle_names_every_expression_involved(self):
        """Reported in one pass, rather than by a recursion guard."""

        with self.assertRaises(PipelineError) as caught:
            order_expressions(["a"], {"a": "b", "b": "c", "c": "a"}, _known)
        for name in ("a", "b", "c"):
            self.assertIn(name, str(caught.exception))

    def test_unresolvable_reference_is_refused(self):
        """And says which expression contains it."""

        with self.assertRaises(PipelineError) as caught:
            order_expressions(["x"], {"x": "no_such + 1"}, _known)
        self.assertIn("no_such", str(caught.exception))

    def test_unknown_target_is_refused(self):
        """A bookmarked URL naming nothing should not evaluate."""

        with self.assertRaises(PipelineError):
            order_expressions(["no_such"], _library, _known)


class TestEvaluation(unittest.TestCase):
    """Turning a library plus values into arrays."""

    def setUp(self):
        self.values = {
            "astrom_residual": numpy.array([1.0, 2.0, numpy.nan, 4.0]),
            "diagonal_fov": numpy.full(4, 2.0),
            "bg_center": numpy.array([10.0, 20.0, 30.0, 40.0]),
        }

    def test_composed_expression(self):
        """``rel`` is 0.5, 1, nan, 2, whose nanmedian is 1."""

        result = evaluate_expressions(["scaled"], _library, self.values)
        numpy.testing.assert_allclose(
            result["scaled"], [0.5, 1.0, numpy.nan, 2.0]
        )

    def test_nan_propagates_through_composition(self):
        """An image missing an input is undefined, not zero."""

        result = evaluate_expressions(["offset"], _library, self.values)
        self.assertTrue(numpy.isnan(result["offset"][2]))
        self.assertFalse(numpy.any(numpy.isnan(result["offset"][[0, 1, 3]])))

    def test_bare_aggregate_poisons_everything(self):
        """Why ``get_bare_aggregates`` exists, stated as behaviour.

        A plain ``median`` over an array with one NaN is NaN, so the whole
        series is, even for the images that do have the diagnostic.
        """

        result = evaluate_expressions(
            ["bad"],
            {"bad": "astrom_residual - median(astrom_residual)"},
            self.values,
        )
        self.assertTrue(numpy.all(numpy.isnan(result["bad"])))

    def test_several_targets_at_once(self):
        """What lets the two axes of a plot share one pass."""

        result = evaluate_expressions(
            ["scaled", "offset"], _library, self.values
        )
        self.assertEqual(set(result), {"scaled", "offset"})

    def test_plain_diagnostic_passes_through(self):
        """An axis need not be an expression."""

        result = evaluate_expressions(["bg_center"], _library, self.values)
        numpy.testing.assert_allclose(
            result["bg_center"], self.values["bg_center"]
        )

    def test_constant_expression_is_broadcast(self):
        """It still has to plot as a series."""

        result = evaluate_expressions(["k"], {"k": "3.5"}, self.values)
        numpy.testing.assert_allclose(result["k"], numpy.full(4, 3.5))

    def test_missing_values_are_refused(self):
        """Rather than evaluating to something meaningless."""

        with self.assertRaises(PipelineError) as caught:
            evaluate_expressions(
                ["rel"], _library, {"astrom_residual": numpy.zeros(4)}
            )
        self.assertIn("diagonal_fov", str(caught.exception))


class TestChecking(unittest.TestCase):
    """What is reported as wrong with a proposed expression."""

    def check(self, name, expression, expressions=None):
        """Return the problems, using the shared library by default."""

        return check_expression(
            name,
            expression,
            _library if expressions is None else expressions,
            _known,
        )

    def test_a_good_expression_has_no_problems(self):
        """The case that must not produce noise."""

        self.assertEqual(self.check("rel_doubled", "rel * 2"), [])

    def test_name_must_be_a_slug(self):
        """Anything else could be stored but never put in a URL."""

        self.assertTrue(self.check("not a slug", "bg_center"))

    def test_name_may_not_shadow_a_diagnostic(self):
        """Both are variables in one flat space, so it is ambiguous."""

        self.assertTrue(self.check("bg_center", "1"))

    def test_statements_are_reported_not_raised(self):
        """The security guard, surfaced as a problem for the user."""

        problems = self.check("x", "import os")
        self.assertEqual(len(problems), 1)
        self.assertIn("single expression", problems[0])

    def test_unknown_name_is_reported(self):
        """Naming what could not be resolved."""

        problems = self.check("x", "no_such * 2")
        self.assertTrue(any("no_such" in problem for problem in problems))

    def test_self_reference_is_a_cycle(self):
        """Caught by ordering the candidate library, not a special case."""

        problems = self.check("x", "x + 1")
        self.assertTrue(any("cycle" in problem for problem in problems))

    def test_cycle_with_an_existing_expression(self):
        """Editing one end of a pair is how a cycle usually arrives."""

        problems = check_expression("rel", "scaled + 1", _library, _known)
        self.assertTrue(any("cycle" in problem for problem in problems))

    def test_problems_accumulate(self):
        """A bad name and a bad body are both worth saying at once."""

        self.assertGreater(len(self.check("not a slug", "no_such")), 1)


if __name__ == "__main__":
    unittest.main()
