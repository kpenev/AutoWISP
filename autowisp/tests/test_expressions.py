"""Tests for the database-free half of diagnostic expressions.

Everything here runs against a library and values passed in as arguments,
so there is no database, no Django and no fixture beyond a dictionary --
which is the point of keeping this tier free of both.
"""

import unittest

import numpy

from autowisp.diagnostics.diagnostic_types import (
    quantiles_quantity,
    time_quantity,
)
from autowisp.diagnostics.expressions import (
    QuantityLookUp,
    check_expression,
    evaluate_quantities,
    get_bare_aggregates,
    get_expression_dependents,
    get_expression_names,
    get_expression_parameters,
    get_indexed_names,
    get_needed_values,
    get_quantity_arity,
    order_expressions,
    rename_references,
)
from autowisp.exceptions import PipelineError

#: A small library with a diamond in it: two expressions share ``rel``.
#: Everything is read in slot 1, this being about composition rather than
#: about channels; the slot classes below bind several.
_library = {
    "rel": "astrom_residual[1] / diagonal_fov[1]",
    "scaled": "rel[1] / nanmedian(rel[1])",
    "offset": "rel[1] + bg_center[1]",
}


class TestReferencedNames(unittest.TestCase):
    """Reading the names out of an expression."""

    def test_names_include_functions(self):
        """Splitting them apart is the caller's job, not this one's."""

        self.assertEqual(
            get_expression_names("rel / nanmedian(rel)"), {"rel", "nanmedian"}
        )

    def test_statements_are_rejected(self):
        """``mode="eval"`` refuses anything that is not one expression."""

        for text in ("import os", "y = 1", "for i in x: pass"):
            with self.subTest(text=text):
                with self.assertRaises(SyntaxError):
                    get_expression_names(text)


class TestReachableNames(unittest.TestCase):
    """What an expression is allowed to reach.

    Parsing is not what makes this safe -- most attacks are perfectly valid
    expressions. The evaluator's symbol table is, and these pin the part of
    it AutoWISP chooses rather than inherits.
    """

    def test_filesystem_access_is_refused(self):
        """``open`` is dropped from the evaluator, so the name is unknown.

        Asteval permits reading files; harmless for text the local user
        typed, but expressions travel between installations in export
        files, so a shared one must not be able to read ``~/.ssh``.
        """

        problems = check_expression("evil", "open('/etc/passwd').read()", {})
        self.assertTrue(any("open" in problem for problem in problems))

    def test_hidden_filesystem_access_is_refused(self):
        """Also when buried in something that would otherwise plot."""

        problems = check_expression(
            "evil",
            "bg_center + len(open('/etc/passwd').read())",
            {},
        )
        self.assertTrue(any("open" in problem for problem in problems))

    def test_side_effects_are_refused(self):
        """``print`` returns nothing, so it was never a valid value."""

        self.assertTrue(check_expression("noisy", "print(bg_center)", {}))

    def test_the_mathematical_subset_still_works(self):
        """The removals must not cost anything anyone would write."""

        self.assertEqual(
            check_expression(
                "fine",
                "bg_center[1] - nanmedian(bg_center[1]) "
                "+ sqrt(abs(bg_center[1]))",
                {},
            ),
            [],
        )


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


class TestRenamingReferences(unittest.TestCase):
    """Carrying dependents through a rename.

    The one place expression text is rewritten, so what it must *not* touch
    matters as much as what it must.
    """

    def test_every_use_is_renamed(self):
        """Including two uses in the one expression."""

        self.assertEqual(
            rename_references("rel / nanmedian(rel)", "rel", "relative"),
            "relative / nanmedian(relative)",
        )

    def test_a_longer_name_containing_it_is_untouched(self):
        """The case that kills a textual replace.

        ``rel_bg`` merely starts with ``rel``; renaming ``rel`` must leave
        it alone, and only ``ast`` knows where one identifier ends.
        """

        self.assertEqual(
            rename_references("rel_bg + rel", "rel", "r"), "rel_bg + r"
        )

    def test_a_string_literal_is_untouched(self):
        """Text that merely looks like the name is not a reference."""

        self.assertEqual(
            rename_references("where(flag == 'rel', rel, 0)", "rel", "r"),
            "where(flag == 'rel', r, 0)",
        )

    def test_spacing_and_parentheses_survive(self):
        """Splicing rather than unparsing, so nothing is reformatted."""

        original = "( rel+1 )*2  # keep me"
        self.assertEqual(
            rename_references(original, "rel", "r"), "( r+1 )*2  # keep me"
        )

    def test_offsets_survive_a_non_ascii_character(self):
        """``ast`` counts utf-8 bytes, so the splice has to as well.

        A multi-byte character earlier in the line shifts every offset
        after it, and getting this wrong corrupts the text rather than
        failing.
        """

        self.assertEqual(
            rename_references("where(unit == 'µm', rel, 0)", "rel", "r"),
            "where(unit == 'µm', r, 0)",
        )

    def test_an_unmentioned_name_changes_nothing(self):
        """Every expression in a library is offered to this, not just the
        dependents."""

        self.assertEqual(
            rename_references("bg_center * 2", "rel", "r"), "bg_center * 2"
        )


class TestOrdering(unittest.TestCase):
    """What has to be evaluated, and in what order."""

    def test_chain_is_ordered(self):
        """A reference comes before the expression using it."""

        order, _ = order_expressions(["scaled"], _library)
        self.assertEqual(order, ["rel", "scaled"])

    def test_diamond_places_the_shared_expression_once(self):
        """The property that makes evaluation non-redundant."""

        order, _ = order_expressions(["scaled", "offset"], _library)
        self.assertEqual(order.count("rel"), 1)
        self.assertLess(order.index("rel"), order.index("scaled"))
        self.assertLess(order.index("rel"), order.index("offset"))

    def test_only_the_targets_subtree_is_ordered(self):
        """Asking for one expression does not drag in the library."""

        order, _ = order_expressions(["offset"], _library)
        self.assertNotIn("scaled", order)

    def test_plain_diagnostic_needs_no_evaluation(self):
        """A target that is simply recorded orders nothing."""

        order, needed = order_expressions(["bg_center"], _library)
        self.assertEqual(order, [])
        self.assertEqual(needed, {"bg_center"})

    def test_needed_diagnostics_are_transitive(self):
        """Reported through the chain, not just one level down."""

        _, needed = order_expressions(["scaled"], _library)
        self.assertEqual(needed, {"astrom_residual", "diagonal_fov"})

    def test_cycle_names_every_expression_involved(self):
        """The whole loop, in the order it closes."""

        with self.assertRaises(PipelineError) as caught:
            order_expressions(["a"], {"a": "b", "b": "c", "c": "a"})
        for name in ("a", "b", "c"):
            self.assertIn(name, str(caught.exception))

    def test_cycle_names_the_loop_and_not_its_dependents(self):
        """An expression merely downstream of a cycle is not implicated.

        ``d`` is unreachable because ``a`` and ``b`` reference each other,
        but ``d`` is not itself part of the problem and naming it would
        send the reader to the wrong expression.
        """

        with self.assertRaises(PipelineError) as caught:
            order_expressions(["d"], {"a": "b", "b": "a", "d": "a + 1"})
        message = str(caught.exception)
        self.assertIn("a", message)
        self.assertIn("b", message)
        self.assertNotIn("d", message)

    def test_unresolvable_reference_is_refused(self):
        """And says which expression contains it."""

        with self.assertRaises(PipelineError) as caught:
            order_expressions(["x"], {"x": "no_such + 1"})
        self.assertIn("no_such", str(caught.exception))

    def test_unknown_target_is_refused(self):
        """A bookmarked URL naming nothing should not evaluate."""

        with self.assertRaises(PipelineError):
            order_expressions(["no_such"], _library)


class TestQuantileNames(unittest.TestCase):
    """The diagnostics named by a pattern rather than listed.

    ``pixel_q*`` diagnostics are created by ``calibrate`` rather than
    seeded, so they are the one part of the vocabulary that cannot be
    enumerated. They must still resolve as variables and still be refused
    as expression names, and both come from the same predicate.
    """

    def test_a_quantile_resolves_as_a_variable(self):
        """Nothing enumerates these, so only the pattern can accept them."""

        self.assertEqual(check_expression("q", "pixel_q999[1] * 2", {}), [])

    def test_a_quantile_may_not_be_taken_as_a_name(self):
        """The predicate reserves as well as resolves, which is easy to miss.

        Otherwise an expression could be named ``pixel_q999`` and shadow a
        real quantile -- ambiguous in a flat name space, and undetectable
        afterwards because by then both are simply variables.
        """

        self.assertTrue(check_expression("pixel_q999", "1", {}))

    def test_a_near_miss_is_still_unresolvable(self):
        """The pattern must not become a licence for anything similar."""

        self.assertTrue(check_expression("q", "pixel_quality * 2", {}))

    def test_the_family_name_may_not_be_taken_either(self):
        """``pixel_quantiles`` is a selector name, so it is reserved too.

        It is the one reserved name that is not a readable quantity: it
        stands for the whole ``pixel_q*`` family, expanding to one series
        per member.  An expression allowed to take it would be swallowed by
        that expansion and silently never drawn.
        """

        self.assertTrue(check_expression(quantiles_quantity, "1", {}))

    def test_the_family_name_does_not_resolve_as_a_variable(self):
        """Reserved is not the same as readable, and here they differ.

        A family has no values of its own, so an expression referencing it
        is a mistake rather than a way of reaching every quantile at once.
        """

        self.assertTrue(check_expression("q", f"{quantiles_quantity} * 2", {}))

    def test_a_quantile_survives_the_ordering_pass(self):
        """A composed expression must not be rejected by the cycle check.

        Ordering and the direct check have to agree about what a name may
        mean, or ``check_expression`` contradicts itself: accepting a name,
        then reporting it unresolvable from the pass looking for cycles.
        One shared predicate is what makes that impossible.
        """

        library = {"q": "pixel_q999[1] / pixel_q500[1]"}
        self.assertEqual(check_expression("q_scaled", "q[1] * 2", library), [])


class TestNoProjectNeeded(unittest.TestCase):
    """Validation is the same everywhere, so it needs no database.

    A ``diagnostic_type`` row can only be seeded from the static catalogue
    or created by the quantile branch, which refuses every other name. So
    the vocabulary cannot vary between projects, and an expression means
    the same thing in all of them -- which is what lets one library be
    shared.
    """

    def test_a_catalogue_diagnostic_resolves(self):
        """Even though nothing here has opened a project database."""

        self.assertEqual(
            check_expression("rel", "astrom_residual[1] / diagonal_fov[1]", {}),
            [],
        )

    def test_a_misspelling_does_not(self):
        """The check is still a check: unknown names are still refused."""

        problems = check_expression("typo", "bg_centre / diagonal_fov", {})
        self.assertTrue(any("bg_centre" in problem for problem in problems))

    def test_jd_is_a_variable(self):
        """Not a diagnostic, but a name in the same flat space."""

        self.assertEqual(
            check_expression("rel_time", "jd - nanmin(jd)", {}), []
        )


class TestChecking(unittest.TestCase):
    """What is reported as wrong with a proposed expression."""

    def check(self, name, expression, expressions=None):
        """Return the problems, using the shared library by default."""

        return check_expression(
            name,
            expression,
            _library if expressions is None else expressions,
        )

    def test_a_good_expression_has_no_problems(self):
        """The case that must not produce noise."""

        self.assertEqual(self.check("rel_doubled", "rel[1] * 2"), [])

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

        problems = check_expression("rel", "scaled[1] + 1", _library)
        self.assertTrue(any("cycle" in problem for problem in problems))

    def test_problems_accumulate(self):
        """A bad name and a bad body are both worth saying at once."""

        self.assertGreater(len(self.check("not a slug", "no_such")), 1)


class SlotTestCase(unittest.TestCase):
    """What the channel-slot tests share.

    The library exercises every shape a slot comes in: two slots; one
    expression used at two different bindings, and one used twice at the
    same; a nested expression binding the slots the other way round; a
    quantity over the time alone, one built on that, and one mixing a
    subscripted diagnostic with a bare reference.
    """

    library = {
        "sky_color": "bg_center[1] / bg_center[2]",
        "silly": "sky_color[1,2] - sky_color[2,3]",
        "twice": "sky_color[1,2] + sky_color[1,2]",
        "inner": "bg_center[1] / bg_center[2]",
        "outer": "inner[2,1] + bg_center[1]",
        "night": "jd - nanmin(jd)",
        "scaled_night": "night * 2",
        "mixed": "bg_center[1] * night",
    }

    #: Two images' worth, distinct per channel so that reading the wrong
    #: one cannot pass by coincidence.
    jd = numpy.array([1.0, 3.0])
    bg = {
        "B": numpy.array([4.0, 3.7]),
        "R": numpy.array([3.9, 2.3]),
        "G0": numpy.array([3.0, 1.0]),
    }

    def needed(self, wanted):
        """Return what has to be read for *wanted*."""

        return get_needed_values(wanted, self.library)

    def fetch(self, wanted):
        """Return exactly the values :meth:`needed` asks for.

        Deliberately not "every channel available": supplying more would
        hide a walk that under-reports, and that walk answers the series
        table as well as the fetch, so an omission there is a wrong count
        rather than merely a missing array.
        """

        return {
            name: {
                channels: (
                    self.jd if name == time_quantity else self.bg[channels[0]]
                )
                for channels in channel_set
            }
            for name, channel_set in self.needed(wanted).items()
        }

    def evaluate(self, wanted):
        """Evaluate *wanted* against exactly what the walk asked for."""

        return evaluate_quantities(wanted, self.library, self.fetch(wanted))


class TestSlotSyntax(SlotTestCase):
    """Reading channel slots out of an expression."""

    def test_references_keep_their_slots_and_repeats(self):
        """One name at two bindings is the point of the parameters."""

        self.assertEqual(
            get_indexed_names(self.library["silly"]),
            [("sky_color", (1, 2)), ("sky_color", (2, 3))],
        )

    def test_one_slot_is_still_a_tuple(self):
        """So no caller has to care how many were written."""

        self.assertEqual(
            get_indexed_names("bg_center[1]"), [("bg_center", (1,))]
        )

    def test_parameters_are_derived_and_ordered(self):
        """Nothing is declared, so nothing can drift out of step."""

        self.assertEqual(
            get_expression_parameters(self.library["silly"]), (1, 2, 3)
        )
        self.assertEqual(get_expression_parameters("jd * 2"), ())

    def test_parameters_are_sorted_whatever_order_the_body_uses(self):
        """The order is what a reference's arguments are matched onto."""

        self.assertEqual(
            get_expression_parameters("bg_center[3] - bg_center[1]"), (1, 3)
        )

    def test_a_slot_must_be_a_whole_number(self):
        """Anything else cannot name a channel, and says so here rather
        than failing later as an unresolvable name."""

        for text in (
            "bg_center[i]",
            "bg_center[1.5]",
            "bg_center[1:2]",
            "bg_center[1][2]",
        ):
            with self.subTest(text=text):
                with self.assertRaises(PipelineError):
                    get_indexed_names(text)

    def test_arity_is_a_rule_not_a_table(self):
        """Two of the three answers are constants."""

        self.assertEqual(get_quantity_arity("bg_center", self.library), 1)
        self.assertEqual(get_quantity_arity(time_quantity, self.library), 0)
        self.assertEqual(get_quantity_arity("silly", self.library), 3)
        with self.assertRaises(PipelineError):
            get_quantity_arity("no_such", self.library)


class TestNeededValues(SlotTestCase):
    """What has to be read before anything is evaluated.

    Answered for two callers at once -- whatever reads a series in one
    query, and the table counting the images that record all of it without
    evaluating anything -- so it has to be exact in both directions. The
    assertions compare whole dictionaries for that reason: what is *not*
    needed matters as much, since over-reporting makes a count wrong
    rather than merely fetching too much.
    """

    def test_one_expression_at_two_bindings_needs_both(self):
        """The channels come from resolving each reference's arguments."""

        self.assertEqual(
            self.needed({"silly": ("B", "R", "G0")}),
            {"bg_center": {("B",), ("R",), ("G0",)}},
        )

    def test_slots_bound_the_other_way_round_are_followed(self):
        """``inner[2,1]`` reads the outer binding reversed."""

        self.assertEqual(
            self.needed({"outer": ("R", "B")}),
            {"bg_center": {("R",), ("B",)}},
        )

    def test_the_time_is_found_through_a_bare_reference(self):
        """``mixed`` reads the time only by way of ``night``.

        The time is the one quantity written without a subscript, so the
        walk over subscripts never reaches it however deep it lies.
        """

        self.assertEqual(
            self.needed({"mixed": ("B",)}),
            {time_quantity: {()}, "bg_center": {("B",)}},
        )

    def test_a_binding_of_the_wrong_length_is_refused(self):
        """Too few or too many: either would plot something else.

        Too many is the one that could pass unnoticed -- the extra channel
        binds no parameter, so it would simply be dropped.
        """

        for channels in (("B",), ("B", "R", "G0")):
            with self.subTest(channels=channels):
                with self.assertRaises(PipelineError):
                    self.needed({"sky_color": channels})


class TestSlotEvaluation(SlotTestCase):
    """Evaluating quantities against the values the walk asked for.

    Every case here fetches exactly what the walk reported, so each also
    asserts the two agree -- the pairing that would otherwise drift apart
    unnoticed.
    """

    def test_one_body_evaluated_at_two_bindings(self):
        """``sky_color`` is B/R in one term and R/G0 in the other."""

        numpy.testing.assert_allclose(
            self.evaluate({"silly": ("B", "R", "G0")})["silly"],
            self.bg["B"] / self.bg["R"] - self.bg["R"] / self.bg["G0"],
        )

    def test_a_nested_binding_is_not_the_callers(self):
        """``outer[1,2] = inner[2,1] + bg_center[1]``.

        The case that returns a wrong number rather than an error if a
        lookup ever holds a binding of its own instead of reading the one
        belonging to the body being evaluated.
        """

        numpy.testing.assert_allclose(
            self.evaluate({"outer": ("R", "B")})["outer"],
            self.bg["B"] / self.bg["R"] + self.bg["R"],
        )

    def test_an_instantiation_is_computed_once(self):
        """Asked for twice, the *same array* comes back.

        Identity rather than equality, and rather than the size of the
        cache: re-evaluating would overwrite the one entry with an equal
        but distinct array, so only identity tells the two apart.
        """

        lookups = QuantityLookUp.library(
            {"sky_color": self.library["sky_color"]},
            {"bg_center": {("B",): self.bg["B"], ("R",): self.bg["R"]}},
        )

        self.assertIs(
            lookups["sky_color"].at(("B", "R")),
            lookups["sky_color"].at(("B", "R")),
        )

    def test_both_axes_at_once_share_their_instantiations(self):
        """Which is why the two are resolved in one call, not one each."""

        drawn = self.evaluate({"sky_color": ("B", "R"), "twice": ("B", "R")})
        numpy.testing.assert_allclose(drawn["twice"], 2 * drawn["sky_color"])

    def test_a_quantity_over_the_time_alone(self):
        """Written bare, there being no channel to subscript it with."""

        numpy.testing.assert_allclose(
            self.evaluate({"night": ()})["night"], self.jd - self.jd.min()
        )

    def test_a_chain_of_channel_free_expressions(self):
        """``scaled_night`` reads ``night`` reads the time, all bare.

        Each has to be a value in the symbol table rather than a lookup,
        since a bare name never reaches ``__getitem__``, and in an order
        that respects what they read.
        """

        numpy.testing.assert_allclose(
            self.evaluate({"scaled_night": ()})["scaled_night"],
            2 * (self.jd - self.jd.min()),
        )

    def test_slotted_and_channel_free_in_one_expression(self):
        """``mixed`` subscripts one quantity and reads another bare."""

        numpy.testing.assert_allclose(
            self.evaluate({"mixed": ("B",)})["mixed"],
            self.bg["B"] * (self.jd - self.jd.min()),
        )

    def test_a_plain_diagnostic_is_a_quantity_too(self):
        """One channel, no expression, and the same call resolves it."""

        numpy.testing.assert_allclose(
            self.evaluate({"bg_center": ("R",)})["bg_center"], self.bg["R"]
        )

    def test_only_what_is_reached_is_built(self):
        """The library holds expressions about data nobody fetched.

        Building those would waste work, and for one read bare -- which is
        evaluated on sight, so that it can be a value -- would raise about
        values nobody asked for.
        """

        self.assertNotIn(time_quantity, self.fetch({"silly": ("B", "R", "G0")}))
        self.evaluate({"silly": ("B", "R", "G0")})

    def test_an_unfetched_channel_is_reported(self):
        """A miss means the walk and the fetch disagree, which is a fault
        in the pair rather than something to repair here."""

        with self.assertRaises(PipelineError):
            evaluate_quantities(
                {"sky_color": ("B", "R")},
                self.library,
                {"bg_center": {("B",): self.bg["B"]}},
            )


if __name__ == "__main__":
    unittest.main()
