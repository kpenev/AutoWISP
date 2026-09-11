"""Slot evaluation with one lookup class -- a prototype, not the pipeline.

Working sketch of what ``diagnostic_expressions_plan.md`` §10 describes,
kept because the design was arrived at by running it rather than by
arguing about it, and because the tier-1 work it stands for has not been
written yet.  Run it directly to see the cases it pins.

Deliberately outside ``autowisp/``: it is not imported by anything, not
installed (the root ``meson.build`` descends only into the package), and
should be **deleted once §10 lands** in
``autowisp/diagnostics/expressions.py``.

A diagnostic is the degenerate case of an expression: every instantiation
of it is known before evaluation starts, so it arrives with its cache full
and its text never consulted.  That leaves one class, one cache and one
question -- ``at(channels)`` -- for everything an expression can read.

The cases below are the ones worth keeping as tests:

* ``silly`` -- one stored body instantiated twice at different channels.
* ``outer`` -- a nested expression binding the slots the other way round,
  which is where a lookup holding a frozen binding would return a wrong
  number rather than an error.
* ``twice`` -- one instantiation however many references reach it.
* an unfetched channel -- reported rather than evaluated.
"""

import numpy

from autowisp.evaluator import Evaluator
from autowisp.exceptions import PipelineError


# The class exists to answer one subscript; that is the whole interface.
# pylint: disable=too-few-public-methods
class QuantityLookUp:
    """One name an expression may read, resolved when it is asked for.

    Built through :meth:`library` rather than one at a time: the lookups
    of one evaluation have to share a binding stack and an evaluator, and
    both are this class's business rather than its caller's.
    """

    def __init__(self, name, shared, *, definition=None, computed=None):
        """
        Args:
            name(str):    What the expressions call it, for error messages
                only -- nothing resolves by it.

            shared(tuple):    The binding stack and evaluator this
                evaluation's lookups share, from :meth:`library`.

            definition(tuple):    The stored text and its parameters, as
                the library holds them, or ``None`` for a diagnostic --
                which is never evaluated, so it has neither.

            computed(dict):    ``channels -> array`` known in advance: the
                whole of a diagnostic, and empty for an expression.
        """

        self._name = name
        self._stack, self._evaluator = shared
        self._text, self._parameters = definition or (None, ())
        self._computed = dict(computed or {})

    @classmethod
    def library(cls, expressions, values):
        """
        Return ``{name: lookup}`` for one evaluation, ready to be asked.

        The stack and the evaluator are created here and captured by the
        lookups, so neither appears outside this class.  The stack in
        particular is how a lookup finds the binding of the body asking it,
        which is nobody else's concern; and it must belong to **one
        evaluation**, not to the class, or two plots drawn at once would
        interleave their bindings on it.

        The evaluator does not come back out either: handing it over would
        hand over a symbol table full of lookups, and with it a way to
        evaluate arbitrary text with none of the parameter machinery.

        Args:
            expressions(dict):    ``name -> (text, parameters)``.

            values(dict):    ``diagnostic -> {channel: array}``, fetched.
        """

        shared = ([], Evaluator({}))
        lookups = {
            name: cls(
                name,
                shared,
                computed={
                    (channel,): array for channel, array in by_channel.items()
                },
            )
            for name, by_channel in values.items()
        }
        lookups.update(
            {
                name: cls(name, shared, definition=definition)
                for name, definition in expressions.items()
            }
        )
        shared[1].symtable.update(lookups)
        return lookups

    def __getitem__(self, slots):
        """Resolve ``name[slots]`` as the body being evaluated means it.

        Python passes ``1`` for ``x[1]`` and ``(1, 2)`` for ``x[1,2]``, so
        the shapes take care of themselves.  The binding read is the top of
        the stack, which is always the body asking: a nested evaluation
        finishes in here before the outer body's next operand is touched.
        """

        if not isinstance(slots, tuple):
            slots = (slots,)
        binding = self._stack[-1]
        return self.at(tuple(binding[slot] for slot in slots))

    def at(self, channels):
        """Return this quantity with its parameters bound to *channels*.

        The cache is what makes an instantiation wanted twice -- by two
        references, or by two different expressions -- computed once. For a
        diagnostic it is also the whole of the answer, so a miss means the
        values were never fetched rather than that something needs
        computing.

        Raises:
            PipelineError:    If a diagnostic was not fetched for this
                channel, which is a fault in whatever decided what to
                fetch.
        """

        if channels not in self._computed:
            if self._text is None:
                raise PipelineError(
                    f"No values supplied for {self._name} in "
                    + ", ".join(channels)
                    + ".",
                    details={"missing": [self._name], "channels": channels},
                )
            self._stack.append(dict(zip(self._parameters, channels)))
            try:
                self._computed[channels] = self._evaluator(self._text)
            finally:
                self._stack.pop()
        return self._computed[channels]


# pylint: enable=too-few-public-methods


def evaluate_quantities(wanted, library, values):
    """Return ``{quantity: array}`` for the quantities of one series.

    Args:
        wanted(dict):    ``quantity name -> channels``, one channel per
            parameter of that quantity, as the table bound them.

        library(dict):    ``name -> (text, parameters)``.

        values(dict):    ``diagnostic -> {channel: array}``, from tier 2.
    """

    lookups = QuantityLookUp.library(library, values)
    return {
        name: lookups[name].at(channels) for name, channels in wanted.items()
    }


if __name__ == "__main__":
    VALUES = {
        "bg_center": {
            "B": numpy.array([4.0, 3.7]),
            "R": numpy.array([3.9, 2.3]),
            "G0": numpy.array([3.9, 2.3]),
            "G1": numpy.array([3.0, 1.0]),
        }
    }
    LIBRARY = {
        "sky_color": ("bg_center[1] / bg_center[2]", (1, 2)),
        "silly": ("sky_color[1,2] - sky_color[2,3]", (1, 2, 3)),
        "inner": ("bg_center[1] / bg_center[2]", (1, 2)),
        "outer": ("inner[2,1] + bg_center[1]", (1, 2)),
        "twice": ("sky_color[1,2] + sky_color[1,2]", (1, 2)),
    }

    for quantity, array in evaluate_quantities(
        {"silly": ("B", "G0", "G1"), "bg_center": ("R",)}, LIBRARY, VALUES
    ).items():
        print(f"{quantity:10} {array}")
    print("expected   silly [-0.27435897 -0.69130435], bg_center [3.9 2.3]")

    print(
        "\nouter (R,B):",
        evaluate_quantities({"outer": ("R", "B")}, LIBRARY, VALUES)["outer"],
    )
    print(
        "expected   :",
        VALUES["bg_center"]["B"] / VALUES["bg_center"]["R"]
        + VALUES["bg_center"]["R"],
    )

    LOOKUPS = QuantityLookUp.library(LIBRARY, VALUES)
    LOOKUPS["twice"].at(("B", "G0"))
    # pylint: disable=protected-access
    print("\nsky_color computed at:", list(LOOKUPS["sky_color"]._computed))

    try:
        evaluate_quantities({"sky_color": ("B", "NOPE")}, LIBRARY, VALUES)
    except PipelineError as error:
        print("\nunfetched channel:", str(error).splitlines()[-1])
