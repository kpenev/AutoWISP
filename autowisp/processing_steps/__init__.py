"""Add all configuration steps to __all__."""

from contextlib import redirect_stdout
from functools import cache
from glob import glob
from io import StringIO
from os.path import dirname, join, basename
from importlib import import_module

__all__ = []


#: Module names under this package that are shared helpers, not pipeline
#: steps, and so are excluded from :func:`get_step_names`.
_NON_STEP_MODULES = frozenset(
    {
        "__init__",
        "manual_util",
        "lc_detrending",
        "lc_detrending_argument_parser",
    }
)


def get_step_names():
    """Return the names of the pipeline processing-step modules.

    Discovered by scanning this package for ``*.py`` modules and dropping
    the shared helpers in :data:`_NON_STEP_MODULES`. This is the single
    source of truth for "what are the pipeline steps", so callers (the
    importer below, tests checking every step is wired up, ...) stay in
    sync as steps are added or removed.

    Returns:
        list:    The step module short names (e.g. ``"calibrate"``),
            sorted for a stable order.
    """

    return sorted(
        basename(step_path)[:-3]
        for step_path in glob(join(dirname(__file__), "*.py"))
        if basename(step_path)[:-3] not in _NON_STEP_MODULES
    )


@cache
def get_path_arguments():
    """
    Return the options of all pipeline steps that are paths.

    Returns:
        dict:    The option names, each mapped to its kind of path: see
            ``path`` in ``ManualStepArgumentParser.add_argument``.
    """

    result = {}
    for step_name in get_step_names():
        step = import_module("autowisp.processing_steps." + step_name)
        if not hasattr(step, "parse_command_line"):
            continue
        # The parsers report their defaults on stdout.
        with redirect_stdout(StringIO()):
            result.update(step.parse_command_line([])["path_arguments"])
    return result


def import_steps():
    """Import all pipeline steps (see :func:`get_step_names`)."""

    for step_name in get_step_names():
        full_name = "autowisp.processing_steps." + step_name
        import_module(full_name)
        __all__.append(full_name)


import_steps()
