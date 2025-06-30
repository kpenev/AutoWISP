"""Add all configuration steps to __all__."""

from glob import glob
from os.path import dirname, join, basename
from importlib import import_module

__all__ = []


def import_steps():
    """Import all configuration steps."""

    steps = filter(
        lambda step_name: step_name not in ["__init__"],
        (
            basename(step_path)[:-3]
            for step_path in glob(join(dirname(__file__), "*.py"))
        ),
    )
    for step_name in steps:

        step = import_module("autowisp.processing_steps." + step_name)
        __all__.append(step)


import_steps()
