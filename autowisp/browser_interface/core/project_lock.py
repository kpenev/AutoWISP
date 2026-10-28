"""Tell the views whether the current project may be changed."""

from autowisp.database.experiments import ExperimentRecords
from autowisp.database.interface import get_db_engine, get_project_home
from autowisp.exceptions import ProjectLockedError


def get_lock_refusal():
    """
    Return why the current project cannot be changed, or None if it can.

    A project is locked while it has experiments that may still be merged.
    The views changing it ask first and show the reason, rather than letting
    the refusal escape and be recorded as an error: it is the expected
    answer, not a failure. Worded for the interface, whose way of unlocking
    is the progress page's, not the command line's.

    Returns:
        str or None:    The refusal, naming the experiments holding the
            lock; None if the project is not locked.
    """

    try:
        ExperimentRecords(
            get_db_engine(), get_project_home()
        ).refuse_if_locked()
    except ProjectLockedError as error:
        return (
            "The project cannot be changed while it has experiments that may "
            "still be merged: "
            + ", ".join(
                f"{name} ({state})"
                for name, state in error.details["experiments"].items()
            )
            + ". Merge or delete them, or unlock the project by clicking "
            '"Locked by experiments" on the progress page, which marks the '
            "current ones stale; one still being created can only be deleted."
        )
    return None
