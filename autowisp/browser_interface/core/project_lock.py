"""Tell the views whether the current project may be changed."""

from autowisp.database.experiments import ProjectExperiments
from autowisp.database.interface import get_db_engine, get_project_home
from autowisp.exceptions import ProjectLockedError


def get_lock_refusal():
    """
    Return why the current project cannot be changed, or None if it can.

    A project is locked while it has experiments that may still be merged.
    The views changing it ask first and show the reason, rather than letting
    the refusal escape and be recorded as an error: it is the expected
    answer, not a failure.

    Returns:
        str or None:    The refusal, naming the experiments holding the
            lock; None if the project is not locked.
    """

    try:
        ProjectExperiments(
            get_db_engine(), get_project_home()
        ).refuse_if_locked()
    except ProjectLockedError as error:
        return str(error)
    return None
