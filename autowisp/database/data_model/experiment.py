"""Define the :class:`Experiment` table, recording a project's experiments."""

from sqlalchemy import Column, ForeignKey, Integer, String, Text, TIMESTAMP

from autowisp.database.data_model.base import DataModelBase

__all__ = ["Experiment"]


# The standard use of SQLAlchemy ORM requires classes with no public methods.
# pylint: disable=too-few-public-methods
class Experiment(DataModelBase):
    """
    An experiment: a copy of this project, to be processed on its own.

    The table lives only in the original's database, and is the only record
    of a project's experiments. An experiment is an ordinary project whose
    database has no such table: creating one, and resetting its processing,
    drop it from the copy, and nothing in the copy refers to its original.
    So **a missing table means the project is an experiment**, which is why
    every read of it goes through
    ``ExperimentRecords._read_experiments``
    (:mod:`autowisp.database.experiments`).

    A later revision changing this table must leave a missing one missing.
    Recreating it would silently turn an experiment into an ordinary project,
    one able to have experiments of its own and to write outside its home.
    """

    __tablename__ = "experiment"

    name = Column(
        String(100),
        nullable=False,
        unique=True,
        doc="The experiment's name, which is also its directory under the "
        "original's ``experiments/``.",
    )
    home = Column(
        String(1000),
        nullable=False,
        doc="The experiment's home, in stored form: "
        "``{PROJHOME}/experiments/<name>``.",
    )
    created = Column(
        TIMESTAMP,
        nullable=False,
        doc="When the original was copied: at creation, and again each time "
        "the experiment's processing is reset.",
    )
    fingerprint = Column(
        Text,
        nullable=False,
        doc="The original's fingerprint when it was copied, as JSON: its "
        "revision and a hash of each table (see "
        ":class:`autowisp.database.experiments.ExperimentRecords`).",
    )
    state = Column(
        String(20),
        nullable=False,
        doc="'creating' until the copy is complete, then 'current', or "
        "'stale' once the original has changed since it was copied.",
    )
    copy_run_id = Column(
        Integer,
        ForeignKey("pipeline_run.id", onupdate="CASCADE"),
        nullable=False,
        doc="The run that last copied the original into the experiment: its "
        "create, or its latest reset. While the experiment is 'creating', "
        "it tells whether the copy is still going or crashed. Follows the "
        "run's id when a merge renumbers it.",
    )

    def __repr__(self):
        return f"({self.id}) {self.name}: {self.state}"


# pylint: enable=too-few-public-methods
