"""Define the table of migrations a project still needs."""

from sqlalchemy import Column, String

from autowisp.database.data_model.base import DataModelBase

__all__ = ["PendingMigration"]


# The standard use of SQLAlchemy ORM requires classes with no public methods.
# pylint: disable=too-few-public-methods
class PendingMigration(DataModelBase):
    """
    A migration of a project that a revision cannot do itself.

    A revision changes only the database, but some changes reach the files a
    project has produced, and rewriting those can take far longer than opening
    a project should. Such a revision records here what remains, a row per
    kind of file so that an interrupted migration resumes where it was, and
    the project is not processed until every row is done and deleted.
    """

    __tablename__ = "pending_migration"

    name = Column(
        String(100),
        nullable=False,
        unique=True,
        doc="What is still to be migrated, e.g. ``lightcurves``.",
    )

    def __repr__(self):
        return f"({self.id}) {self.name}: pending since {self.timestamp}"
