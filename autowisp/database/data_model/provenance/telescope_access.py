"""Define the telescope access dataset table for the pipeline"""

from sqlalchemy import Column, Integer, Index, ForeignKey

from autowisp.database.data_model.base import DataModelBase

# pylint false positive: this is actually a class name
# pylint: disable=invalid-name
# pylint: enable=invalid-name
__all__ = ["TelescopeAccess"]


# The standard use of SQLAlchemy ORM requires classes with no public methods.
# pylint: disable=too-few-public-methods
class TelescopeAccess(DataModelBase):
    """The table describing the telescope access"""

    __tablename__ = "telescope_access"

    observer_id = Column(
        Integer,
        ForeignKey("observer.id", onupdate="CASCADE", ondelete="RESTRICT"),
        doc="A unique identifier for the observer",
    )
    telescope_id = Column(
        Integer,
        ForeignKey("telescope.id", onupdate="CASCADE", ondelete="RESTRICT"),
        doc="A unique identifier of the telescope",
    )

    __table_args__ = (
        Index("tel_access_key2", "observer_id", "telescope_id", unique=True),
    )
