"""Define the mount access dataset table for the pipeline"""

from sqlalchemy import Column, Integer, Index, ForeignKey

from autowisp.database.data_model.base import DataModelBase

# pylint false positive: this is actually a class name
# pylint: disable=invalid-name
# pylint: enable=invalid-name
__all__ = ["MountAccess"]


# The standard use of SQLAlchemy ORM requires classes with no public methods.
# pylint: disable=too-few-public-methods
class MountAccess(DataModelBase):
    """The table dscribing the mount access"""

    __tablename__ = "mount_access"

    observer_id = Column(
        Integer,
        ForeignKey("observer.id", onupdate="CASCADE", ondelete="RESTRICT"),
        doc="A unique identifier for the observer",
    )
    mount_id = Column(
        Integer,
        ForeignKey("mount.id", onupdate="CASCADE", ondelete="RESTRICT"),
        doc="A unique identifier of the mount",
    )

    __table_args__ = (
        Index("mount_access_key2", "observer_id", "mount_id", unique=True),
    )
