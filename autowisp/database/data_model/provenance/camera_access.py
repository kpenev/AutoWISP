"""Define the camera access dataset table for the pipeline"""

from sqlalchemy import Column, Integer, Index, ForeignKey

from autowisp.database.data_model.base import DataModelBase

# pylint false positive: this is actually a class name
# pylint: disable=invalid-name
# pylint: enable=invalid-name
__all__ = ["CameraAccess"]


# The standard use of SQLAlchemy ORM requires classes with no public methods.
# pylint: disable=too-few-public-methods
# association table
class CameraAccess(DataModelBase):
    """The table describing the camera access"""

    __tablename__ = "camera_access"

    observer_id = Column(
        Integer,
        ForeignKey("observer.id", onupdate="CASCADE", ondelete="RESTRICT"),
        doc="A unique identifier for the observer",
    )
    camera_id = Column(
        Integer,
        ForeignKey("camera.id", onupdate="CASCADE", ondelete="RESTRICT"),
        doc="A unique identifier of the camera",
    )

    __table_args__ = (
        Index("cam_access_key2", "observer_id", "camera_id", unique=True),
    )
