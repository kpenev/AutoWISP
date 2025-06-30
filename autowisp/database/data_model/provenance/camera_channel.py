"""Define the camera color channel table for the pipeline"""

from sqlalchemy import Column, Integer, String, Index, ForeignKey

from autowisp.database.data_model.base import DataModelBase

# pylint false positive: this is actually a class name
# pylint: disable=invalid-name
# pylint: enable=invalid-name
__all__ = ["CameraChannel"]


class CameraChannel(DataModelBase):
    """The table describing each of the channels of a type of camera."""

    __tablename__ = "camera_channel"

    camera_type_id = Column(
        Integer,
        ForeignKey("camera_type.id"),
        doc="The camera type to which this channel belongs.",
    )
    name = Column(String(10), doc="A label to assign to the channel.")
    x_offset = Column(
        Integer,
        nullable=False,
        doc="The x index of the first pixel of the channel.",
    )
    y_offset = Column(
        Integer,
        nullable=False,
        doc="The y index of the first pixel of the channel.",
    )
    x_step = Column(
        Integer,
        nullable=False,
        default=2,
        doc="The step in the x direction between consecutive pixels of the "
        "channel.",
    )
    y_step = Column(
        Integer,
        nullable=False,
        default=2,
        doc="The step in the y direction between consecutive pixels of the "
        "channel.",
    )

    __table_args__ = (
        Index("cam_channel_key2", "camera_type_id", "name", unique=True),
    )

    def __repr__(self):
        return (
            f"{self.name}("
            f"{self.x_offset}, {self.x_step}; "
            f"{self.y_offset}, {self.y_step})"
        )
