"""Define the class that sets the processing order of step/image type."""

from sqlalchemy import Column, Integer, ForeignKey

from sqlalchemy.orm import relationship

from autowisp.database.data_model.base import DataModelBase

__all__ = ["ProcessingSequence"]


class ProcessingSequence(DataModelBase):
    """The sequence of steps/image type to be processed by the pipeline."""

    __tablename__ = "processing_sequence"

    step_id = Column(
        Integer,
        ForeignKey("step.id"),
        nullable=False,
        doc="The step to be executed.",
    )
    image_type_id = Column(
        Integer,
        ForeignKey("image_type.id"),
        nullable=True,
        doc="The image type to be processed by the step.",
    )

    step = relationship("Step")
    image_type = relationship("ImageType")

    def __repr__(self):
        return f"{self.step.name} {self.image_type.name}"
