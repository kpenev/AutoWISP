"""Define class to specify dependencies between steps."""

from sqlalchemy import Column, Integer, ForeignKey, Index

from sqlalchemy.orm import relationship

from autowisp.database.data_model.base import DataModelBase

__all__ = ["StepDependencies"]


class StepDependencies(DataModelBase):
    """The table describing the prerequisites for a step to run"""

    __tablename__ = "step_dependencies"

    blocked_step_id = Column(
        Integer,
        ForeignKey("step.id"),
        doc="The step for which this prerequisite applies.",
    )
    blocked_image_type_id = Column(
        Integer,
        ForeignKey("image_type.id"),
        doc="The image type for which this prerequisite applies.",
    )
    blocking_step_id = Column(
        Integer,
        ForeignKey("step.id"),
        doc="The step which must be completed before the blocked step can "
        "begin.",
    )
    blocking_image_type_id = Column(
        Integer,
        ForeignKey("image_type.id"),
        doc="The image type for which the prerequisite step must be completed.",
    )

    __table_args__ = (
        Index(
            "dependency_key2",
            "blocked_step_id",
            "blocked_image_type_id",
            "blocking_step_id",
            "blocking_image_type_id",
        ),
    )

    blocked_step = relationship(
        "Step", primaryjoin="StepDependencies.blocked_step_id == Step.id"
    )
    blocked_imtype = relationship(
        "ImageType",
        primaryjoin="StepDependencies.blocked_image_type_id == ImageType.id",
    )
    blocking_step = relationship(
        "Step", primaryjoin="StepDependencies.blocking_step_id == Step.id"
    )
    blocking_imtype = relationship(
        "ImageType",
        primaryjoin="StepDependencies.blocking_image_type_id == ImageType.id",
    )

    def __str__(self):
        """Describe the dependency."""

        return (
            f"Applying {self.blocked_step.name} to {self.blocked_imtype.name} "
            " images requires completing"
            f" {self.blocking_step.name} on {self.blocking_imtype.name} images"
        )
