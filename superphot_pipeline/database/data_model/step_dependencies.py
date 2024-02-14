"""Define class to specify dependencies between steps."""

from sqlalchemy import\
    Column,\
    Integer,\
    TIMESTAMP,\
    ForeignKey

from superphot_pipeline.database.data_model.base import DataModelBase

__all__= ['StepDependencies']


class StepDependencies(DataModelBase):
    """The table describing the prerequisites for a step to run"""


    __tablename__ = 'step_dependencies'

    blocked_step_id = Column(
        Integer,
        ForeignKey('step.id'),
        primary_key=True,
        doc='The step for which this prerequisite applies.'
    )
    blocked_image_type_id = Column(
        Integer,
        ForeignKey('image_type.id'),
        primary_key=True,
        doc='The image type for which this prerequisite applies.'
    )
    blocking_step_id = Column(
        Integer,
        ForeignKey('step.id'),
        primary_key=True,
        doc='The step which must be completed before the blocked step can '
        'begin.'
    )
    blocking_image_type_id = Column(
        Integer,
        ForeignKey('image_type.id'),
        primary_key=True,
        doc='The image type for which the prerequisite step must be completed.'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When was this record last changed.'
    )
