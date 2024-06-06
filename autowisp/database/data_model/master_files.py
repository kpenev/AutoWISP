"""Define the MasterFiles table for the pipeline."""

from typing import List

from sqlalchemy import\
    Column,\
    Index,\
    Integer,\
    String,\
    Boolean,\
    TIMESTAMP,\
    ForeignKey
from sqlalchemy.orm import relationship, Mapped


from autowisp.database.data_model.base import DataModelBase
from autowisp.database.data_model.condition import Condition
from autowisp.database.data_model.condition_expression import \
    ConditionExpression


__all__ = ['MasterType', 'MasterFile']


class MasterType(DataModelBase):
    """The table tracking the types of master files used by the pipeline."""

    __tablename__ = 'master_type'

    id = Column(
        Integer,
        primary_key=True,
        doc='A unique identifier for the master file type.'
    )
    name = Column(
        String(50),
        unique=True,
        nullable=False,
        doc='The name of the master file type.'
    )
    condition_id = Column(
        Integer,
        ForeignKey('condition.id'),
        nullable=False,
        doc='The collection of expression involving header keywords that must '
        'match between an image and a master for a master to be useable for '
        'calibrating that image.'
    )
    maker_step_id = Column(
        Integer,
        ForeignKey('step.id'),
        nullable=True,
        doc='The step which produces this type of master frames.'
    )
    maker_image_type_id = Column(
        Integer,
        ForeignKey('image_type.id'),
        nullable=True,
        doc='The image type to which the maker step is applied when creating '
        'masters of this type.'
    )
    maker_image_split_condition_id = Column(
        Integer,
        ForeignKey('condition.id'),
        nullable=True,
        doc='The collection of expression involving header keywords that must '
        'match between all images that are combined into a master in addition '
        'to the expressions specified by ``condition_id``.'
    )
    description = Column(
        String(1000),
        doc='A description of the master file type.'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When was this record last changed.'
    )

    Index("maker", "maker_step_id", "maker_image_type_id")

    master_files = relationship('MasterFile', back_populates='master_type')
    match_expressions: Mapped[List[ConditionExpression]] = relationship(
        secondary=Condition.__table__,
        primaryjoin="MasterType.condition_id == Condition.id",
        viewonly=True
    )
    split_expressions: Mapped[List[ConditionExpression]] = relationship(
        secondary=Condition.__table__,
        primaryjoin="MasterType.maker_image_split_condition_id == Condition.id",
        viewonly=True
    )

    def __str__(self):
        return f'({self.id}) {self.name}: {self.description} ({self.timestamp})'


class MasterFile(DataModelBase):
    """The table tracking master files generated and used by the pipeline."""

    __tablename__ = 'master_file'

    id = Column(
        Integer,
        primary_key=True,
        doc='A unique identifier for the master file.'
    )
    type_id = Column(
        Integer,
        ForeignKey('master_type.id'),
        nullable=False,
        doc='The type of master file.'
    )
    progress_id = Column(
        Integer,
        ForeignKey('image_processing_progress.id'),
        nullable=True,
        doc='The ImageProcessingProgress that generated this master, if any.'
    )
    filename = Column(
        String(1000),
        nullable=False,
        doc='The full path of the master file.'
    )
    enabled = Column(
        Boolean,
        nullable=False,
        default=True,
        doc='Can this master be used for calibrations?'
    )
    use_smallest = Column(
        String(1000),
        nullable=True,
        doc='Use the master of this type for which this expression evaluated '
        'against image header gives the smallest value'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When was this record last changed.'
    )
    master_type = relationship('MasterType', back_populates='master_files')

    def __str__(self):
        return (
            f'({self.id}) {self.filename} - {self.master_type.name}: '
            f'minimize {self.use_smallest} ({self.timestamp})'
        )
