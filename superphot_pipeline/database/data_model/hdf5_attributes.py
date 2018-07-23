"""Define the hdf5_attributes table."""

from sqlalchemy import\
    Column,\
    Integer,\
    String,\
    TIMESTAMP,\
    ForeignKey,\
    Index
from sqlalchemy.orm import relationship

from superphot_pipeline.database.data_model.base import DataModelBase

__all__ = ['HDF5Attribute']

#The standard use of SQLAlchemy ORM requires classes with no public methods.
#pylint: disable=too-few-public-methods
class HDF5Attribute(DataModelBase):
    """The table describing how to add attributes in HDF5 files."""

    __tablename__ = 'hdf5_attributes'

    hdf5_attribute_id = Column(
        Integer,
        primary_key=True,
        doc='A unique identifier of each dataset.'
    )
    hdf5_structure_version_id = Column(
        Integer,
        ForeignKey('hdf5_structure_versions.hdf5_structure_version_id',
                   onupdate='CASCADE',
                   ondelete='RESTRICT'),
        doc='Which structure version of which pipeline product is this '
        'element configuration for.'
    )
    pipeline_key = Column(
        String(100),
        nullable=False,
        doc='How is this dataset referred to by the pipeline.'
    )
    parent = Column(
        String(1000),
        nullable=False,
        doc='The full absolute path to the group or dataset to add this '
        'attribute to.'
    )
    name = Column(
        String(1000),
        nullable=False,
        doc='The name to give to use for this attribute within the HDF5 file.'
    )
    dtype = Column(
        String(100),
        nullable=False,
        doc='The data type to use for this dataset. See h5py for possible '
        'values and their meanings.'
    )
    description = Column(
        String(1000),
        nullable=False,
        doc='A brief description of what this attribute tracks.'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When was this record last changed.'
    )

    __table_args__ = (
        Index('version_attribute',
              'hdf5_structure_version_id',
              'pipeline_key',
              unique=True),
    )

    structure_version = relationship('HDF5StructureVersion',
                                     back_populates='attributes')

    def __str__(self):
        """Print the contents of the given attribute."""

        return (
            str(self.hdf5_attribute_id) + ':\n\t'
            +
            '\n\t'.join([
                'structure version ID = ' + str(self.hdf5_structure_version_id),
                'pipeline key = ' + str(self.pipeline_key),
                'parent = ' + repr(self.parent),
                'name = ' + repr(self.name),
                'dtype = ' + str(self.dtype),
                'description = ' + repr(self.description),
                'timestamp = ' + str(self.timestamp)
            ])
        )

#pylint: enable=too-few-public-methods
