"""Define the data model for database tables defining HDF5 file structure."""

from sqlalchemy import\
    Column,\
    Integer,\
    String,\
    Boolean,\
    Float,\
    TIMESTAMP,\
    ForeignKey,\
    Index
from sqlalchemy.ext.declarative import declarative_base

#pylint false positive: this is actually a class name
#pylint: disable=invalid-name
DataModelBase = declarative_base()
#pylint: enable=invalid-name

#The standard use of SQLAlchemy ORM requires classes with no public methods.
#pylint: disable=too-few-public-methods

class HDF5DataSets(DataModelBase):
    """The table describing how to include datasets in HDF5 files."""

    __tablename__ = 'hdf5_datasets'

    hdf5_dataset_id = Column(
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
        String,
        nullable=False,
        doc='How is this dataset referred to by the pipeline.'
    )
    abspath = Column(
        String,
        nullable=False,
        doc='The full absolute path to the dataset within the HDF5 file.'
    )
    dtype = Column(
        String,
        nullable=False,
        doc='The data type to use for this dataset. See h5py for possible '
        'values and their meanings.'
    )
    compression = Column(
        String,
        nullable=True,
        server_defalut='NULL',
        doc='If not NULL, which compression filter to use when creating the '
        'dataset.'
    )
    compression_options = Column(
        String,
        nullable=True,
        server_defalut='NULL',
        doc='Any options to pass to the compression filter. For gzip, this is '
        'passed as int(compression_options).'
    )
    scaleoffset = Column(
        Integer,
        nullable=True,
        server_defalut='NULL',
        doc='If not null, enable the scale/offset filter for this dataset with '
        'the specified precision.'
    )
    shuffle = Column(
        Boolean,
        nullable=False,
        server_default='0',
        doc='Should the shuffle filter be enabled?'
    )
    replace_nonfinite = Column(
        Float,
        nullable=True,
        server_defalut='NULL',
        doc='For floating point datasets, if this is not NULL, any non-finite '
        'values are replaced by this value.'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When was this record last changed.'
    )

    __table_args__ = (
        Index('version_dataset',
              'hdf5_structure_version_id',
              'pipeline_key',
              unique=True),
    )


class HDF5Attributes(DataModelBase):
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
        String,
        nullable=False,
        doc='How is this dataset referred to by the pipeline.'
    )
    parent = Column(
        String,
        nullable=False,
        doc='The full absolute path to the group or dataset to add this '
        'attribute to.'
    )
    name = Column(
        String,
        nullable=False,
        doc='The name to give to use for this attribute within the HDF5 file.'
    )
    dtype = Column(
        String,
        nullable=False,
        doc='The data type to use for this dataset. See h5py for possible '
        'values and their meanings.'
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


class HDF5Products(DataModelBase):
    """The types of pipeline products stored as HDF5 files."""

    __tablename__ = 'hdf5_products'

    hdf5_product_id = Column(
        Integer,
        primary_key=True,
        doc='An identifier for each HDF5 product type.'
    )
    pipeline_key = Column(
        String(100),
        index=True,
        unique=True,
        doc='How is this product referred to in the pipeline (e.g. '
        '"datareduction" or "lightcurve"'
    )
    description = Column(
        String(100),
        nullable=False,
        doc='A description of the product type.'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When was this record last changed.'
    )

class HDF5StructureVersions(DataModelBase):
    """The versions of structures for the HDF5 pipeline products."""

    __tablename__ = 'hdf5_structure_versions'

    hdf5_structure_version_id = Column(
        Integer,
        primary_key=True,
        doc='A unique identifier for the given product/version combination.'
    )
    hdf5_product_id = Column(
        Integer,
        ForeignKey('hdf5_products.hdf5_product_id',
                   onupdate='CASCADE',
                   ondelete='RESTRICT'),
        doc='The type of pipeline product this structure configuration version '
        'is for.'
    )
    version = Column(
        Integer,
        nullable=False,
        doce='An identifier for distinguishing the separate configuration '
        'versions of a single pipeline product type.'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When was this record last changed.'
    )

    __table_args__ = (
        Index('product_version',
              'hdf5_product_id',
              'version',
              unique=True),
    )

#pylint: enable=too-few-public-methods
