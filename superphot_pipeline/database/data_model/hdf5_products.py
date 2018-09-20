"""Define the hdf5_attributes table."""

from sqlalchemy import\
    Column,\
    Integer,\
    String,\
    TIMESTAMP
from sqlalchemy.orm import relationship

from superphot_pipeline.database.data_model.base import DataModelBase
#Pylint false positive due to quirky imports.
#pylint: disable=no-name-in-module
from superphot_pipeline.database.data_model.hdf5_structure_versions import\
    HDF5StructureVersion
#pylint: enable=no-name-in-module

__all__ = ['HDF5Product']

#The standard use of SQLAlchemy ORM requires classes with no public methods.
#pylint: disable=too-few-public-methods
class HDF5Product(DataModelBase):
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

    structure_versions = relationship(
        'HDF5StructureVersion',
        order_by=HDF5StructureVersion.hdf5_structure_version_id,
        back_populates='product'
    )
#pylint: enable=too-few-public-methods
