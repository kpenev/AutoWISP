"""Define the hdf5_datasets table."""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    Float,
    ForeignKey,
    Index,
    text,
)
from sqlalchemy.orm import relationship

from autowisp.database.data_model.base import DataModelBase

__all__ = ["HDF5DataSet"]


# The standard use of SQLAlchemy ORM requires classes with no public methods.
# pylint: disable=too-few-public-methods
class HDF5DataSet(DataModelBase):
    """The table describing how to include datasets in HDF5 files."""

    __tablename__ = "hdf5_datasets"

    hdf5_structure_version_id = Column(
        Integer,
        ForeignKey(
            "hdf5_structure_versions.id",
            onupdate="CASCADE",
            ondelete="RESTRICT",
        ),
        doc="Which structure version of which pipeline product is this "
        "element configuration for.",
    )
    pipeline_key = Column(
        String(100),
        nullable=False,
        doc="How is this dataset referred to by the pipeline.",
    )
    abspath = Column(
        String(1000),
        nullable=False,
        doc="The full absolute path to the dataset within the HDF5 file.",
    )
    dtype = Column(
        String(100),
        nullable=False,
        doc="The data type to use for this dataset. See h5py for possible "
        "values and their meanings.",
    )
    compression = Column(
        String(100),
        nullable=True,
        server_default=text("NULL"),
        doc="If not NULL, which compression filter to use when creating the "
        "dataset.",
    )
    compression_options = Column(
        String(1000),
        nullable=True,
        server_default=text("NULL"),
        doc="Any options to pass to the compression filter. For gzip, this is "
        "passed as int(compression_options).",
    )
    scaleoffset = Column(
        Integer,
        nullable=True,
        server_default=text("NULL"),
        doc="If not null, enable the scale/offset filter for this dataset with "
        "the specified precision.",
    )
    shuffle = Column(
        Boolean,
        nullable=False,
        server_default="0",
        doc="Should the shuffle filter be enabled?",
    )
    replace_nonfinite = Column(
        Float,
        nullable=True,
        server_default=text("NULL"),
        doc="For numeric datasets, if this is not NULL, any non-finite "
        "values are replaced by this value.",
    )
    description = Column(
        String(1000),
        nullable=False,
        doc="A brief description of what this attribute tracks.",
    )

    __table_args__ = (
        Index(
            "version_dataset",
            "hdf5_structure_version_id",
            "pipeline_key",
            unique=True,
        ),
    )

    structure_version = relationship(
        "HDF5StructureVersion", back_populates="datasets"
    )

    def __str__(self):
        """Human readable description."""

        return (
            str(self.id)
            + ":\n\t"
            + "\n\t".join(
                [
                    "structure version ID = "
                    + str(self.hdf5_structure_version_id),
                    "pipeline key = " + str(self.pipeline_key),
                    "|path| = " + repr(self.abspath),
                    "dtype = " + str(self.dtype),
                    "compression = " + str(self.compression),
                    "compression options = " + repr(self.compression_options),
                    "scale-offset = " + str(self.scaleoffset),
                    "shuffle = " + str(self.shuffle),
                    "replace non-finite = " + str(self.replace_nonfinite),
                    "timestamp = " + str(self.timestamp),
                ]
            )
        )


# pylint: enable=too-few-public-methods
