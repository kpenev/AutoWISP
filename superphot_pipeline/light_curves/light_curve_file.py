"""Define a class for working with light curve files."""

from superphot_pipeline.database.hdf5_file_structure import\
    HDF5FileDatabaseStructure

#Come from H5py.
#pylint: disable=too-many-ancestors
class LightCurveFile(HDF5FileDatabaseStructure):
    """Interface for working with the pipeline generated light curve files."""

    @classmethod
    def _product(cls):
        return 'light_curve'

    @classmethod
    def _get_root_tag_name(cls):
        """The name of the root tag in the layout configuration."""

        return 'LightCurve'

#pylint: enable=too-many-ancestors
