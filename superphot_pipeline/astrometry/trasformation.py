"""Define class to apply sky-to-frame transformation stored in DR files."""

from superphot_pipeline import DataReductionFile
from superphot_pipeline.astrometry import map_projections

class Transformation:
    """
    A class that applies transformation stored in DR files.

    Attributes:
        pre_projection(callable):    One of the map projections in
            superphot_pipeline.astrometry.map_projections that is applied first
            to transform RA, Dec -> xi, eta. The latter are then projected to
            x, y.
    """

    def __init__(self, dr_fname=None):
        """
        Prepare to apply the transformation stored in the given DR file.

        Args:
            dr_fname(str):    The filename of the data reduction file to read a
                transformation from. If not specified, read_transformation()
                must be called before using this class.

        Returns:
            None
        """

        self.pre_projection = None
        if dr_fname is not None:
            with DataReductionFile(dr_fname, 'r') as dr_file:
                self.read_transformation(dr_file)


    def read_transformation(self, dr_file):
        """Read the transformation from the given DR file (already opened)."""
