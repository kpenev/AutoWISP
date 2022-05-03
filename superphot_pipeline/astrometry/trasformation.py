"""Define class to apply sky-to-frame transformation stored in DR files."""

from functools import partial

import numpy

from superphot_pipeline import DataReductionFile
from superphot_pipeline.astrometry import map_projections
import fit_expression

class Transformation:
    """
    A class that applies transformation stored in DR files.

    Attributes:
        pre_projection(callable):    One of the map projections in
            superphot_pipeline.astrometry.map_projections that is applied first
            to transform RA, Dec -> xi, eta. The latter are then projected to
            x, y.

        evaluate_transformation(fit_expression.Interface):    Evaluator for the
            pre-projected catalog -> frame coordinates transformation.
    """

    @staticmethod
    def _create_projected_arrays(sources, save_intermediate, in_place):
        """Create a numpy structured array to hold the projected sources."""

        intermediate_dtype = [('xi', numpy.float64), ('eta', numpy.float64)]
        if in_place:
            projected = sources
        else:
            projected_dtype = [('x', numpy.float64), ('y', numpy.float64)]
            #pylint: enable=no-member
            if save_intermediate:
                projected_dtype.extend(intermediate_dtype)
            projected = numpy.empty(len(sources), dtype=projected_dtype)
        intermediate = (projected if save_intermediate
                        else numpy.empty(projected.shape, intermediate_dtype))
        return intermediate, projected


    def __init__(self, dr_fname=None, **dr_path_substitutions):
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
        self.evaluate_transformation = None
        self._coefficients = None
        if dr_fname is not None:
            with DataReductionFile(dr_fname, 'r') as dr_file:
                self.read_transformation(dr_file, **dr_path_substitutions)


    def read_transformation(self, dr_file, **dr_path_substitutions):
        """Read the transformation from the given DR file (already opened)."""

        pre_projection_name = (
            dr_file.get_attribute('skytoframe.sky_preprojection',
                                  **dr_path_substitutions)
            +
            '_projection'
        )
        pre_projection_center = dr_file.get_attribute('skytoframe.sky_center',
                                                      **dr_path_substitutions)
        self.pre_projection = partial(
            getattr(map_projections, pre_projection_name),
            RA=pre_projection_center[0],
            Dec=pre_projection_center[1]
        )

        self.evaluate_terms = fit_expression.Interface(
            dr_file.get_attribute('skytoframe.terms', **dr_path_substitutions)
        )
        self._coefficients = dr_file.get_attribute('skytoframe.coefficients',
                                                   **dr_path_substitutions)

    def __call__(self, sources, save_intermediate=False, in_place=False):
        """
        Return the projected positions of the given catalogue sources.

        Args:
            sources(structure numpy array or pandas.DataFrame):    The
                catalogue sources to project.

            save_intermediate(bool):    If True, the result includes the
                coordinate of the pre-projection, in addition to the final frame
                coordinates.

            in_place(bool):    If True, the input `sources` are updated with the
                projected coordinates (`sources` must allow setting entries for
                `x` and `y`  columns, also for `xi` and `eta` if
                `save_intermediate` is True).

        Returns:
            numpy structured array:
                The projected source positions with labels 'x', 'y', and
                optionally (if save_intermediate == True) the pre-projected
                coordinates `xi` and `eta`. If `in_place` is True, return None.
        """

        intermediate, projected = self._create_projected_arrays(
            sources,
            save_intermediate,
            in_place
        )
        self.pre_projection(sources, intermediate)
        terms = self.evaluate_terms(sources, intermediate)
        for index, coord in enumerate('xy'):
            projected[coord] = self._coefficients[index].dot(terms)

        return None if in_place else projected
