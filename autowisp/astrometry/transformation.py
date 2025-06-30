#!/usr/bin/env python3

"""Define class to apply sky-to-frame transformation stored in DR files."""

from functools import partial

import numpy
from numpy.lib.recfunctions import unstructured_to_structured
from scipy.optimize import root
from configargparse import ArgumentParser, DefaultsFormatter

from autowisp import DataReductionFile
from autowisp.astrometry import map_projections
from autowisp import fit_expression


class Transformation:
    """
    A class that applies transformation stored in DR files.

    Attributes:
        pre_projection(callable):    One of the map projections in
            autowisp.astrometry.map_projections that is applied first
            to transform RA, Dec -> xi, eta. The latter are then projected to
            x, y.

        evaluate_transformation(fit_expression.Interface):    Evaluator for the
            pre-projected catalog -> frame coordinates transformation.
    """

    @staticmethod
    def _create_projected_arrays(sources, save_intermediate, in_place):
        """Create a numpy structured array to hold the projected sources."""

        intermediate_dtype = [("xi", numpy.float64), ("eta", numpy.float64)]
        if in_place:
            projected = sources
        else:
            projected_dtype = [("x", numpy.float64), ("y", numpy.float64)]
            # pylint: enable=no-member
            if save_intermediate:
                projected_dtype.extend(intermediate_dtype)
            projected = numpy.empty(
                len(numpy.atleast_1d(sources["RA"])), dtype=projected_dtype
            )
        intermediate = (
            projected
            if save_intermediate
            else numpy.empty(projected.shape, intermediate_dtype)
        )
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
        self.pre_projection_center = None
        self.inverse_pre_projection = None
        self.evaluate_transformation = None
        self.evaluate_terms = None
        self._coefficients = None
        self._term_indices = None
        if dr_fname is not None:
            with DataReductionFile(dr_fname, "r") as dr_file:
                self.read_transformation(dr_file, **dr_path_substitutions)

    def read_transformation(self, dr_file, **dr_path_substitutions):
        """Read the transformation from the given DR file (already opened)."""

        self.set_transformation(
            pre_projection_name=(
                dr_file.get_attribute(
                    "skytoframe.cfg.sky_preprojection", **dr_path_substitutions
                )
                + "_projection"
            ),
            pre_projection_center=dr_file.get_attribute(
                "skytoframe.sky_center", **dr_path_substitutions
            ),
            terms_expression=dr_file.get_attribute(
                "skytoframe.terms", **dr_path_substitutions
            ),
            coefficients=dr_file.get_dataset(
                "skytoframe.coefficients", **dr_path_substitutions
            ),
        )

    def set_transformation(
        self,
        pre_projection_center,
        terms_expression,
        coefficients,
        pre_projection_name="tan_projection",
    ):
        """Set the projection to use."""

        self.pre_projection_center = pre_projection_center
        self.pre_projection = partial(
            getattr(map_projections, pre_projection_name),
            RA=pre_projection_center[0],
            Dec=pre_projection_center[1],
        )
        self.inverse_pre_projection = partial(
            getattr(map_projections, "inverse_" + pre_projection_name),
            RA=pre_projection_center[0],
            Dec=pre_projection_center[1],
        )
        self.evaluate_terms = fit_expression.Interface(terms_expression)
        self._term_indices = {
            term: index
            for index, term in enumerate(
                self.evaluate_terms.get_term_str_list()
            )
        }
        self._coefficients = coefficients

    def __call__(self, sources, save_intermediate=False, in_place=False):
        """
        Return the projected positions of the given catalogue sources.

        Args:
            sources(structured numpy array or pandas.DataFrame):    The
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
            sources, save_intermediate, in_place
        )
        self.pre_projection(sources, intermediate)

        terms = self.evaluate_terms(sources, intermediate)
        for index, coord in enumerate("xy"):
            projected[coord] = self._coefficients[index].dot(terms)

        return None if in_place else projected

    def inverse(self, x, y, result="equatorial", **source_properties):
        """Return the sky coordinates of the given source."""

        def projection_error(xi_eta):
            """Apply the (xi, eta, source preperties) -> (x, y) projection."""

            source_properties["xi"], source_properties["eta"] = xi_eta
            terms = self.evaluate_terms(source_properties)
            return numpy.array(
                [
                    float(coef.dot(terms)) - target
                    for coef, target in zip(self._coefficients, [x, y])
                ]
            )

        assert result in ("equatorial", "pre_projected", "both")
        offset_term = self._term_indices["1"]
        xi_term = self._term_indices["(xi)"]
        eta_term = self._term_indices["(eta)"]
        xi_eta_guess = numpy.linalg.solve(
            numpy.array(
                [
                    [
                        self._coefficients[0][xi_term],
                        self._coefficients[0][eta_term],
                    ],
                    [
                        self._coefficients[1][xi_term],
                        self._coefficients[1][eta_term],
                    ],
                ]
            ),
            numpy.array(
                [
                    x - self._coefficients[0][offset_term],
                    y - self._coefficients[1][offset_term],
                ]
            ),
        )
        solution = root(projection_error, xi_eta_guess)
        assert solution.success
        pre_projected = unstructured_to_structured(
            solution.x, names=["xi", "eta"]
        )
        if result == "pre_projected":
            return pre_projected

        equatorial = numpy.empty(1, dtype=[("RA", float), ("Dec", float)])
        self.inverse_pre_projection(equatorial, pre_projected)

        if result == "equatorial":
            return equatorial

        return pre_projected, equatorial


def parse_command_line():
    """Return the configuration for running this module as a script."""

    parser = ArgumentParser(
        description="Provide command line interface to the transformation in a "
        "DR file",
        default_config_files=[],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False,
    )
    parser.add_argument(
        "dr_fname",
        metavar="DR_FILE",
        help="The data reduction file to read the transformation from.",
    )
    parser.add_argument(
        "--skytoframe_version",
        type=int,
        default=0,
        help="The version of the sky-to-frame transformation to use.",
    )
    parser.add_argument(
        "--project",
        metavar=("RA", "Dec"),
        nargs=2,
        type=float,
        default=None,
        help="Sky coordinates to project.",
    )
    parser.add_argument(
        "--inverse",
        metavar=("x", "y"),
        nargs=2,
        type=float,
        default=None,
        help="Frame coordinates for which to find the corresponding RA, Dec .",
    )
    return parser.parse_args()


def main(config):
    """Avoid global variables."""

    transformation = Transformation(
        config.dr_fname, skytoframe_version=config.skytoframe_version
    )
    if config.project is not None:
        projected = transformation(
            {"RA": config.project[0], "Dec": config.project[1]}
        )
        print(
            f"RA: {config.project[0]!r}, Dec {config.project[1]!r} -> "
            f"({projected!r})"
        )

    if config.inverse is not None:
        unprojected = transformation.inverse(
            config.inverse[0], config.inverse[1], result="both"
        )
        print(
            f"({config.inverse[0]!r}, {config.inverse[1]!r}) -> "
            f"{unprojected!r}"
        )


if __name__ == "__main__":
    main(parse_command_line())
