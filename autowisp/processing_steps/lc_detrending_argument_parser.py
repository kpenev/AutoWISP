"""Implement shared argument parsing for LC detrending processing steps."""

import re
from os import path

from asteval import Interpreter

from autowisp.processing_steps.manual_util import ManualStepArgumentParser


class DetrendDatasetIter:
    """Iterate over the detrending datasets specified in a cmdline argument."""

    def __init__(self, argument):
        """Set up the iterator for the given argument."""

        self._splits = argument.split(";")

    def __iter__(self):
        return self

    def __next__(self):
        """Return the next dataset specification."""

        if len(self._splits) == 0:
            raise StopIteration

        result = self._splits.pop(0)
        while len(self._splits) > 0 and self._splits[0] == "":
            self._splits.pop(0)
            result += ";"
            if len(self._splits) > 0:
                result += self._splits.pop(0)


def _split_delimited_string(argument, separator):
    """Split the detrending dataset argument at `;`, handling `;;`."""

    if argument is not None:
        splits = argument.split(separator)
        while splits:
            result = splits.pop(0)
            while splits and splits[0] == "":
                splits.pop(0)
                result += separator
                if splits:
                    result += splits.pop(0)
            yield result.strip()


def _parse_substitutions(substitutions_str_iter):
    """Generator of all possible combination of substitutions."""

    substitution_rex = re.compile(
        r"^(?P<key>\w+)\s*(?P<type>(=|in))\s*(?P<value>\S.*)$"
    )
    try:
        parsed_substitution = substitution_rex.match(
            next(substitutions_str_iter)
        )
    except StopIteration:
        yield {}
        return
    assert parsed_substitution
    if parsed_substitution["type"] == "in":
        values = Interpreter()(parsed_substitution["value"])
    else:
        values = [Interpreter()(parsed_substitution["value"])]
    for result in _parse_substitutions(substitutions_str_iter):
        for val in values:
            result[parsed_substitution["key"]] = val
            yield dict(result)


def _parse_fit_datasets(argument):
    """Parse the fit datasets argument (see help for details)."""

    dset_specfication_rex = re.compile(
        r"^(?P<from>[\w.]+)"
        r"\s*->\s*"
        r"(?P<to>[\w.]+)"
        r"\s*"
        r"(:\s*(?P<substitutions>.*))?"
        r"$"
    )

    result = []
    for specification in _split_delimited_string(argument, ";"):
        parsed_dset = dset_specfication_rex.match(specification)
        assert parsed_dset
        result.extend(
            [
                (
                    parsed_dset["from"],
                    substitution,
                    parsed_dset["to"],
                )
                for substitution in _parse_substitutions(
                    _split_delimited_string(parsed_dset["substitutions"], "&")
                )
            ]
        )
    return result


def _parse_lc_variables(argument):
    """Parse the EPD variables command line argument (see help for details)."""

    var_specfication_rex = re.compile(
        r"^(?P<varname>\w+)"
        r"\s*=\s*"
        r"(?P<dset>[\w.]+)"
        r"\s*"
        r"(:\s*(?P<substitutions>.*))?"
        r"$"
    )
    result = []
    for specification in _split_delimited_string(argument, ";"):
        print(f"Parsing: {specification!r}.")
        parsed_var = var_specfication_rex.match(specification)
        assert parsed_var
        substitutions = list(
            _parse_substitutions(
                _split_delimited_string(parsed_var["substitutions"], "&")
            )
        )
        assert len(substitutions) == 1
        result.append(
            (parsed_var["varname"], (parsed_var["dset"], substitutions[0]))
        )
    return result


class LCDetrendingArgumentParser(ManualStepArgumentParser):
    """Boiler plate handling of LC detrending command line arguments."""

    @staticmethod
    def add_transit_parameters(
        parser,
        *,
        timing=True,
        duration=True,
        geometry="",
        limb_darkening=False,
        fit_flags=False,
    ):
        """
        Add command line parameters to the current parser to specify a transit.

        Args:
            timing(bool):    Should arguments specifying the timing of the
                transit be included, i.e. period, ephemerides, duration?

            duration(bool):    Sholud a potentially redundant argument for
                transit duration be added?

            geometry(bool or str):     Should arguments specifying the geometry
                of the transit be included. If the value converts to False,
                nothing is included. Otherwise this argument should be either:

                    * 'circular': i.e. include radius ratio, inclination, and
                      semimajor axis.

                    * 'eccentric': i.e. in additition to the above, include
                      eccentricity and argument of periastron.

            limb_darkening(bool):    Should arguments for specifying the stellar
                limb-darkening be included?

        Returns:
            None
        """

        assert (not geometry) or (geometry in ["circular", "eccentric"])

        if timing:
            parser.add_argument(
                "--mid-transit",
                type=float,
                default=2455867.402743,
                help="The time of mit-transit in BJD. Default: %(default)s "
                "(from https://ui.adsabs.harvard.edu/abs/2019AJ....157...82W).",
            )
            parser.add_argument(
                "--period",
                type=float,
                default=2.15000820,
                help="The orbital period (used to convert time to phase). "
                "Default: %(default)s (from "
                "https://ui.adsabs.harvard.edu/abs/2019AJ....157...82W).",
            )
        if duration:
            parser.add_argument(
                "--transit-duration",
                type=float,
                default=3.1205,
                help="The transit duration in hours. Default: %(default)s (from"
                " https://ui.adsabs.harvard.edu/abs/2019AJ....157...82W).",
            )

        if geometry:
            parser.add_argument(
                "--radius-ratio",
                type=float,
                default=0.14886,
                help="The planet to star radius ratio. Default: %(default)s "
                "(HAT-P-32).",
            )
            parser.add_argument(
                "--inclination",
                type=float,
                default=88.98,
                help="The inclination angle between the orbital angular "
                "momentum and the line of sight in degrees. "
                "Default: %(default)s (HAT-P-32).",
            )
            parser.add_argument(
                "--scaled-semimajor",
                type=float,
                default=5.344,
                help="The ratio of the semimajor axis to the stellar radius. "
                "Default: %(default)s (HAT-P-32).",
            )
            if geometry == "eccentric":
                parser.add_argument(
                    "--eccentricity",
                    "-e",
                    type=float,
                    default=0.0,
                    help="The orbital eccentricity to assume for the transiting"
                    " planet. Default: %(default)s.",
                )
                parser.add_argument(
                    "--periastron",
                    type=float,
                    default=0.0,
                    help="The argument of periastron. Default: %(default)s.",
                )

        if limb_darkening:
            parser.add_argument(
                "--limb-darkening",
                nargs=2,
                type=float,
                default=(0.316, 0.303),
                help="The limb darkening coefficients to assume for the star "
                "(quadratic model). Default: %(default)s (HAT-P-32).",
            )

        if fit_flags:
            choices = []
            if timing:
                choices.extend(["mid_transit", "period"])
            if geometry:
                choices.extend(["depth", "inclination", "semimajor"])
                if geometry == "eccentric":
                    choices.extend(["eccentricity", "periastron"])
            if limb_darkening:
                choices.extend(["limbdark"])
            parser.add_argument(
                "--mutable-transit-params",
                nargs="*",
                choices=choices,
                default=[],
                help="List the transit parameters for which best-fit values "
                "should be found rather than assuming they are fixed at what is"
                " specified on the command line. Default: %(default)s.",
            )

    @staticmethod
    def _add_epd_arguments(parser):
        """Add parameters required for EPD only."""

        parser.add_argument(
            "--epd-terms-expression",
            default="O3{1/cos(z)}",
            type=str,
            help="A fitting terms expression involving only variables from "
            "`epd_variables` which expands to the various terms to use "
            "in a linear least squares EPD correction."
            "Default: %(default)s",
        )
        parser.add_argument(
            "--fit-weights",
            default=None,
            type=str,
            help="An expression involving only variables from `epd_variables` "
            "which should evaluate to the weights to use per LC point in a "
            "linear least squares EPD correction. If left unspecified, no "
            "weighting is performed.",
        )
        parser.add_argument(
            "--skip-outlier-prerejection",
            action="store_false",
            dest="pre_reject_outliers",
            default=True,
            help="If passed the initial rejection of outliers before the fit "
            "begins is not performed.",
        )

    @staticmethod
    def _add_tfa_arguments(parser):
        """Add parameters required for TFA only."""

        parser.add_argument(
            "--saturation-magnitude",
            type=float,
            default=7.0,
            help="The magnitude at which sources start to saturate. "
            "Default: %(default)s",
        )
        parser.add_argument(
            "--tfa-forbid-saturated-templates",
            dest="allow_saturated_templates",
            action="store_false",
            default=True,
            help="If passed, saturated stars are precluded from being selected "
            "as templates.",
        )
        parser.add_argument(
            "--tfa-mag-rms-dependence-order",
            type=int,
            default=2,
            help="The maximum order of magnitude to include in the fit for "
            "typical rms vs magnitude. Default: %(default)s.",
        )
        parser.add_argument(
            "--tfa-mag-rms-outlier-threshold",
            type=float,
            default=4,
            help="Stars are not allowed to be in the template if their RMS is "
            "more than this many sigma away from the mag-rms fit. This is also "
            "the threshold used for rejecting outliers when doing the iterative"
            " fit for the rms as a function of magnutude. Default: %(default)s.",
        )
        parser.add_argument(
            "--tfa-mag-rms-max-rej-iter",
            type=int,
            default=10,
            help="The maximum number of rejection fit iterations to do when "
            "deriving the rms(mag) dependence. Default: %(default)s.",
        )
        parser.add_argument(
            "--tfa-max-rms",
            type=float,
            default=0.05,
            help="Stars are allowed to be in the template only if their RMS is "
            "no larger than this. Default: %(default)s.",
        )
        parser.add_argument(
            "--tfa-faint-mag-limit",
            type=float,
            default=10.0,
            help="Stars fainter than this cannot be template stars. Default: "
            "%(default)s.",
        )
        parser.add_argument(
            "--tfa-min-observations-quantile",
            type=float,
            default=0.5,
            help="The minimum number of observations required of template stars"
            " is the smaller of this quantile among the input collection of "
            "stars and the number per --tfa-min-observations-fraction.",
        )
        parser.add_argument(
            "--tfa-min-observations-fraction",
            type=float,
            default=0.8,
            help="The minimum number of observations required of template stars"
            " is the smaller of this number times the maximum number of "
            "observations of any star and the number per "
            "--tfa-min-observations-quantile.",
        )
        parser.add_argument(
            "--tfa-sqrt-num-templates",
            type=int,
            default=8,
            help="The number of template stars is the square of this number. "
            "Default: %(default)s.",
        )
        parser.add_argument(
            "--tfa-observation-id",
            type=str,
            nargs="+",
            default=("fitsheader.fnum",),
            help="The datasets to use for matching observations across light "
            "curves. For example, the following works for HAT: "
            "fitseader.cfg.stid fitsheader.cfg.cmpos fitsheader.fnum.",
        )
        parser.add_argument(
            "--tfa-selected-plots",
            type=str,
            default="tfa_template_selection_%(plot_id)s_phot%(phot_index)s.eps",
            help="Optional template for naming plots showing the template "
            "selection in action. If not specified, no such plots are "
            "generated. Should include %%(plot_id)s and %%(phot_index)d "
            "substitutions.",
        )
        parser.add_argument(
            "--epd-statistics-fname",
            default="epd_statistics.txt",
            help="The statistics filename for the results of the EPD fit. "
            "Default: %(default)s",
        )
        parser.add_argument(
            "--tfa-radius-splits",
            nargs="*",
            type=float,
            default=[],
            help="The threshold radius values where to split sources into "
            "groups. By default, no splitting by radius is done.",
        )
        parser.add_argument(
            "--tfa-mag-split-source-count",
            type=int,
            default=None,
            help="If passed, after spltting by radius (if any), sources are "
            "further split into groups by magnitude such that each group "
            "contains at least this many sources. By default, no splitting by "
            "magnitude is done.",
        )

    def __init__(
        self,
        mode,
        description,
        *,
        add_reconstructive=True,
        convert_to_dict=True,
        input_type="lc",
    ):
        """
        Initialize the parser with options common to all LC detrending steps.

        Args:
            See ManualStepArgumentParser.__init__().

        Returns:
            None
        """

        self._mode = mode.lower()
        super().__init__(
            input_type=input_type,
            description=description,
            allow_parallel_processing=self._mode in ["epd", "tfa"],
            convert_to_dict=convert_to_dict,
            add_lc_fname_arg=(self._mode == "tfa"),
        )

        if self._mode != "epd":
            self.add_argument(
                "--single-photref-dr-fname",
                default=None,
                help="The filename of the single photometric reference DR file."
                " Used for string substitutions of command line arguments.",
            )
        self.add_argument(
            "--variables",
            type=_parse_lc_variables,
            default=(
                "sphotref = apphot.magfit.cfg.single_photref : "
                "aperture_index = 0"
                + (
                    (
                        "; x = srcproj.columns : "
                        "srcproj_column_name = 'x' & srcproj_version = 0; "
                        "y = srcproj.columns : "
                        "srcproj_column_name = 'y' & srcproj_version = 0; "
                        "bg = bg.value; "
                        "z = skypos.zenith_distance; "
                        "S = srcextract.psf_map.eval: "
                        "srcextract_psf_param = 's'"
                    )
                    if self._mode == "epd"
                    else ""
                )
            ),
            help="Define variables"
            "to be used in --lc-points-filter-expression, "
            "--epd-terms-expression, --fit-weights, etc. Should be "
            "formatted as a `;` separated list of <varname> = <dataset key>"
            " [: <substitutions>]. For example: "
            '"x = srcproj.columns : src_proj_column_name = x  & '
            "srcproj_version = 0; y = srcproj.columns : "
            'src_proj_column_name = y", defines `x` and `y` variables that '
            "correspond to the projected x and y positions of the source in"
            " the frame.",
        )

        self.add_argument(
            "--lc-points-filter-expression",
            default=None,
            help="An expression using variables` which evaluates to either"
            " True or False indicating if a given point in the lightcurve "
            "should be fit and corrected. Default: %(default)s",
        )
        self.add_argument(
            f"--{self._mode[:3]!s}-datasets",
            type=_parse_fit_datasets,
            default=None,
            help="A `;` separated list of the datasets to detrend. Each entry "
            "should be formatted as: `<input-key> -> <output-key> "
            "[: <substitution> (= <value>| in <expression>) "
            "[& <substitution> (= <value> | in <expression>) ...]]. "
            'For example: `"apphot.magfit.magnitude -> '
            "apphot.epd.magnitude : magfit_iteration = 5 & aperture_index in "
            'range(39)"`. White space is ignored. Literal `;` and `&` can be '
            "used in the specification of a dataset as `;;` and `&&` "
            "respectively. Configurations of how the fitting was done and the "
            "resulting residual and non-rejected points are added to "
            "configuration datasets generated by removing the tail of the "
            'destination and adding `".cfg." + <parameter name>` for '
            "configurations and just ` + <parameter name>` for "
            "fitting statistics. For example, if the output dataset key is"
            '`"shapefit.epd.magnitude"`, the configuration datasets will look'
            'like `"shapefit.epd.cfg.fit_terms"`, and'
            '`"shapefit.epd.residual"`.',
        )
        self.add_argument(
            "--detrend-reference-avg",
            default="nanmedian",
            help="How to average magnitudes to define reference point for "
            "calculating residuals.",
        )
        self.add_argument(
            "--detrend-error-avg",
            default="nanmedian",
            help="How to average fitting residuals for outlier rejection and "
            "perforamnce reporting.",
        )
        self.add_argument(
            "--detrend-rej-level",
            type=float,
            default=5.0,
            help="How far away from the fit should a point be before "
            "it is rejected in units of detrend_error_avg. Default: %(default)s",
        )
        self.add_argument(
            "--detrend-max-rej-iter",
            type=int,
            default=20,
            help="The maximum number of rejection/re-fitting iterations to "
            "perform. If the fit has not converged by then, the latest "
            "iteration is accepted. Default: %(default)s",
        )
        if add_reconstructive:
            self.add_argument(
                "--target-id",
                default=None,
                help="The lightcurve of the given source (any one of the "
                "catalogue identifiers stored in the LC file) will be fit using"
                " reconstructive detrending, starting the transit parameter fit"
                " with the values supplied, and fitting for the values allowed "
                "to vary. If not specified all LCs are fit in "
                "non-reconstructive way.",
            )
            self.add_transit_parameters(
                self,
                timing=True,
                geometry="circular",
                limb_darkening=True,
                fit_flags=True,
            )

        self.add_argument(
            "--detrending-catalog",
            "--detrending-catalogue",
            "--cat",
            default=None,
            help="The name of a catalogue file containing the sources in the "
            "frame. Used only to generate performance statistics reports. If "
            "not specified, performance is not reported.",
        )
        if self._mode.endswith("stat"):
            self.add_argument(
                f"--{self._mode[:3]}-statistics-fname",
                default=f"{self._mode[:3]}_statistics.txt",
                help="The statistics filename for the results of the "
                f"{self._mode[:3].upper()} fit.",
            )
        self.add_argument(
            "--magnitude-column",
            default="phot_g_mean_mag",
            help="The magnitude column to use for the performance statistics. "
            "Default: %(default)s",
        )

        if self._mode in ["epd", "tfa"]:
            getattr(self, f"_add_{self._mode}_arguments")(self)
