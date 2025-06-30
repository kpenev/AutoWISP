"""Collection of functions used by many processing steps."""

import logging
from sys import argv
from os import path

import numpy
from astropy.io import fits

from configargparse import ArgumentParser, DefaultsFormatter


class ManualStepArgumentParser(ArgumentParser):
    """Incorporate boiler plate handling of command line arguments."""

    def _add_version_args(self, components):
        """Add arguments to select versions of the given components."""

        version_arg_help = {
            "srcextract": "The version of the extracted sources to use/create.",
            "catalogue": (
                "The version of the input catalogue of sources in the DR file "
                "to use/create."
            ),
            "skytoframe": (
                "The vesrion of the astrometry solution in the DR file."
            ),
            "srcproj": (
                "The version of the datasets containing projected photometry "
                "sources to use/create."
            ),
            "background": (
                "The version identifier of background measurements to "
                "use/create."
            ),
            "shapefit": (
                "The version identifier of PSF/PRF map fit to use/create."
            ),
            "apphot": (
                "The version identifier of aperture photometry to use/create."
            ),
            "magfit": ("The version of magnitude fitting to use/create."),
        }
        for comp in components:
            self.add_argument(
                "--" + comp + "-version",
                type=int,
                default=0,
                help=version_arg_help[comp],
            )

    def _add_catalog_args(self, catalog_config):
        """Add arguments to specify a catalog query."""

        prefix = catalog_config["prefix"]
        self.add_argument(
            f"--{prefix}-catalog",
            f"--{prefix}-catalogue",
            "--cat",
            default=catalog_config.get("fname", "Gaia/{checksum:s}.fits"),
            help="A file containing (approximately) all the same stars that "
            "were extracted from the frame for the area of the sky covered by "
            "the image. It is perferctly fine to include a larger area of sky "
            "and fainter brightness limit. Different brightness limits can then"
            " be imposed for each color channel using the ``--catalog-filter`` "
            "argument. If the file does not exist one is automatically "
            "generated to cover an area larger than the field of view by "
            "``--catalog-safety-factor``, centered on the (RA * cos(Dec), Dec) "
            "of the frame rounded to ``--catalog-pointing-precision``, and to "
            "have magnitude range set by. The filename can be a format string "
            "which will be substituted with the any header keywords or "
            "configuration for the query. It may also include ``{checksum}`` "
            "which will be replaced with the MD5 checksum of the parameters "
            "defining the query.",
        )
        self.add_argument(
            f"--{prefix}-catalog-magnitude-expression",
            f"--{prefix}-catalogue-magnitude-expression",
            "--cat-mag-expression",
            default=catalog_config.get(
                "magnitude_expression", "phot_g_mean_mag"
            ),
            help="An expression involving the catalog columns that correlates "
            "as closely as possible with the brightness of the star in the "
            "images in units of magnitude. Only relevant if the catalog does "
            "not exist.",
        )
        self.add_argument(
            f"--{prefix}-catalog-max-magnitude",
            f"--{prefix}-catalogue-max-magnitude",
            "--cat-max-mag",
            type=float,
            default=catalog_config.get("max_magnitude", 12.0),
            help="The faintest magnitude to include in the catalog.",
        )
        self.add_argument(
            f"--{prefix}-catalog-min-magnitude",
            f"--{prefix}-catalogue-min-magnitude",
            "--cat-min-mag",
            type=float,
            default=catalog_config.get("min_magnitude"),
            help="The brightest magnitude to include in the catalog.",
        )
        self.add_argument(
            f"--{prefix}-catalog-pointing-precision",
            f"--{prefix}-catalogue-pointing-precision",
            "--cat-pointing-precision",
            type=float,
            default=catalog_config.get("pointing_precision", 1.0),
            help="The precision with which to round the center of the frame to "
            "determine the center of the catalog to use in degrees. The catalog"
            " FOV is also expanded by this amount to ensure coverage",
        )
        self.add_argument(
            f"--{prefix}-catalog-fov-precision",
            f"--{prefix}-catalogue-fov-precision",
            "--cat-fov-precision",
            type=float,
            default=catalog_config.get("fov_precision", 1.0),
            help="The precision with which to round the center of the frame to "
            "determine the center of the catalog to use in degrees.",
        )
        self.add_argument(
            f"--{prefix}-catalog-filter",
            f"--{prefix}-catalogue-filter",
            "--cat-filter",
            metavar=("CHANNEL:EXPRESSION"),
            type=lambda e: e.split(":"),
            action="append",
            default=catalog_config.get("filter"),
            help="An expression to evaluate for each catalog source to "
            "determine if the source should be used for astrometry of a given "
            "channel. If filter for a given channel is not specified, the full "
            "catalog is used for that channel.",
        )
        self.add_argument(
            f"--{prefix}-catalog-epoch",
            f"--{prefix}-catalogue-epoch",
            "--cat-epoch",
            type=str,
            default=catalog_config.get(
                "epoch",
                "(JD_OBS // 365.25 - 4711.5) * units.yr",
            ),
            help="An expression to evaluate for each catalog source to "
            "determine the epoch to which to propagate star positions.",
        )
        self.add_argument(
            f"--{prefix}-catalog-columns",
            f"--{prefix}-catalogue-columns",
            "--cat-columns",
            type=str,
            nargs="+",
            default=catalog_config.get(
                "columns",
                [
                    "source_id",
                    "ra",
                    "dec",
                    "pmra",
                    "pmdec",
                    "phot_g_n_obs",
                    "phot_g_mean_mag",
                    "phot_g_mean_flux",
                    "phot_g_mean_flux_error",
                    "phot_bp_n_obs",
                    "phot_bp_mean_mag",
                    "phot_bp_mean_flux",
                    "phot_bp_mean_flux_error",
                    "phot_rp_n_obs",
                    "phot_rp_mean_mag",
                    "phot_rp_mean_flux",
                    "phot_rp_mean_flux_error",
                    "phot_proc_mode",
                    "phot_bp_rp_excess_factor",
                ],
            ),
            help="The columns to include in the catalog file. Use '*' to "
            "include everything.",
        )
        self.add_argument(
            f"--{prefix}-catalog-fov-safety-margin",
            f"--{prefix}-catalogue-fov-safety-margin",
            "--cat-fov-safety",
            type=float,
            default=catalog_config.get("fov_safety_margin", 0.1),
            help="The fractional safety margin to require of the field of view "
            "of the catalog. More specifically, the absolute valueso f xi and "
            "eta of the corners of the frame times this factor must be less "
            "than the half width and half height of the catalog respectively.",
        )
        if prefix in ["lc", "magfit"]:
            self.add_argument(
                f"--{prefix}-catalog-max-pointing-offset",
                type=float,
                default=catalog_config.get("max_pointing_offset", 10.0),
                help="The maximum difference in degrees between pointings of "
                "individual frames in the RA or Dec directions before frames "
                "are considered outliers.",
            )

    def _add_exposure_timing(self):
        """Add command line arguments to determine exposure start & duration."""

        self.add_argument(
            "--exposure-start-utc",
            "--start-time-utc",
            default='DATE_OBS + "T" + TIME_OBS',
            help="The UTC time at which the exposure started. Can be arbitrary "
            "expression involving header keywords.",
        )
        self.add_argument(
            "--exposure-start-jd",
            "--start-time-jd",
            default=None,
            help="The JD at which the exposure started. Can be arbitrary "
            "expression involving header keywords.",
        )
        self.add_argument(
            "--exposure-seconds",
            default="EXPTIME",
            help="The length of the exposure in seconds. Can be arbitrary "
            "expression involving header keywords.",
        )

    def __init__(
        self,
        *,
        input_type,
        description,
        add_component_versions=(),
        add_catalog=False,
        add_photref=False,
        inputs_help_extra="",
        allow_parallel_processing=False,
        convert_to_dict=True,
        add_lc_fname_arg=False,
        add_exposure_timing=False,
        skip_io=False,
    ):
        """
        Initialize the praser with options common to all manual steps.

        Args:
            input_type(str):    What kind of files does the step process.
                Possible values are ``'raw'``, ``'calibrated'``, ``'dr'``,
                or ``'calibrated + dr'``.

            description(str):    The description of the processing step to add
                to the help message.

            add_component_versions(str iterable):    A list of DR file version
                numbers the step needs to know. For example ``('srcextract',)``.

            add_catalog(False or dict):    Whether to add an arguments to
                specify a catalog query. If not False should specify defaults
                for some or all of the option values and a prefix for the
                option names.

            inputs_help_extra(str):    Additional text to append to the help
                string for the input files. Usually describing what requirements
                they must satisfy.

            allow_parallel_processing(bool):    Should an argument be added to
                specify the number of paralllel processes to use.

            convert_to_dict(bool):    Whether to return the parsed configuration
                as a dictionary (True) or attributes of a namespace (False).

        Returns:
            None
        """

        self.argument_descriptions = {}
        self.argument_defaults = {}

        self._convert_to_dict = convert_to_dict
        super().__init__(
            description=description,
            default_config_files=[],
            formatter_class=DefaultsFormatter,
            ignore_unknown_config_file_keys=True,
        )
        self.add_argument(
            "--config-file",
            "-c",
            is_config_file=True,
            # default=config_file,
            help="Specify a configuration file in liu of using command line "
            "options. Any option can still be overriden on the command line.",
        )
        self.add_argument(
            "--extra-config-file",
            is_config_file=True,
            help="Hack around limitation of configargparse to allow for "
            "setting a second config file.",
        )

        if input_type == "raw":
            input_name = "raw_images"
        elif input_type.startswith("calibrated"):
            input_name = "calibrated_images"
        elif input_type == "dr":
            input_name = "dr_files"
        elif input_type == "lc":
            input_name = "lc_files"
        else:
            input_name = None

        if input_name is not None:
            self.add_argument(
                input_name,
                nargs="+",
                help=(
                    # Would not work with calculated arngument
                    # pylint: disable=consider-using-f-string
                    (
                        "A combination of individual {0}s and {0} directories "
                        "to process. Directories are not searched recursively."
                    ).format(input_name[:-1].replace("_", " "))
                    # pylint: enable=consider-using-f-string
                    + inputs_help_extra
                ),
            )
        if "+" in input_type and input_type.split("+")[1].strip() == "dr":
            self.add_argument(
                "--data-reduction-fname",
                default="DR/{RAWFNAME}.h5",
                help="Format string to generate the filename(s) of the data "
                "reduction files where extracted sources are saved. Replacement"
                " fields can be anything from the header of the calibrated "
                "image.",
            )
        if allow_parallel_processing:
            self.add_argument(
                "--num-parallel-processes",
                type=int,
                default=12,
                help="The number of simultaneous fitpsf/fitprf processes to "
                "run.",
            )

        self._add_version_args(add_component_versions)
        if add_lc_fname_arg:
            self.add_argument(
                "--lc-fname",
                default="LC/GDR3_{:d}.h5",
                help="The light curve dumping filename pattern to use.",
            )

        if not skip_io:
            self.add_argument(
                "--std-out-err-fname",
                default="{processing_step:s}_{task:s}_{now:s}_pid{pid:d}.outerr",
                help="The filename pattern to redirect stdout and stderr during"
                "multiprocessing. Should include substitutions to distinguish "
                "output from different multiprocessing processes. May include "
                "substitutions for any configuration arguments for a given "
                "processing step.",
            )
            self.add_argument(
                "--fname-datetime-format",
                default="%Y%m%d%H%M%S",
                help="How to format date and time as part of filenames (e.g. when "
                "creating output files for multiprocessing.",
            )
            self.add_argument(
                "--logging-fname",
                default="{processing_step:s}_{task:s}_{now:s}_pid{pid:d}.log",
                help="The filename pattern to use for log files. Should include"
                " substitutions to distinguish logs from different "
                "multiprocessing processes. May include substitutions for any "
                "configuration arguments for a given processing step.",
            )
            self.add_argument(
                "--verbose",
                default="info",
                choices=["debug", "info", "warning", "error", "critical"],
                help="The type of verbosity of logger.",
            )
            self.add_argument(
                "--logging-message-format",
                default=(
                    "%(levelname)s %(asctime)s %(name)s: %(message)s | "
                    "%(pathname)s.%(funcName)s:%(lineno)d"
                ),
                help="The format string to use for log messages. See python logging"
                " module for details.",
            )
            self.add_argument(
                "--logging-datetime-format",
                default=None,
                help="How to format date and time as part of filenames (e.g. when "
                "creating output files for multiprocessing.",
            )
        if add_catalog:
            self._add_catalog_args(add_catalog)

        if add_exposure_timing:
            self._add_exposure_timing()

        if add_photref:
            self.add_argument(
                "--single-photref-dr-fname",
                default="single_photref.hdf5.0",
                help="The name of the data reduction file of the single "
                "photometric reference to use or used to start the magnitude "
                "fitting iterations.",
            )

    def add_argument(self, *args, **kwargs):
        """Store each argument's description in self.argument_descriptions."""

        argument_name = args[0].lstrip("-")
        if kwargs.get("action", None) == "store_false":
            self.argument_descriptions[argument_name] = {
                "rename": kwargs["dest"],
                "help": kwargs["help"],
            }
        else:
            self.argument_descriptions[argument_name] = kwargs["help"]

        if "default" in kwargs:
            nargs = kwargs.get("nargs", 1)
            if isinstance(kwargs["default"], str) or kwargs["default"] is None:
                self.argument_defaults[argument_name] = kwargs["default"]
            else:
                if kwargs.get("action", None) == "store_true":
                    assert kwargs.get("default", False) is False
                    self.argument_defaults[argument_name] = "False"
                elif kwargs.get("action", None) == "store_false":
                    assert kwargs.get("default", True) is True
                    self.argument_defaults[argument_name] = repr(
                        kwargs["dest"] == argument_name
                    )
                else:
                    self.argument_defaults[argument_name] = repr(
                        kwargs["default"]
                    )
                if (
                    "type" not in kwargs
                    and kwargs.get("action", None)
                    not in ["store_true", "store_false"]
                    and nargs == 1
                ):
                    raise ValueError(
                        f'Non-string default value ({kwargs["default"]}) and '
                        f"no type specified for {argument_name}."
                    )
                if kwargs.get("action", None) not in [
                    "store_true",
                    "store_false",
                    "append",
                ]:
                    if nargs in ["*", "+"] or nargs > 1:
                        self.argument_defaults[argument_name] = (
                            "["
                            + ", ".join(
                                [
                                    x if isinstance(x, str) else repr(x)
                                    for x in kwargs["default"]
                                ]
                            )
                            + "]"
                        )
                    elif (
                        kwargs["type"](self.argument_defaults[argument_name])
                        != kwargs["default"]
                    ):
                        raise ValueError(
                            "Could not convert default value of "
                            f'{argument_name} for DB: {kwargs["default"]}'
                        )

        return super().add_argument(*args, **kwargs)

    # pylint: disable=signature-differs
    def parse_args(self, *args, **kwargs):
        """Set-up logging and return cleaned up dict instead of namespace."""

        result = super().parse_args(*args, **kwargs)
        result.processing_step = path.basename(argv[0])
        if result.processing_step.endswith(".py"):
            result.processing_step = result.processing_step[:-3]
        else:
            assert result.processing_step.startswith("wisp-")
            result.processing_step = result.processing_step[5:].replace(
                "-", "_"
            )

        try:
            logging_level = getattr(logging, result.verbose.upper())
            logging.basicConfig(
                level=logging_level,
                format="%(levelname)s %(asctime)s %(name)s: %(message)s | "
                "%(pathname)s.%(funcName)s:%(lineno)d",
            )
            logging.getLogger("sqlalchemy.engine").setLevel(logging_level)
        except AttributeError:
            pass

        if self._convert_to_dict:
            result = vars(result)
            del result["config_file"]
            del result["extra_config_file"]
        else:
            del result.config_file
            del result.extra_config_file
            del result.verbose

        if args or kwargs:
            result["argument_descriptions"] = self.argument_descriptions
            result["argument_defaults"] = self.argument_defaults

        return result

    # pylint: enable=signature-differs


def add_image_options(parser, include=("subpixmap", "gain", "magnitude-1adu")):
    """Add options specifying the properties of the image."""

    if "subpixmap" in include:
        parser.add_argument(
            "--subpixmap",
            default=None,
            help="The sub-pixel sensitivity map to assume. If not specified "
            "uniform sensitivy is assumed.",
        )
    if "gain" in include:
        parser.add_argument(
            "--gain",
            type=float,
            default=1.0,
            help="The gain to assume for the input images.",
        )
    if "magnitude-1adu" in include:
        parser.add_argument(
            "--magnitude-1adu",
            type=float,
            default=10.0,
            help="The magnitude which corresponds to a source flux of 1ADU",
        )


def read_subpixmap(fits_fname):
    """Read the sub-pixel sensitivity map from a FITS file."""

    if fits_fname is None:
        return numpy.ones((1, 1), dtype=float)
    with fits.open(fits_fname, "readonly") as subpixmap_file:
        # False positive, pylint does not see data member.
        # pylint: disable=no-member
        return numpy.copy(subpixmap_file[0].data).astype("float64")
        # pylint: enable=no-member


# These must be acceptable as keyword arguments
# pylint: disable=unused-argument
def ignore_progress(input_fname, status=1, final=True):
    """Dummy function to replace progress tracking of auto processing."""

    return


# pylint: enable=unused-argument


def get_catalog_config(cmdline_args, prefix):
    """Return the configuration for querrying a catalog per command line."""

    prefix = prefix + "_catalog"
    result = {
        "fname" if key == prefix else key[len(prefix) + 1 :]: value
        for key, value in cmdline_args.items()
        if key.startswith(prefix)
    }
    if "frame_fov_estimate" in cmdline_args:
        result["frame_fov_estimate"] = cmdline_args["frame_fov_estimate"]
    return result
