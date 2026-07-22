"""Collection of functions used by many processing steps."""

import logging
import threading
from contextlib import contextmanager
from sys import argv
from os import path

import numpy
from astropy.io import fits
from astropy.utils import iers

from configargparse import ArgumentParser, DefaultsFormatter

from autowisp.catalog import WISPGaia
from autowisp.exceptions import ConfigurationError


# When active (set via ``raise_config_parse_errors``),
# ``ManualStepArgumentParser`` reports a bad value by raising
# ``ConfigurationError`` instead of argparse's default "print usage to
# stderr + sys.exit(2)". The pipeline turns stored configuration into a
# step config dict by feeding it through the step's parser
# (``ProcessingManager.get_config``); a bad value there must surface as a
# recordable error rather than an uncatchable ``SystemExit`` that silently
# ends a detached run.
_config_parse_mode = threading.local()


def parse_iers_max_age(value):
    """Parse the ``--iers-auto-max-age`` value into what astropy expects.

    Args:
        value:    The command line value: a number of days, or one of
            ``none``/``off``/``disable`` (case insensitive) to switch off the
            staleness check.

    Returns:
        float or None:    The maximum age in days, or ``None`` to disable the
            check (and hence the automatic IERS download) entirely. A
            non-positive number is also treated as ``None``.
    """

    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in (
        "none",
        "off",
        "disable",
        "disabled",
    ):
        return None
    age = float(value)
    return age if age > 0 else None


@contextmanager
def raise_config_parse_errors():
    """Make ``ManualStepArgumentParser`` raise on bad input, not exit.

    Use around programmatic parsing of stored configuration, where there
    is no interactive user to read an argparse usage message and the
    process may be a detached pipeline run. A parse failure then becomes a
    catchable, recordable
    :class:`~autowisp.exceptions.ConfigurationError` instead of
    ``SystemExit``. Nestable; restores the previous mode on exit.

    Yields:
        None
    """

    previous = getattr(_config_parse_mode, "active", False)
    _config_parse_mode.active = True
    try:
        yield
    finally:
        _config_parse_mode.active = previous


class ManualStepArgumentParser(ArgumentParser):
    """Incorporate boiler plate handling of command line arguments."""

    def error(self, message):
        """Report a parse error, honouring :func:`raise_config_parse_errors`.

        argparse's default reports a bad value by printing usage to stderr
        and calling ``sys.exit(2)``. During pipeline config generation that
        ``SystemExit`` would escape the error-handling machinery (it is a
        ``BaseException``, not caught by ``capture_errors`` /
        ``run_pipeline.main``) and silently end a detached run, so there we
        raise a recordable :class:`~autowisp.exceptions.ConfigurationError`
        instead. Interactive CLI parsing keeps argparse's helpful
        usage-and-exit behaviour.

        Args:
            message(str):    The argparse-formatted error description.

        Returns:
            None
        """

        if getattr(_config_parse_mode, "active", False):
            raise ConfigurationError(f"Invalid configuration: {message}")
        super().error(message)

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
            "--gaia-user",
            default=None,
            help="The username to use for Gaia archive queries. If not "
            "specified, queries are submitted without logging in.",
        )
        self.add_argument(
            "--gaia-password",
            default=None,
            help="The password to use for Gaia archive queries. If not "
            "specified, queries are submitted without logging in.",
        )
        self.add_argument(
            f"--{prefix}-catalog",
            f"--{prefix}-catalogue",
            "--cat",
            default=catalog_config.get(
                "fname", "{PROJHOME}/MASTERS/Gaia/{checksum:s}.fits"
            ),
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
        max_mag_extra_help = {
            "astrometry": (
                "This should approximately correspond to the "
                ":option:`brightness-threshold` argument used for source "
                "extraction. This can be automatically determined in most cases"
                " using ``wisp-tune-astromety-max-mag``."
            ),
            "photometry": (
                "This determines the faintest stars that will receive flux "
                "measuremets from AutoWISP. Going too deep will result in "
                "excessive number of stars being photometerd, producing large "
                "files and slow processing. Especially for wide-field images."
            ),
            "magfit": (
                "In most cases, the photometry catalog should be used for "
                "magnitude fitting as well."
            ),
            "lc": (
                "In most cases, the photometry catalog should be used for "
                "creating lightcurves as well."
            ),
        }
        self.add_argument(
            f"--{prefix}-catalog-max-magnitude",
            f"--{prefix}-catalogue-max-magnitude",
            "--cat-max-mag",
            type=float,
            default=catalog_config.get("max_magnitude", 12.0),
            help="The faintest magnitude to include in the catalog."
            + max_mag_extra_help[prefix],
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
            f"--{prefix}-catalog-source-list",
            f"--{prefix}-catalogue-source-list",
            "--cat-source-list",
            default=catalog_config.get("source_list"),
            help="Path to a file listing the GAIA source IDs (one per line) to "
            "keep in the catalog. When given, the catalog is restricted to "
            "these sources at read time, so the full cached catalog query is "
            "reused. If not specified, all sources in the catalog are kept.",
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

    def _add_provenance_args(self):
        """Add provenance flags mirroring ``wisp-add-images-to-db``.

        These let a manual step resolve a FITS header to the survey-database
        rows (``Camera``, ``Telescope``, ``Mount``, ``Observatory``, ``Observer``,
        ``Target``) via
        :func:`autowisp.database.provenance_resolver.get_or_create_observing_session`,
        so that the ``/Provenance`` group can be written to the DR file
        without requiring ``wisp-add-images-to-db`` to have been run.

        Flag names and defaults intentionally mirror
        ``wisp-add-images-to-db`` so a shared config file works for both.
        """

        self.add_argument(
            "--observer",
            default="ORIGIN",
            help="The name of the observer who/which collected the images. Can "
            "be arbitrary expression involving header keywords. Must already "
            "have an entry in the ``observer`` table.",
        )
        self.add_argument(
            "--camera-serial-number",
            "--cam-sn",
            default="CAMSN",
            help="The serial number of the camera which collected the images. "
            "Can be arbitrary expression involving header keywords. Must "
            "already have an entry in the ``camera`` table.",
        )
        self.add_argument(
            "--telescope-serial-number",
            "--tel-sn",
            default="INTSN",
            help="The serial number of the telescope (lens) which collected "
            "the images. Can be arbitrary expression involving header "
            "keywords. Must already have an entry in the ``telescope`` table.",
        )
        self.add_argument(
            "--mount-serial-number",
            "--mount-sn",
            default="OBSERVER",
            help="The serial number of the mount which collected the images. "
            "Can be arbitrary expression involving header keywords. Must "
            "already have an entry in the ``mount`` table.",
        )
        self.add_argument(
            "--observatory",
            default=None,
            help="The name of the observatory from where the images were "
            "collected. Can be arbitrary expression involving header keywords. "
            "Must already have an entry in the ``observatory`` table. If not "
            "specified, ``--observatory-location`` is used to determine the "
            "observatory.",
        )
        self.add_argument(
            "--observatory-location",
            metavar=("LATITUDE", "LONGITUDE", "ALTITUDE"),
            default=["LAT_OBS", "LONG_OBS", "ALT_OBS"],
            nargs=3,
            help="The latitude, longitude, and altitude of the observatory. "
            "Can be arbitrary expressions involving header keywords. The "
            "matching ``observatory`` row must be within ~100km. Only used if "
            "``--observatory`` is not specified.",
        )
        self.add_argument(
            "--target-ra",
            "--ra",
            default="RA_MNT",
            help="The RA targeted by the observations in degrees. Can be "
            "arbitrary expression involving header keywords.",
        )
        self.add_argument(
            "--target-dec",
            "--dec",
            default="DEC_MNT",
            help="The Dec targeted by the observations in degrees. Can be "
            "arbitrary expression involving header keywords.",
        )
        self.add_argument(
            "--target-name",
            "--target",
            default="FIELD",
            help="The name of the targeted area of the sky. Can be arbitrary "
            "expression involving header keywords.",
        )
        self.add_argument(
            "--target-match-tolerance",
            type=float,
            default=0.05,
            help="Maximum allowed separation between image and target as a "
            "fraction of the field of view (default: 0.05 = 5%%).",
        )
        self.add_argument(
            "--observing-session-label",
            "--session-label",
            "--session",
            default="SEQID",
            help="Unique label for the observing session. Can be arbitrary "
            "expression involving header keywords.",
        )
        self.add_argument(
            "--no-provenance",
            action="store_true",
            default=False,
            help="Skip resolving and writing the ``/Provenance`` HDF5 group. "
            "Use when the survey database has not been populated via "
            "``wisp-survey import``. ``wisp-add-images-to-db`` ignores this "
            "flag and always resolves provenance.",
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

    def __init__(  # pylint: disable=too-many-arguments, too-many-branches
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
        add_provenance_args=False,
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
        self.alternate_names = {}

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
        self.add_argument(
            "--project-home",
            default=None,
            help="The name of the project home directory to "
            "attach processing to. Should at least define HDF5 "
            "structure for DR files and lightcurves"
            " for BUI a lot more is needed but that is automatically managed "
            "by the BUI.",
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
                default="{PROJHOME}/DR/{RAWFNAME}_{CLRCHNL}.h5",
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
                default="{PROJHOME}/LC/GDR3_{:d}.h5",
                help="The light curve dumping filename pattern to use.",
            )

        if not skip_io:
            self.add_argument(
                "--std-out-err-fname",
                default="{project_home}/LOGS/"
                "{processing_step:s}_{task:s}_{now:s}_pid{pid:d}"
                ".outerr",
                help="The filename pattern to redirect stdout and stderr during"
                "multiprocessing. Should include substitutions to distinguish "
                "output from different multiprocessing processes. May include "
                "substitutions for any configuration arguments for a given "
                "processing step.",
            )
            self.add_argument(
                "--fname-datetime-format",
                default="%Y%m%d%H%M%S",
                help="How to format date and time as part of filenames (e.g. "
                "when creating output files for multiprocessing.",
            )
            self.add_argument(
                "--logging-fname",
                default="{project_home}/LOGS/"
                "{processing_step:s}_{task:s}_{now:s}_pid{pid:d}.log",
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
                help="The format string to use for log messages. See python "
                "logging module for details.",
            )
            self.add_argument(
                "--logging-datetime-format",
                default=None,
                help="How to format date and time as part of filenames (e.g. "
                "when creating output files for multiprocessing.",
            )
            self.add_argument(
                "--iers-auto-max-age",
                type=parse_iers_max_age,
                default=30.0,
                help="Maximum age (in days) of the bundled/cached astropy IERS "
                "table before astropy tries to download a fresh copy for time "
                "transformations. Pass ``none`` (or a non-positive number) to "
                "disable the staleness check entirely, so the existing table is "
                "used without ever triggering a download. Useful when the IERS "
                "download location is unreliable.",
            )
        if add_catalog:
            self._add_catalog_args(add_catalog)

        if add_exposure_timing or add_provenance_args:
            self._add_exposure_timing()
        if add_provenance_args:
            self._add_provenance_args()

        if add_photref:
            self.add_argument(
                "--single-photref-dr-fname",
                default="single_photref.hdf5.0",
                help="The name of the data reduction file of the single "
                "photometric reference to use or used to start the magnitude "
                "fitting iterations.",
            )
            self.add_argument(
                "--max-photref-separation",
                type=float,
                default=0.2,
                help="The maximum angular separation between the center of an "
                "image and the center of the photometric reference, expressed "
                "in units of the image's diagonal field of view (mean angular "
                "distance from the image center to its four corners on the "
                "sky, stored as the ``diagonal_fov`` astrometry diagnostic). "
                "Images whose centers lie farther than this multiple of their "
                "diagonal FOV from the photref center are excluded.",
            )

    def add_argument(self, *args, **kwargs):
        """Store each argument's description in self.argument_descriptions."""

        argument_name = args[0].lstrip("-")
        self.alternate_names[argument_name] = [
            entry.lstrip("-") for entry in args[1:]
        ]
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
        elif result.processing_step.startswith("wisp-"):
            result.processing_step = result.processing_step[5:].replace(
                "-", "_"
            )
        else:
            # ``argv[0]`` is neither a ``wisp-*`` console entry point nor a
            # ``*.py`` script. Reachable when the parser is invoked from a
            # test harness or coverage runner (e.g. ``coverage run -m
            # autowisp.tests``) where ``argv[0]`` is something like
            # ``coverage`` or ``__main__``. ``processing_step`` is used only
            # to template log filenames, so falling back to the basename
            # keeps things working without crashing the test.
            result.processing_step = result.processing_step.replace(
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

        if hasattr(result, "iers_auto_max_age"):
            iers.conf.auto_max_age = result.iers_auto_max_age

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
            result["alternate_argument_names"] = self.alternate_names

        if result.get("gaia_user") and result.get("gaia_password"):
            WISPGaia.set_credentials(
                user=result["gaia_user"], password=result["gaia_password"]
            )

        return result

    # pylint: enable=signature-differs


def add_image_options(parser, include=("subpixmap", "gain", "magnitude-1adu")):
    """Add options specifying the properties of the image."""

    if "subpixmap" in include:
        parser.add_argument(
            "--subpixmap",
            default=None,
            help="The sub-pixel sensitivity map to assume. If not specified "
            "uniform sensitivy is assumed. This is especially important if "
            "processing images from color cameras. For a standard Bayer array, "
            "the built-in map called ``dslr_subpixmap.fits`` can be used.",
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
def ignore_progress(  # pylint: disable=unused-argument
    input_fname, status=1, final=True, **kwargs
):
    """Dummy function to replace progress tracking of auto processing."""

    return
