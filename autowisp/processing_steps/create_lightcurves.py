#!/usr/bin/env python3

"""Create light curves from a collection of DR files."""

from bisect import bisect
import logging
from os import path, makedirs

from asteval import Interpreter
import numpy
from astropy.io import fits
from astropy.table import Table

from autowisp.multiprocessing_util import setup_process
from autowisp.error_cli import cli_entry_point
from autowisp.exceptions import Component
from autowisp.data_reduction.data_reduction_file import DataReductionFile
from autowisp.file_utilities import find_dr_fnames
from autowisp.light_curves.light_curve_file import LightCurveFile
from autowisp.processing_steps.manual_util import (
    ManualStepArgumentParser,
    ignore_progress,
)
from autowisp.light_curves.collect_light_curves import (
    collect_light_curves,
    get_combined_sources,
    DecodingStringFormatter,
)
from autowisp.catalog import read_catalog_file, get_catalog_config
from autowisp.error_context import error_context
from autowisp.exceptions import FileKind, RelatedFile

input_type = "dr"
#: On top of the always-allowed "nothing done yet": ``1`` means a previous
#: run got as far as adding the points but not marking the DR files final,
#: so it only needs finishing off.
allowed_start_status_values = (1,)
#: Unlike the steps that only ever record "started", lightcurve creation
#: adds the points (``0`` -> ``1``) before marking the DR files final, so
#: an interrupted run can be found in either state.
allowed_interrupted_status_values = (0, 1)

_logger = logging.getLogger(__name__)


def parse_command_line(*args):
    """Return the parsed command line arguments."""

    parser = ManualStepArgumentParser(
        description=__doc__,
        input_type="" if args else input_type,
        inputs_help_extra=(
            "The corresponding DR files must alread contain all "
            "photometric measurements and be magnitude fitted."
        ),
        add_catalog={"prefix": "lc"},
        add_photref=True,
        add_component_versions=(
            "srcproj",
            "background",
            "shapefit",
            "apphot",
            "magfit",
            "srcextract",
            "skytoframe",
            "catalogue",
        ),
        add_lc_fname_arg=True,
        allow_parallel_processing=True,
    )
    parser.add_argument(
        "--lc-only-if",
        default="True",
        help="Expression involving the header of the input images that "
        "evaluates to True/False if a particular image from the specified "
        "image collection should/should not be processed.",
    )
    parser.add_argument(
        "--apertures",
        nargs="+",
        default=[],
        type=float,
        help="The apretures aperturte photometry was measured for (only their "
        "number is used).",
    )
    parser.add_argument(
        "--num-magfit-iterations",
        type=int,
        default=0,
        help="The number of iterations of deriving a master photometric"
        " referene and re-fitting to copy from DR to LCs. If left as "
        "zero the number of magnitude fitting iterations of the first DR file "
        "is used instead.",
    )
    parser.add_argument(
        "--srcproj-column-names",
        nargs="+",
        default=["x", "y", "xi", "eta", "enabled"],
        help="List of the source projected columns to include in the "
        "lightcurves.",
    )
    parser.add_argument(
        "--max-memory",
        default="4096",
        type=int,
        help="The maximum amount of memory (in bytes) to allocate for "
        "temporaririly storing source information before dumping to LC.",
    )
    parser.add_argument(
        "--mem-block-size",
        default="Mb",
        help="The block size to use for memblocksize to use. Options are "
        "``Mb`` or ``Gb``.",
    )
    parser.add_argument(
        "--sort-frame-by",
        default="{FNUM}",
        help="A format string involving header keywords to sort the the "
        "lightcurve entries by (alphabetically).",
    )
    parser.add_argument(
        "--latitude-deg",
        default="{LATITUDE}",
        help="A format string of header keywords specifying the latitude in "
        "degrees at which the observations of the lightcurves being generated "
        "were collected. Will be evaluated so it is OK to use mathematical "
        "functions or operators. Alternatively just specify a number.",
    )
    parser.add_argument(
        "--longitude-deg",
        default="{LONGITUD}",
        help="Same as `--latitude-deg` but for the longitude.",
    )
    parser.add_argument(
        "--altitude-meters",
        default="{ALTITUDE}",
        help="Same as `--latitude-deg` but for the altitude in meters.",
    )
    parser.add_argument(
        "--lightcurve-catalog-fname",
        default="{PROJHOME}/MASTERS/"
        "lc_catalog_{TARGETID}_{CLRCHNL}_{EXPTIME}.fits",
        help="The filename of the master catalog to create, possibly with "
        "substitutions from the header of the single photometric reference.",
    )

    result = parser.parse_args(*args)

    mem_block_size = result.pop("mem_block_size")
    if mem_block_size == "Mb":
        result["max_memory"] = int(result["max_memory"] * 1024**2)
    elif mem_block_size == "Gb":
        result["max_memory"] = int(result["max_memory"] * 1024**3)

    result["max_apertures"] = len(result.pop("apertures"))

    return result


def dummy_fname_parser(_):
    """No extra keywords to add from filename."""

    return {}


def get_observatory(header, configuration):
    """Return (latitude, longitude, altitude) where given data was collected."""

    evaluate = Interpreter()
    _logger.debug(
        "Observatory expressions: %s",
        repr(
            tuple(
                configuration[coordinate].format_map(header)
                for coordinate in [
                    "latitude_deg",
                    "longitude_deg",
                    "altitude_meters",
                ]
            )
        ),
    )
    return tuple(
        evaluate(configuration[coordinate].format_map(header))
        for coordinate in ["latitude_deg", "longitude_deg", "altitude_meters"]
    )


def sort_by_observatory(dr_collection, configuration):
    """Split the DR files by observatory coords and sort within observatory."""

    result = {}
    sort_keys = {}
    for dr_fname in dr_collection:
        with DataReductionFile(dr_fname, "r") as dr_file:
            header = dr_file.get_frame_header()
            observatory = get_observatory(header, configuration)
            sort_key = configuration["sort_frame_by"].format_map(header)
            if observatory not in result:
                result[observatory] = [dr_fname]
                sort_keys[observatory] = [sort_key]
            else:
                insert_pos = bisect(sort_keys[observatory], sort_key)
                sort_keys[observatory].insert(insert_pos, sort_key)
                result[observatory].insert(insert_pos, dr_fname)
    return result


def get_path_substitutions(configuration):
    """Return the path substitutions for getting data out of DR files."""

    path_substitutions = {}
    for option, value in configuration.items():
        if option.endswith("_version"):
            path_substitutions[option] = value
    return path_substitutions


class MasterCatalog:
    """Class for managing the master catalog for LC creation."""

    def __init__(self, configuration):
        """Prepare to add sources to the catalog."""

        with DataReductionFile(
            configuration["single_photref_dr_fname"], "r"
        ) as sphotref_dr:
            sphotref_header = sphotref_dr.get_frame_header()
            self._master_cat_fname = configuration[
                "lightcurve_catalog_fname"
            ].format_map(sphotref_header)
            if path.exists(self._master_cat_fname):
                self._sources, self._header = read_catalog_file(
                    self._master_cat_fname, return_metadata=True
                )
                self._centers = None
                self._new = False
            else:
                self._sources = None
                self._header = sphotref_header.copy()
                for k in ["NAXIS", "NAXIS1", "NAXIS2", "BITPIX", "XTENSION"]:
                    if k in self._header:
                        del self._header[k]
                self._centers = {"RA": [], "Dec": []}
                self._new = True

    def add_batch(self, new_sources, new_header):
        """Add any sources after a batch of newly created lightcurves."""

        if self._sources is None:
            self._sources = new_sources.copy()
        else:
            self._sources = self._sources.combine_first(new_sources)

        if self._centers is not None:
            for coord in ["RA", "Dec"]:
                self._centers[coord].append(new_header[coord])

    def save(self):
        """Save the catalog and return the filename."""

        table_hdu = fits.table_to_hdu(
            Table.from_pandas(self._sources, index=True)
        )
        table_hdu.header.update(self._header)
        if self._centers is not None:
            for coord in ["RA", "Dec"]:
                table_hdu.header[coord] = numpy.median(self._centers[coord])

        cat_dir = path.dirname(self._master_cat_fname)
        if not path.exists(cat_dir):
            makedirs(cat_dir)
        fits.HDUList([fits.PrimaryHDU(), table_hdu]).writeto(
            self._master_cat_fname, overwrite=True
        )
        return self._master_cat_fname

    def is_new(self):
        """True iff the given master is newly created rather than extension."""

        return self._new

    def as_related_file(self):
        """This catalog as a :class:`RelatedFile`, with the fitting role.

        The resolved filename is known only here (the configured value is
        a header-substituted template), and the role depends on whether
        this run reads an existing catalog or writes a new one.
        """

        return RelatedFile(
            FileKind.CATALOG,
            self._master_cat_fname,
            role="expected_output" if self._new else "lc_catalog",
        )


def _catalog_related_files(master_catalog, configuration):
    """The lightcurve catalog and, if configured, its source-list filter.

    Args:
        master_catalog(MasterCatalog):    Knows the resolved catalog
            filename and whether this run creates it.

        configuration:    The step configuration, holding the ``lc``
            catalog query settings.

    Returns:
        list:    The ``RelatedFile`` entries known for the whole step.
    """

    related = [master_catalog.as_related_file()]
    source_list = get_catalog_config(configuration, "lc").get("source_list")
    if source_list:
        related.append(
            RelatedFile(FileKind.CONFIG, source_list, role="source_list_filter")
        )
    return related


def create_lightcurves(
    dr_collection, start_status, configuration, mark_start, mark_end
):
    """Create lightcurves from the data in the given DR files."""

    # TODO: figure out source extraction map variables from DR file
    _logger.debug("Organizing DR files by observatory")

    dr_by_observatory = sort_by_observatory(dr_collection, configuration)

    if start_status == 1:
        for dr_fname in dr_collection:
            mark_end(dr_fname)
        return None

    path_substitutions = get_path_substitutions(configuration)
    _logger.debug("Path substitutions: %s", repr(path_substitutions))

    _logger.debug(
        "Creating master catalog with configuration: %s", repr(configuration)
    )
    master_catalog = MasterCatalog(configuration)

    # Step-wide inputs, on top of the single photref the LC manager already
    # scopes: a wrong catalog or source-list filter silently changes which
    # sources get lightcurves at all. The per-DR and per-lightcurve files
    # are scoped deeper, in ``collect_light_curves``.
    with error_context(
        related_files=_catalog_related_files(master_catalog, configuration)
    ):
        for (lat, lon, alt), dr_filename_list in dr_by_observatory.items():
            _logger.debug(
                "Collecting lightcurves for %d DR files at observatory "
                "(lat, lon, alt) = (%.4f, %.4f, %.4f)",
                len(dr_filename_list),
                lat,
                lon,
                alt,
            )
            master_catalog.add_batch(
                *collect_light_curves(
                    dr_filename_list,
                    configuration,
                    mark_start=mark_start,
                    mark_end=mark_end,
                    dr_fname_parser=dummy_fname_parser,
                    optional_header="all",
                    observatory={
                        "SITELAT": lat,
                        "SITELONG": lon,
                        "SITEALT": alt,
                    },
                    **path_substitutions,
                )
            )

    if master_catalog.is_new():
        return {
            "filename": master_catalog.save(),
            "preference_order": None,
            "type": "lightcurve_catalog",
        }

    return None


def cleanup_interrupted(interrupted, configuration):
    """Cleanup the file system after interrupted lightcurve creation."""

    min_status = min(interrupted, key=lambda x: x[1])[1]
    max_status = max(interrupted, key=lambda x: x[1])[1]
    _logger.info(
        "Cleaning up interrupted lightcurve generation with status %d to %d",
        min_status,
        max_status,
    )
    # The manager has already checked every status against
    # ``allowed_interrupted_status_values``, so the only case left is 1.
    if max_status == 0:
        return -1

    dr_sources = get_combined_sources(
        map(lambda x: x[0], interrupted),
        **get_path_substitutions(configuration),
    )

    srcid_formatter = DecodingStringFormatter()
    for source_id in dr_sources:
        with LightCurveFile(
            srcid_formatter.format(
                configuration["lc_fname"], *numpy.atleast_1d(source_id)
            ),
            "a",
        ) as lc_file:
            lc_file.confirm_lc_length()

    return 1


def has_magfit(dr_fname, substitutions):
    """Check if the DR file contains magnitude fitting data."""

    _logger.debug(
        "Checking %s for magnitude fitting data with substitutions %s",
        repr(dr_fname),
        repr(substitutions),
    )
    with DataReductionFile(dr_fname, mode="r") as dr_file:
        try:
            dr_file.check_for_dataset(
                "apphot.magfit.magnitude", **substitutions
            )
            return True
        except IOError:
            return False


@cli_entry_point(component=Component.STEP)
def main():
    """Run the light curve creation step from the command line."""

    configuration = parse_command_line()
    setup_process(task="main", **configuration)

    dr_path_substitutions = get_path_substitutions(configuration)
    with DataReductionFile(
        configuration["single_photref_dr_fname"], "r+"
    ) as single_photref_dr:
        dr_path_substitutions["aperture_index"] = (
            single_photref_dr.get_num_apertures(**dr_path_substitutions) - 1
        )
        if configuration["num_magfit_iterations"] == 0:
            dr_path_substitutions["magfit_iteration"] = (
                single_photref_dr.get_num_magfit_iterations(
                    **dr_path_substitutions
                )
                - 1
            )
        else:
            dr_path_substitutions["magfit_iteration"] = (
                configuration["num_magfit_iterations"] - 1
            )
    create_lightcurves(
        [
            dr_fname
            for dr_fname in find_dr_fnames(
                configuration.pop("dr_files"), configuration.pop("lc_only_if")
            )
            if has_magfit(dr_fname, dr_path_substitutions)
        ],
        None,
        configuration,
        ignore_progress,
        ignore_progress,
    )


if __name__ == "__main__":
    main()
