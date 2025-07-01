#!/usr/bin/env python3

"""Apply magnitude fitting to hdf5 files"""

from types import SimpleNamespace
from itertools import count
import os
import logging

from sqlalchemy import func, select

from autowisp.multiprocessing_util import setup_process
from autowisp import magnitude_fitting, DataReductionFile
from autowisp.file_utilities import find_dr_fnames
from autowisp.catalog import ensure_catalog
from autowisp.processing_steps.manual_util import (
    ManualStepArgumentParser,
    ignore_progress,
    get_catalog_config,
)
from autowisp.database.interface import Session

# False positive due to unusual importing
# pylint: disable=no-name-in-module
from autowisp.database.data_model import MasterFile

# pylint: enable=no-name-in-module

input_type = "dr"
_logger = logging.getLogger(__name__)


def parse_command_line(*args):
    """Return the parsed command line arguments."""

    parser = ManualStepArgumentParser(
        description=__doc__,
        input_type=("" if args else input_type),
        inputs_help_extra=(
            "The corresponding DR files must alread contain all "
            "photometric measurements."
        ),
        add_catalog={"prefix": "magfit"},
        add_component_versions=(
            "skytoframe",
            "srcproj",
            "background",
            "shapefit",
            "apphot",
            "magfit",
        ),
        add_photref=True,
        allow_parallel_processing=True,
    )
    parser.add_argument(
        "--magfit-only-if",
        default="True",
        help="Expression involving the header of the input images that "
        "evaluates to True/False if a particular image from the specified "
        "image collection should/should not be processed.",
    )
    parser.add_argument(
        "--master-photref-fname",
        default=None,
        help="The name of a master photometric reference to use. If specified, "
        "the sintgle reference is ignored and magnitude fitting proceeds "
        "without any iterations.",
    )
    parser.add_argument(
        "--master-photref-fname-format",
        default=(
            "MASTERS/mphotref_"
            "{FIELD}_{CLRCHNL}_{EXPTIME}sec_iter{magfit_iteration:03d}.fits"
        ),
        help="A format string involving a {magfit_iteration} substitution along"
        " with any variables from the header of the single photometric "
        "reference or passed through the path_substitutions arguments, that "
        "expands to the name of the file to save the master photometric "
        "reference for a particular iteration.",
    )
    parser.add_argument(
        "--magfit-stat-fname-format",
        default=(
            "MASTERS/mfit_stat_{FIELD}_{CLRCHNL}_{EXPTIME}sec_iter"
            "{magfit_iteration:03d}.txt"
        ),
        help="Similar to ``master_photref_fname_format``, but defines the name"
        " to use for saving the statistics of a magnitude fitting iteration.",
    )
    parser.add_argument(
        "--correction-parametrization",
        type=str,
        default="O3{phot_g_mean_mag, x, y}",
        help="A string that expands to the terms to include in the magnitude "
        "fitting correction.",
    )
    parser.add_argument(
        "--mphotref-scatter-fit-terms",
        default="O2{phot_g_mean_mag}",
        help="Terms to include in the fit for the scatter when deciding which "
        "stars to include in the master.",
    )
    parser.add_argument(
        "--reference-subpix",
        action="store_true",
        default=False,
        help="Should the magnitude fitting correction depend on the "
        "sub-pixel position of the source in the reference frame"
        "Default: %(default)s",
    )
    parser.add_argument(
        "--fit-source-condition",
        type=str,
        default="isfinite(phot_g_mean_mag)",
        help="An expression involving catalog, reference and/or photometry "
        "variables which evaluates to zero if a source should be excluded and "
        "any non-zero value if it  should be included in the magnitude fit.",
    )
    parser.add_argument(
        "--grouping",
        default=None,
        help=(
            "An expression using catalog, and/or photometry variables which "
            "evaluates to several distinct values. Each value "
            "defines a separate fitting group (i.e. a group of sources which "
            "participate in magnitude fitting together, excluding sources "
            "belonging to other groups). Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--error-avg",
        default="weightedmean",
        help="How to average fitting residuals for outlier rejection.",
    )
    parser.add_argument(
        "--rej-level",
        type=float,
        default=5.0,
        help="How far away from the fit should a point be before "
        "it is rejected in units of error_avg. Default: %(default)s",
    )
    parser.add_argument(
        "--max-rej-iter",
        type=int,
        default=20,
        help="The maximum number of rejection/re-fitting iterations to perform."
        " If the fit has not converged by then, the latest iteration is "
        "accepted.",
    )
    parser.add_argument(
        "--noise-offset",
        type=float,
        default=0.01,
        help="Additional offset to format magnitude error estimates when they "
        "are used to determine the fitting weights. ",
    )
    parser.add_argument(
        "--max-mag-err",
        type=float,
        default=0.1,
        help="The largest the formal magnitude error is allowed "
        "to be before the source is excluded. Default: %(default)s",
    )
    parser.add_argument(
        "--max-photref-change",
        type=float,
        default=1e-4,
        help="The maximum square average change of photometric reference "
        "magnitudes to consider the iterations converged.",
    )
    parser.add_argument(
        "--max-magfit-iterations",
        type=int,
        default=5,
        help="The maximum number of iterations of deriving a master photometric"
        " referene and re-fitting to allow.",
    )
    return parser.parse_args(*args)


def get_path_substitutions(configuration, sphotref_header):
    """Return the path substitutions to find magfit datasets."""

    result = {
        what + "_version": configuration[what + "_version"]
        for what in ["shapefit", "srcproj", "apphot", "background", "magfit"]
    }
    if configuration["master_photref_fname"] is not None:
        for iteration in count():
            if (
                configuration["master_photref_fname_format"].format(
                    **sphotref_header, **result, magfit_iteration=iteration
                )
                == configuration["master_photref_fname"]
            ):
                result["magfit_iteration"] = iteration
                break
            if iteration >= configuration["max_magfit_iterations"]:
                raise ValueError(
                    "Master photometric reference "
                    f"{configuration['master_photref_fname']!r} does not appear"
                    " to follow the specified filename format: "
                    f"{configuration['master_photref_fname_format']!r}!"
                )
    return result


def get_dr_fnames_and_catalog(
    dr_collection, configuration, sphotref_header, mark_start, mark_end
):
    """Return the DR files to process and the catalog."""

    dr_fnames = sorted(dr_collection)
    catalog_sources, outliers, catalog_fname = ensure_catalog(
        dr_files=dr_fnames,
        configuration=get_catalog_config(configuration, "magfit"),
        return_metadata=False,
        skytoframe_version=configuration["skytoframe_version"],
        header=sphotref_header,
    )
    for outlier_ind in reversed(outliers):
        outlier_dr = dr_fnames.pop(outlier_ind)
        _logger.warning(
            "Data reduction file %s has outlier pointing. Discarding!",
            outlier_dr,
        )
        mark_start(outlier_dr)
        mark_end(outlier_dr, -2, final=True)

    return dr_fnames, catalog_sources, catalog_fname


def fit_magnitudes(
    dr_collection, start_status, configuration, mark_start, mark_end
):
    """Perform magnitude fitting for the given DR files."""

    if start_status is None:
        start_status = -1
    else:
        assert start_status % 2 == 1

    with DataReductionFile(
        configuration["single_photref_dr_fname"], "r"
    ) as sphotref_dr_file:
        sphotref_header = sphotref_dr_file.get_frame_header()

    dr_fnames, catalog_sources, catalog_fname = get_dr_fnames_and_catalog(
        dr_collection, configuration, sphotref_header, mark_start, mark_end
    )

    kwargs = {
        "fit_dr_filenames": dr_fnames,
        "configuration": SimpleNamespace(
            **configuration,
            continue_from_iteration=(start_status + 1) // 2,
            source_name_format="{0:d}",
        ),
        "mark_start": mark_start,
        "mark_end": mark_end,
        "path_substitutions": get_path_substitutions(
            configuration, sphotref_header
        ),
    }

    if configuration["master_photref_fname"] is not None:
        _logger.info(
            "Using existing master photometric reference: %s with\n\t%s",
            configuration["master_photref_fname"],
            "\n\t".join(f"{k}: {v!r}" for k, v in kwargs.items()),
        )
        assert start_status == -1
        magnitude_fitting.single_iteration(
            photref=magnitude_fitting.get_master_photref(
                configuration["master_photref_fname"]
            ),
            **kwargs,
        )
        return None

    _logger.info(
        "Starting iterative magfit for single photref: %s with\n\t%s",
        configuration["single_photref_dr_fname"],
        "\n\t".join(f"{k}: {v!r}" for k, v in kwargs.items()),
    )

    master_photref_fname, magfit_stat_fname = magnitude_fitting.iterative_refit(
        single_photref_dr_fname=configuration["single_photref_dr_fname"],
        catalog_sources=catalog_sources,
        **kwargs,
    )
    return [
        {
            "filename": master_photref_fname,
            "preference_order": None,
            "type": "master_photref",
        },
        {
            "filename": magfit_stat_fname,
            "preference_order": None,
            "type": "magfit_stat",
        },
        {
            "filename": catalog_fname,
            "preference_order": None,
            "type": "magfit_catalog",
        },
    ]


def delete_master(filename, master_type):
    """Delete the master from file system and database."""

    # False positivie
    # pylint: disable=no-member
    with Session.begin() as db_session:
        # pylint: enable=no-member
        assert (
            # False positive
            # pylint: disable=not-callable
            db_session.scalar(
                select(func.count())
                .select_from(MasterFile)
                .filter_by(filename=filename)
            )
            == 0
            # pylint: enable=not-callable
        )

    if os.path.exists(filename):
        _logger.warning(
            "Removing potentially partial %s file '%s'!",
            master_type,
            filename,
        )
        os.remove(filename)


def check_no_master(filename, master_type):
    """Raise an exception if the given master exists."""

    if os.path.exists(filename):
        raise RuntimeError(
            f"{master_type} file {filename!r} should not exist!"
            "Cleaning up interrupted magnitude fitting failed!"
        )


def clean_dr(dr_fname, dr_substitutions):
    """Remove a magfit iteration from the given DR file."""

    _logger.warning(
        "Deleting magfit iteration %d from '%s'!",
        dr_substitutions["magfit_iteration"],
        dr_fname,
    )
    with DataReductionFile(dr_fname, "r+") as dr_file:
        dr_file.delete_dataset("shapefit.magfit.magnitude", **dr_substitutions)
        for aperture_index in count():
            if not dr_file.check_for_dataset(
                "apphot.magnitude",
                aperture_index=aperture_index,
                **dr_substitutions,
            ):
                break
            dr_file.delete_dataset(
                "apphot.magfit.magnitude",
                aperture_index=aperture_index,
                **dr_substitutions,
            )


def cleanup_interrupted(interrupted, configuration):
    """Remove DR entries stats and magfit references for magfit_iteration."""

    min_status = min(interrupted, key=lambda x: x[1])[1]
    max_status = max(interrupted, key=lambda x: x[1])[1]
    _logger.info(
        "Cleaning up interrupted magfit with status %d to %d",
        min_status,
        max_status,
    )

    if min_status < max_status - 1 - max_status % 2:
        raise ValueError(
            "Encountered internally inconsistent interrupted status values when"
            " cleaning up magnitude fitting "
            f"(min: {min_status}, max: {max_status})!"
        )

    with DataReductionFile(
        configuration["single_photref_dr_fname"], "r+"
    ) as single_photref_dr:
        sphotref_header = single_photref_dr.get_frame_header()
        fname_substitutions = dict(sphotref_header)
        dr_substitutions = get_path_substitutions(
            configuration, sphotref_header
        )
        fname_substitutions.update(dr_substitutions)

    dr_substitutions["magfit_iteration"] = max_status // 2
    for dr_fname, _ in interrupted:
        clean_dr(dr_fname, dr_substitutions)

    if configuration["master_photref_fname"] is None:
        for master_type in ["master_photref", "magfit_stat"]:
            fname_substitutions["magfit_iteration"] = max_status // 2
            delete_master(
                configuration[master_type + "_fname_format"].format_map(
                    fname_substitutions
                ),
                master_type,
            )

            for iteration in range(0, max_status // 2):
                fname_substitutions["magfit_iteration"] = iteration
                check_fname = configuration[
                    master_type + "_fname_format"
                ].format_map(fname_substitutions)
                _logger.debug("Checking existence of %s", repr(check_fname))
                assert os.path.exists(check_fname)
            fname_substitutions["magfit_iteration"] = max_status // 2 + 1
            check_no_master(
                configuration[master_type + "_fname_format"].format_map(
                    fname_substitutions
                ),
                master_type,
            )

    return max_status - 1 - max_status % 2


def has_apphot(dr_fname, substitutions):
    """Check if the DR file contains a sky-to-frame transformation."""

    with DataReductionFile(dr_fname, mode="r") as dr_file:
        try:
            dr_file.check_for_dataset("apphot.magnitude", **substitutions)
            return True
        except IOError:
            return False


def main():
    """Run the step from command line."""

    configuration = parse_command_line()
    setup_process(task="manage", **configuration)
    with DataReductionFile(
        configuration["single_photref_dr_fname"], "r+"
    ) as single_photref_dr:
        dr_substitutions = get_path_substitutions(
            configuration, single_photref_dr.get_frame_header()
        )
        dr_substitutions["aperture_index"] = (
            single_photref_dr.get_num_apertures(**dr_substitutions) - 1
        )

    fit_magnitudes(
        [
            dr_fname
            for dr_fname in find_dr_fnames(
                configuration.pop("dr_files"),
                configuration.pop("magfit_only_if"),
            )
            if has_apphot(dr_fname, dr_substitutions)
        ],
        None,
        configuration,
        ignore_progress,
        ignore_progress,
    )


if __name__ == "__main__":
    main()
