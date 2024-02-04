#!/usr/bin/env python3

"""Apply magnitude fitting to hdf5 files"""

from types import SimpleNamespace
from itertools import count
import os

from general_purpose_python_modules.multiprocessing_util import setup_process

from superphot_pipeline import magnitude_fitting, DataReductionFile
from superphot_pipeline.file_utilities import find_dr_fnames
from superphot_pipeline.catalog import ensure_catalog
from superphot_pipeline.processing_steps.manual_util import\
    ManualStepArgumentParser,\
    ignore_progress,\
    get_catalog_config

input_type = 'dr'


def parse_command_line(*args):
    """Return the parsed command line arguments."""

    parser = ManualStepArgumentParser(
        description=__doc__,
        input_type=('' if args else input_type),
        inputs_help_extra=('The corresponding DR files must alread contain all '
                           'photometric measurements.'),
        add_catalog={'prefix': 'magfit'},
        add_component_versions=('srcproj',
                                'background',
                                'shapefit',
                                'apphot',
                                'magfit'),
        allow_parallel_processing=True
    )
    parser.add_argument(
        '--magfit-only-if',
        default='True',
        help='Expression involving the header of the input images that '
        'evaluates to True/False if a particular image from the specified '
        'image collection should/should not be processed.'
    )
    parser.add_argument(
        '--single-photref-dr-fname',
        default='single_photref.hdf5.0',
        help='The name of the data reduction file of the single photometric '
        'reference to use to start the magnitude fitting iterations.'
    )
    parser.add_argument(
        '--master-photref-fname-format',
        default='MASTERS/mphotref_iter{magfit_iteration:03d}.fits',
        help='A format string involving a {magfit_iteration} substitution along'
        ' with any variables from the header of the single photometric '
        'reference or passed through the path_substitutions arguments, that '
        'expands to the name of the file to save the master photometric '
        'reference for a particular iteration.'
    )
    parser.add_argument(
        '--magfit-stat-fname-format',
        default='MASTERS/mfit_stat_iter{magfit_iteration:03d}.txt',
        help='Similar to ``master_photref_fname_format``, but defines the name'
        ' to use for saving the statistics of a magnitude fitting iteration.'
    )
    parser.add_argument(
        '--correction-parametrization',
        type=str,
        default='O3{phot_g_mean_mag, x, y}',
        help='A string that expands to the terms to include in the magnitude '
        'fitting correction.'
    )
    parser.add_argument(
        '--mphotref-scatter-fit-terms',
        default='O2{phot_g_mean_mag}',
        help='Terms to include in the fit for the scatter when deciding which '
        'stars to include in the master.'
    )
    parser.add_argument(
        '--reference-subpix',
        action='store_true',
        default=False,
        help='Should the magnitude fitting correction depend on the '
             'sub-pixel position of the source in the reference frame'
             'Default: %(default)s'
    )
    parser.add_argument(
        '--fit-source-condition',
        type=str,
        default='isfinite(phot_g_mean_mag)',
        help='An expression involving catalog, reference and/or photometry '
        'variables which evaluates to zero if a source should be excluded and '
        'any non-zero value if it  should be included in the magnitude fit.'
    )
    parser.add_argument(
        '--grouping',
        default=None,
        help=(
            'An expression using catalog, and/or photometry variables which '
            'evaluates to several distinct values. Each value '
            'defines a separate fitting group (i.e. a group of sources which '
            'participate in magnitude fitting together, excluding sources '
            'belonging to other groups). Default: %(default)s'
        )
    )
    parser.add_argument(
        '--error-avg',
        default='weightedmean',
        help='How to average fitting residuals for outlier rejection.'
    )
    parser.add_argument(
        '--rej-level',
        type=float,
        default=5.0,
        help='How far away from the fit should a point be before '
             'it is rejected in units of error_avg. Default: %(default)s'
    )
    parser.add_argument(
        '--max-rej-iter',
        type=int,
        default=20,
        help='The maximum number of rejection/re-fitting iterations to perform.'
        ' If the fit has not converged by then, the latest iteration is '
        'accepted.'
    )
    parser.add_argument(
        '--noise-offset',
        type=float,
        default=0.01,
        help='Additional offset to format magnitude error estimates when they '
        'are used to determine the fitting weights. '
    )
    parser.add_argument(
        '--max-mag-err',
        type=float,
        default=0.1,
        help='The largest the formal magnitude error is allowed '
             'to be before the source is excluded. Default: %(default)s'
    )
    parser.add_argument(
        '--max-photref-change',
        type=float,
        default=1e-4,
        help='The maximum square average change of photometric reference '
             'magnitudes to consider the iterations converged.'
    )
    parser.add_argument(
        '--max-magfit-iterations',
        type=int,
        default=5,
        help='The maximum number of iterations of deriving a master photometric'
        ' referene and re-fitting to allow.'
    )
    return parser.parse_args(*args)


def get_path_substitutions(configuration):
    """Return the path substitutions to find magfit datasets."""

    return {what + '_version': configuration[what + '_version']
            for what in ['shapefit',
                         'srcproj',
                         'apphot',
                         'background',
                         'magfit']}


def magnitude_fit(dr_collection,
                  start_status,
                  configuration,
                  mark_start,
                  mark_end):
    """Perform magnitude fitting for the given DR files."""

    if start_status is None:
        start_status = -1
    else:
        assert start_status % 2 == 1

    dr_fnames = sorted(dr_collection)
    magnitude_fitting.iterative_refit(
        fit_dr_filenames=dr_fnames,
        single_photref_dr_fname=configuration['single_photref_dr_fname'],
        catalog_sources=ensure_catalog(
            dr_files=dr_fnames,
            configuration=get_catalog_config(configuration, 'magfit'),
            return_metadata=False,
            skytoframe_version=configuration['skytoframe_version']
        ),
        configuration=SimpleNamespace(**configuration),
        master_photref_fname_format=(
            configuration['master_photref_fname_format']
        ),
        magfit_stat_fname_format=configuration['magfit_stat_fname_format'],
        master_scatter_fit_terms=configuration['mphotref_scatter_fit_terms'],
        mark_start=mark_start,
        mark_end=mark_end,
        max_iterations=configuration['max_magfit_iterations'],
        continue_from_iteration=(start_status + 1) // 2,
        **get_path_substitutions(configuration)
    )


def cleanup_interrupted(interrupted, configuration):
    """Remove DR entries stats and magfit references for magfit_iteration."""

    dr_substitutions = get_path_substitutions(configuration)

    min_status = min(interrupted, key=lambda x: x[1])[1]
    max_status = max(interrupted, key=lambda x: x[1])[1]

    if min_status < max_status - 1 - max_status % 2:
        raise ValueError(
            'Encountere internally inconsistent interrupted status values when '
            'cleaning up magnitude fitting!'
        )

    with DataReductionFile(
        configuration['single_photref_dr_fname'],
        'r+'
    ) as single_photref_dr:
        fname_substitutions = dict(single_photref_dr.get_frame_header())
        fname_substitutions.update(dr_substitutions)


    for master_type in ['master_photref', 'magfit_stat']:
        fname_substitutions['magfit_iteration'] = max_status // 2
        to_delete = configuration[master_type + '_fname_format'].format_map(
            fname_substitutions
        )
        if os.path.exists(to_delete):
            os.remove(to_delete)
        for iteration in range(0, max_status // 2):
            fname_substitutions['magfit_iteration'] = iteration
            assert os.path.exists(
                configuration[master_type + '_fname_format'].format_map(
                    fname_substitutions
                )
            )
        fname_substitutions['magfit_iteration'] = max_status // 2 + 1
        if os.path.exists(
            configuration[master_type + '_fname_format'].format_map(
                fname_substitutions
            )
        ):
            raise RuntimeError(
                f'{master_type} file {to_delete!r} should not exist!'
                'Cleaning up interrupted magnitude fitting failed!'
            )

    for dr_fname, from_status in interrupted:
        if from_status % 2:
            continue

        dr_substitutions['magfit_iteration'] = from_status // 2
        with DataReductionFile(dr_fname, 'r+') as dr_file:
            dr_file.delete_dataset('shapefit.magfit.magnitude',
                                   **dr_substitutions)
            for aperture_index in count():
                if not dr_file.check_for_dataset('apphot.magnitude',
                                                 aperture_index=aperture_index,
                                                 **dr_substitutions):
                    break
                dr_file.delete_dataset('apphot.magfit.magnitude',
                                       aperture_index=aperture_index,
                                       **dr_substitutions)


if __name__ == '__main__':
    cmdline_config = parse_command_line()
    setup_process(task='manage', **cmdline_config)
    magnitude_fit(
        find_dr_fnames(cmdline_config.pop('dr_files'),
                       cmdline_config.pop('magfit_only_if')),
        None,
        cmdline_config,
        ignore_progress,
        ignore_progress
    )
