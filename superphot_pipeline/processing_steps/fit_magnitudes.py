#!/usr/bin/env python3

"""Apply magnitude fitting to hdf5 files"""

from types import SimpleNamespace

from superphot_pipeline import magnitude_fitting
from superphot_pipeline.file_utilities import find_dr_fnames
from superphot_pipeline.processing_steps.manual_util import\
    ManualStepArgumentParser

def parse_command_line():
    """Return the parsed command line arguments."""

    parser = ManualStepArgumentParser(
        description=__doc__,
        input_type='dr',
        inputs_help_extra=('The corresponding DR files must alread contain all '
                           'photometric measurements.'),
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
        '--master-catalogue-fname',
        default='magfit_catalogue.ucac4',
        help='The name of the catalogue file to use as extra information in '
             'magnitude fitting terms and for excluding sources from the fit.'
             'Default: %(default)s'
    )
    parser.add_argument(
        '--master-photref-fname-format',
        default='MASTERS/mphotref_iter%(magfit_iteration)03d.fits',
        help='A format string involving a {magfit_iteration} substitution along'
        ' with any variables from the header of the single photometric '
        'reference or passed through the path_substitutions arguments, that '
        'expands to the name of the file to save the master photometric '
        'reference for a particular iteration.'
    )
    parser.add_argument(
        '--magfit-stat-fname-format',
        default='MASTERS/mfit_stat_iter%(magfit_iteration)03d.txt',
        help='Similar to ``master_photref_fname_format``, but defines the name'
        ' to use for saving the statistics of a magnitude fitting iteration.'
    )
    parser.add_argument(
        '--correction-parametrization',
        type=str,
        default=' + '.join(
            [
                'O4{xi, eta}',
                'O2{R} * O1{J-K} * O2{xi, eta}',
                (
                    ' * '.join([
                        '{1, '
                        +
                        ', '.join([
                            (
                                'sin(%(freq).1f * pi * (%(coord)c %% 1)), '
                                'cos(%(freq).1f * pi * (%(coord)c %% 1))'
                            )
                            %
                            dict(freq=(2 * freq), coord=coord)
                            for freq in range(1, 4)
                        ])
                        +
                        '}'
                        for coord in 'xy'
                    ])
                )
            ]
        ),
        help='A string that expands to the terms to include in the magnitude '
        'fitting correction.'
    )
    parser.add_argument(
        '--mphotref-scatter-fit-terms',
        default='O2{r,xi,eta}',
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
        default='(r > 0) * (r < 16) * (J - K > 0) * (J - K < 1)',
        help='An expression involving catalogue, reference and/or photometry '
        'variables which evaluates to zero if a source should be excluded and '
        'any non-zero value if it  should be included in the magnitude fit.'
    )
    parser.add_argument(
        '--grouping',
        action='append',
        help=(
            'An expressions using catalogue, and/or photometry variables which '
            'evaluates to a tuple of boolean values. Each distinct tuple '
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
             'it is rejected in utins of error_avg. Default: %(default)s'
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
    return parser.parse_args()


def magnitude_fit(dr_collection, configuration):
    """Perform magnitude fitting for the given DR files."""

    path_substitutions = {what + '_version': configuration[what + '_version']
                          for what in ['shapefit',
                                       'srcproj',
                                       'apphot',
                                       'background',
                                       'magfit']}

    magnitude_fitting.iterative_refit(
        fit_dr_filenames=sorted(dr_collection),
        single_photref_dr_fname=configuration['single_photref_dr_fname'],
        master_catalogue_fname=configuration['master_catalogue_fname'],
        configuration=SimpleNamespace(**configuration),
        master_photref_fname_format=(
            configuration['master_photref_fname_format']
        ),
        magfit_stat_fname_format=configuration['magfit_stat_fname_format'],
        master_scatter_fit_terms=configuration['mphotref_scatter_fit_terms'],
        max_iterations=configuration['max_magfit_iterations'],
        **path_substitutions
    )


if __name__ == '__main__':
    cmdline_config = parse_command_line()
    if cmdline_config['grouping'] is not None:
        cmdline_config['grouping'] = (
            '('
            +
            ', '.join(cmdline_config['grouping'])
            +
            ')'
        )
    magnitude_fit(
        find_dr_fnames(cmdline_config.pop('dr_files'),
                       cmdline_config.pop('magfit_only_if')),
        cmdline_config
    )
