#!/usr/bin/env python3

"""Apply magnitude fitting to hdf5 files"""

from configargparse import ArgumentParser, DefaultsFormatter
import logging
from command_line_util import add_filename_options
from superphot_pipeline import magnitude_fitting
import filename_patterns


def parse_command_line():
    """Return the parsed command line arguments."""

    parser = ArgumentParser(
        description=__doc__,
        default_config_files=['configurations/R1magfit.cfg'],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=True
    )
    parser.add_argument(
        'frames',
        nargs='+',
        help='A list of the data reduction files to fit.'
    )
    add_filename_options(
        parser,
        add_all=True
    )
    parser.add_argument(
        '--single-photref-dr-fname',
        default='/data/HAT10DSLR_reprocess/FITPSF_single_O3pos_frame_center/10-465248_2_R1.hdf5.0',
        help='The name of the data reduction file of the single photometric '
             'reference to use to start the magnitude fitting iterations.'
             'Default: %(default)s'
    )
    parser.add_argument(
        '--master-catalogue-fname',
        default=filename_patterns.catalogue,
        help='The name of the catalogue file to use as extra information in '
             'magnitude fitting terms and for excluding sources from the fit.'
             'Default: %(default)s'
    )
    parser.add_argument(
        '--master-photref-fname-pattern',
        default='/data/HAT10DSLR_reprocess/MASTERS/mphotref_iter%(magfit_iteration)03d.fits',
        help='A %-substitution pattern involving a %(magfit_iteration)s substitution '
             'along with any variables passed through the path_substitutions arguments, '
             'that expands to the name of the file to save the master photometric reference '
             'for a particular iteration.'
             'Default: %(default)s'
    )
    parser.add_argument(
        '--magfit-stat-fname-pattern',
        default='/data/HAT10DSLR_reprocess/MASTERS/mfit_stat_iter%(magfit_iteration)03d.txt',
        help='Similar to ``master_photref_fname_pattern``, but defines the name '
             'to use for saving the statistics of a magnitude fitting iteration.'
             'Default: %(default)s'
    )
    parser.add_argument(
        '--correction-parametrization',
        type=str,
        default='O4{xi, eta} + O2{R} * O2{xi, eta} + O1{J-K} * O1{xi, eta} + O1{x % 1, y % 1}',
        help='A string that expands to the terms to include in the magnitude fitting correction.'
             'Default: %(default)s'
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
        help='An expression involving catalogue, reference and/or photometry variables which '
             'evaluates to zero if a source should be excluded and any non-zero value if it '
             'should be included in the magnitude fit.'
             'Default: %(default)s'
    )
    parser.add_argument(
        '--grouping',
        action='append',
        help='An expressions using catalogue, and/or photometry variables which evaluates to a '
             'tuple of boolean values. Each distinct tuple defines a separate fitting group '
             '(i.e. a group of sources which participate in magnitude fitting together, excluding '
             'sources belonging to other groups).'
             'Default: %(default)s'
    )
    parser.add_argument(
        '--error-avg',
        default='weightedmean',
        help='How to average fitting residuals for outlier rejection. Default: %(default)s'
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
        help='The maximum number of rejection/re-fitting iterations to perform. '
             'If the fit has not converged by then, the latest iteration is accepted.'
             'Default: %(default)s'
    )
    parser.add_argument(
        '--noise-offset',
        type=float,
        default=0.01,
        help='Additional offset to format magnitude error estimates '
             'when they are used to determine the fitting weights. '
             'Default: %(default)s'
    )
    parser.add_argument(
        '--max-mag-err',
        type=float,
        default=0.1,
        help='The largest the formal magnitude error is allowed '
             'to be before the source is excluded. Default: %(default)s'
    )
    parser.add_argument(
        '--num-parallel-processes',
        type=int,
        default=1,
        help='How many processes to use for simultaneus fitting. Default: %(default)s'
    )
    parser.add_argument(
        '--max-photref-change',
        type=float,
        default=1e-4,
        help='The maximum square average change of photometric reference '
             'magnitudes to consider the iterations converged. Default: %(default)s'
    )
    parser.add_argument(
        '--verbose',
        default='info',
        help='The type of verbosity of logger: Options are info, debug, warning, error, or critical.'
             'Default: %(default)s'
    )
    loglevel = dict(critical=logging.CRITICAL,
                    debug=logging.DEBUG,
                    error=logging.ERROR,
                    info=logging.INFO,
                    warning=logging.WARNING
                    )
    arguments = parser.parse_args()
    arguments.verbose = loglevel[arguments.verbose]
    return arguments


if __name__ == '__main__':
    cmdline_args = parse_command_line()
    cmdline_args.grouping = '(' + ', '.join(cmdline_args.grouping) + ')'
    logging.basicConfig(level=cmdline_args.verbose)
    path_substitutions = dict(shapefit_version=0,
                              srcproj_version=0,
                              apphot_version=0,
                              background_version=0,
                              magfit_version=0)

    magnitude_fitting.iterative_refit(
        fit_dr_filenames=sorted(cmdline_args.frames),
        single_photref_dr_fname=cmdline_args.single_photref_dr_fname,
        master_catalogue_fname=cmdline_args.master_catalogue_fname,
        configuration=cmdline_args,
        master_photref_fname_pattern=cmdline_args.master_photref_fname_pattern,
        magfit_stat_fname_pattern=cmdline_args.magfit_stat_fname_pattern,
        **path_substitutions
    )
