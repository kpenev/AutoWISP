#!/usr/bin/env python3

"""Fit a model for the shape of stars (PSF or PRF) in images."""

from multiprocessing import Pool

import numpy
import pandas

from superphot_pipeline import Evaluator, PiecewiseBicubicPSFMap
from superphot_pipeline.astrometry import Transformation
from superphot_pipeline.image_utilities import find_fits_fnames
from superphot_pipeline.fits_utilities import get_primary_header
from superphot_pipeline.processing_steps.manual_util import\
    ManualStepArgumentParser,\
    add_image_options,\
    read_catalogue,\
    read_subpixmap
from superphot_pipeline.split_sources import SplitSources

def parse_grid_arg(grid_str):
    """Parse the string specifying the grid on which to model PSF/PRF."""

    grid_str = grid_str.strip('\'"')
    if ';' not in grid_str:
        grid_str = ';'.join([grid_str, grid_str])
    return [
        [
            float(value)
            for value in sub_grid.split(',')
        ]
        for sub_grid in grid_str.strip('"\'').split(';')
    ]


def add_background_options(parser):
    """Add options configuring the background extraction."""

    parser.add_argument(
        '--background-annulus',
        nargs=2,
        type=float,
        default=[6, 7],
        help='The annulus to use when estimating the background under '
        'sources.'
    )


def add_source_selction_options(parser):
    """Add options configuring the selection of sources for fitting."""

    parser.add_argument(
        '--shapefit-disable-cover-grid',
        dest='shapefit_src_cover_grid',
        action='store_false',
        default=True,
        help='Should pixels be selected to cover the full PSF/PRF grid.'
    )
    parser.add_argument(
        '--shapefit-src-min-bg-pix',
        type=int,
        default=50,
        help='The minimum number of pixels background should be based on if'
        ' the source is to be used for shape fitting.'
    )
    parser.add_argument(
        '--shapefit-src-max-sat-frac',
        type=float,
        default=0.0,
        help='The maximum fraction of saturated pixels of a source before '
        'it is discarded from shape fitting.'
    )
    parser.add_argument(
        '--shapefit-src-min-signal-to-noise',
        type=float,
        default=3.0,
        help='The S/N threshold when selecting pixels around sources. '
        'Ignored if --shapefit-cover-grid.'
    )
    parser.add_argument(
        '--shapefit-src-max-aperture',
        type=float,
        default=10.0,
        help='The largest distance from the source center before pixel '
        'selection causes an error.'
    )
    parser.add_argument(
        '--shapefit-src-min-pix',
        type=int,
        default=5,
        help='The smallest number of pixels to require be assigned for a '
        'source if it is to be included in the shape fit.'
    )
    parser.add_argument(
        '--shapefit-src-max-pix',
        type=int,
        default=1000,
        help='The largest number of pixels to require be assigned for a '
        'source if it is to be included in the shape fit.'
    )
    parser.add_argument(
        '--shapefit-max-sources',
        type=int,
        default=10000,
        help='The maximum number of sources to include in the fit. Excess '
        'sources (those with lowest signal to noise) are not included in '
        'the shape fit, though still get photometry measured.'
    )
    parser.add_argument(
        '--discard-faint',
        default=None,
        help='If used, should indicate a faint magnitude limit in some '
        'band-pass, e.g. B>14.0. Sources fainter than the specified limit '
        'will not be included in the input source lists for PRF fitting at '
        'all.'
    )


def add_fitting_options(parser):
    """Add options controlling the fitting process."""

    parser.add_argument(
        '--shapefit-smoothing',
        type=float,
        default=None,
    )
    parser.add_argument(
        '--shapefit-max-chi2',
        type=float,
        default=100,
    )
    parser.add_argument(
        '--shapefit-pixel-rejection-threshold',
        type=float,
        default=1000,
        help='Pixels away from best fit values by more than this many '
        'sigma are discarded from the fit.'
    )
    parser.add_argument(
        '--shapefit-max-abs-amplitude-change',
        type=float,
        default=0,
        help='If the absolute sum square change in amplitudes falls below '
        'this, the fit is declared converged.'
    )
    parser.add_argument(
        '--shapefit-max-rel-amplitude-change',
        type=float,
        default=1e-5,
        help='If the relative sum square change in amplitudes falls below '
        'this, the fit is declared converged.'
    )
    parser.add_argument(
        '--shapefit-min-convergence-rate',
        type=float,
        default=-10.0,
        help='If the rate of convergence of the amplitudes falls below '
        'this, an error is thrown.'
    )
    parser.add_argument(
        '--shapefit-max-iterations',
        type=int,
        default=1000,
        help='The maximum number of shape-amplitude fitting iterations to '
        'allow.'
    )
    parser.add_argument(
        '--shapefit-initial-aperture',
        type=float,
        default=5.0,
        help='The aperture to use when estimating the initial flux of '
        'sources to start the first shape-amplitude fitting iteration.'
    )
    parser.add_argument(
        '--num-simultaneous',
        type=int,
        default=1,
        help='The number of frames to fit simultaneously, with a unified '
        'PSF/PRF model. Each simultaneous group consists of consecutive '
        'entries in the input list of frames.'
    )


def add_shape_options(parser):
    """Add options defining how the PSF/PRF shape will be modeled."""

    parser.add_argument(
        '--shape-mode',
        default='psf',
        help='Is the mode representing PSF or PRF?'
    )
    parser.add_argument(
        '--shape-grid',
        default='-3,-2,-1,0,1,2,3',
        type=parse_grid_arg,
        help='The grid to use for representing the PSF/PRF.'
    )
    parser.add_argument(
        '--shape-terms-expression', '--shape-terms',
        default='O0{(x-1991.5)/1991.5, (y-1329.5)/1329.5}',
        help='The term in the PSF shape parameter dependence.'
    )
    parser.add_argument(
        '--map-variables',
        metavar='<varname>, <expression>',
        nargs='+',
        default=[],
        help='Extra variables to allow the PRF to depend on in addition to '
        '(x and y). The <expression> can involve any catologue column , '
        'header variable, and `STID`, `FNUM`, `CMPOS`. The extra variables '
        'are added as extra columns after ID, x, y to the source list '
        'passed to fitpsf/fitprf in the order specified on the command '
        'line. Make sure to include in the input-columns option in '
        '--superphot-config-file.'
    )


def add_grouping_options(parser):
    """Add options controlling splitting of sources into fitting groups."""

    parser.add_argument(
        '--split-magnitude-column',
        default='B',
        help='The catalogue column to use as the brightness indicator of '
        'the sources when splitting into groups.'
    )
    parser.add_argument(
        '--radius-splits',
        nargs='+',
        type=float,
        default=[],
        help='The threshold radius values where to split sources into '
        'groups. By default, no splitting by radius is done.'
    )
    parser.add_argument(
        '--mag-split-source-count',
        type=int,
        default=None,
        help='If passed, after spltting by radius (if any), sources are '
        'further split into groups by magnitude such that each group '
        'contains at least this many sources. By default, no splitting by '
        'magnitude is done.'
    )
    parser.add_argument(
        '--grouping-frame',
        default=None,
        help='If sources are being split per any of the above arguments, '
        'specifying a frame here results in the split being done based on '
        'the locations of sources in this frame and thus does not change '
        'from frame to frame. If not specified, grouping is done '
        'independently for each frame.'
    )
    parser.add_argument(
        '--remove-group-id',
        type=int,
        default=None,
        nargs='+',
        help='If passed, this will remove the groups to fit in an indexable'
        ' fashion. Multiple values may be passed e.g. 0 1 5 where each is '
        'the index corresponding to the group_id'
    )


def parse_command_line():
    """Return the parsed command line arguments."""

    parser = ManualStepArgumentParser(
        description=__doc__,
        input_type='calibrated + dr',
        inputs_help_extra=('The corresponding DR files must alread contain an '
                           'astrometric transformation.'),
        add_component_versions=('srcproj', 'background', 'shapefit'),
        allow_parallel_processing=True
    )
    parser.add_argument(
        '--shapefit-only-if',
        default='True',
        help='Expression involving the header of the input images that '
        'evaluates to True/False if a particular image from the specified '
        'image collection should/should not be processed.'
    )
    parser.add_argument(
        '--skytoframe-version',
        type=int,
        default=0,
        help='The version of the sky -> frame transformation to use for '
        'projecting the photometry catalogue.'
    )

    parser.add_argument(
        '--photometry-catalogue', '--photometry-catalog', '--cat',
        required=True,
        help='A file containing the list of stars to perform photometry on.'
    )

    add_image_options(
        parser.add_argument_group('options describing the image')
    )
    add_background_options(
        parser.add_argument_group('background extraction options')
    )
    add_source_selction_options(
        parser.add_argument_group('source selection options')
    )
    add_fitting_options(
        parser.add_argument_group('shape fitting options')
    )
    add_shape_options(
        parser.add_argument_group('shape model options')
    )
    add_grouping_options(
        parser.add_argument_group('options controlling splitting of sources in '
                                  'fitting groups')
    )
    return parser.parse_args()


#Goal is to provide callable
#pylint: disable=too-few-public-methods
class SourceListCreator:
    """Class for creating PRF fitting source lists for a single frame."""

    def _project_sources(self, header):
        """
        Add to `self._sources` the projected positions and extra fit variables.

        Args:
            header:    The header of the FITS frame currently being processed.

        Returns:
            None
        """

        Transformation(
            self._dr_fname_format.format_map(header),
            **self._dr_path_substitutions
        )(
            self._sources,
            True,
            True
        )
        eval_var = Evaluator(header, self._sources)
        for var_name, var_expression in self._fit_variables:
            self._sources[var_name] = eval_var(var_expression)


    def _group_and_flag_in_frame(self, header):
        """
        Return the group each source belongs to and flag for is source in frame.

        Args:
            header:    The FITS header of the frame being fit.

        Returns:
            numpy.array(int):
                The group index for each source

            numpy.array(bool):
                True iff the source center is inside the frame boundaries
        """

        print('Projecting sources')
        self._project_sources(header)
        print('Projected sources:\n' + repr(self._sources))

        print('Grouping')
        if callable(self._grouping):
            return self._grouping(
                self._sources,
                (header['NAXIS2'], header['NAXIS1'])
            )

        return (
            self._grouping,
            numpy.logical_and(
                numpy.logical_and(
                    self._sources['x'] > 0,
                    self._sources['x'] < header['NAXIS1'],
                ),
                numpy.logical_and(
                    self._sources['y'] > 0,
                    self._sources['y'] < header['NAXIS2'],
                )
            )
        )


    def __init__(self,
                 *,
                 dr_fname_format,
                 catalogue_fname,
                 fit_variables,
                 grouping,
                 grouping_frame=None,
                 discard_faint=False,
                 remove_group_id=None,
                 **dr_path_substitutions):
        """
        Set up to create source lists for PSF/PRF fitting.

        Args:
            dr_fname_format:    A format string to generate the name of the
                data reduction file that corresponds to each frame.

            catalogue_fname:    The filename containing a list of catalogue
                sources to fit the shape and measure the brightness of.

            extra_fit_variables:    See --map-variables command line argument.

            grouping:    A splitting of the input sources in groups, each of
                which is enabled separately during PRF fitting. Should be a
                callable taking the input projected sources, catalogue
                information, frame header and extra fit variables and returning
                A numpy integer array indicating for each source the PRF fitting
                group it is in. Sources assigned to negative group IDs are never
                enabled.

            grouping_frame:    If None, grouping is derived for each input
                frames separately, potentially resulting in a different set of
                sources enabled for each frame. If not None, it should specify
                the filename of a FITS frame on which to derive the grouping
                once and all subsequent fits are based on the same exact
                sources, regardless of where on the frame they appear, as long
                as they are within the frame.

            discard_faint:    See `--discard-faint` command line argument.

            remove_group_id:    See '--remove-group-id' command line argument.

            dr_path_substitutions:    Any keywords needed to specify unique
                paths in the data reduction files for the inputs and output
                required for shape fitting.

        Returns:
            None
        """

        self._sources = read_catalogue(catalogue_fname)
        if discard_faint is not None:
            discard_filter, faint_limit = discard_faint.split('>')
            faint_limit = float(faint_limit)
            self._sources = self._sources[
                self._sources[discard_filter] <= faint_limit
            ]

        self._fit_variables = fit_variables

        self._dr_fname_format = dr_fname_format
        self._dr_path_substitutions = dr_path_substitutions

        print('Sources: ' + repr(self._sources))

        self._id_length = max(
            len(id_value) for id_value in self._sources.index
        )
        self.remove_group_id = remove_group_id

        if grouping_frame:
            header = get_primary_header(grouping_frame, True)
            self._project_sources(header)
            self._grouping = grouping(
                self._sources,
                #False positive
                #pylint: disable=unsubscriptable-object
                (header['NAXIS2'], header['NAXIS1'])
                #pylint: enable=unsubscriptable-object
            )[0]
        else:
            self._grouping = grouping


    def __call__(self, frame_fname):
        """
        Return the `fitpsf`/`fitprf` source list for this frame.
        Args:
            frame_fname:    The filename of the frame to get PRF fitting
                sources of.
        Returns:
            [numpy record array]:
                The values of all source variables PSF/PRF fitting will use for
                each fitting group. Each entry is suitable as input to
                FitStarShape.fit(), with only one fitting group enabled.
        """

        print('Getting sources from ' + repr(frame_fname))
        header = get_primary_header(frame_fname, True)

        grouping, in_frame = self._group_and_flag_in_frame(header)
        print(
            'Found {0:d}/{1:d} sources inside the frame.'.format(
                in_frame.sum(),
                len(in_frame)
            )
        )
        fit_sources = self._sources[in_frame]
        print('Fit source columns: ' + repr(self._sources.columns))
        grouping = grouping[in_frame]

        number_fit_groups = grouping.max() + 1

        if self.remove_group_id is not None:
            number_fit_groups = sorted(range(number_fit_groups))
            for remove_group_id in self.remove_group_id:
                print('Removing group_id: ' + repr(remove_group_id))
                del number_fit_groups[remove_group_id]
            result = [pandas.DataFrame(fit_sources, copy=True)
                      for group_id in number_fit_groups]
            for group_id in number_fit_groups:
                print('Group ' + str(group_id) + ':\n' + repr(result[group_id]))
                result[group_id]['enabled'] = (grouping == group_id)
            print(result)
        else:
            result = [pandas.DataFrame(fit_sources, copy=True)
                      for group_id in range(number_fit_groups)]
            for group_id in range(number_fit_groups):
                print('Group ' + str(group_id) + ':\n' + repr(result[group_id]))
                result[group_id]['enabled'] = (grouping == group_id)

            print(result)
        return result
#pylint: enable=too-few-public-methods


def create_source_list_creator(configuration):
    """Return a fully configured instance of SourceListCreator."""

    return SourceListCreator(
        dr_fname_format=configuration['data_reduction_fname'],
        catalogue_fname=configuration['photometry_catalogue'],
        fit_variables=configuration['map_variables'],
        grouping=SplitSources(
            magnitude_column=configuration['split_magnitude_column'],
            radius_splits=configuration['radius_splits'],
            mag_split_by_source_count=configuration['mag_split_source_count']
        ),
        **{option: configuration[option] for option in ['grouping_frame',
                                                        'discard_faint',
                                                        'remove_group_id',
                                                        'skytoframe_version']}
    )


def get_shape_fitter_config(configuration):
    """Return a fully configured instance of FitStarShape."""

    result = dict(
        require_convergence=False,
        mode=configuration['shape_mode'],
        grid=configuration['shape_grid'],
        bg_min_pix=configuration['shapefit_src_min_bg_pix'],
        cover_grid=configuration['shapefit_src_cover_grid'],
        src_max_count=configuration['shapefit_max_sources'],
        dr_path_substitutions={
            version_name + '_version': configuration[version_name + '_version']
            for version_name in ['background', 'shapefit', 'srcproj']
        }
    )

    if configuration['subpixmap'] is not None:
        result['subpixmap']=read_subpixmap(configuration['subpixmap'])
    for option in ['background_annulus',
                   'gain',
                   'magnitude_1adu',
                   'shape_terms_expression']:
        result[option] = configuration[option]
    for option in ['initial_aperture',
                   'smoothing',
                   'max_chi2',
                   'pixel_rejection_threshold',
                   'max_abs_amplitude_change',
                   'max_rel_amplitude_change',
                   'min_convergence_rate',
                   'max_iterations',
                   'src_min_signal_to_noise',
                   'src_max_sat_frac',
                   'src_max_aperture',
                   'src_min_pix',
                   'src_max_pix']:
        result[option] = configuration['shapefit_' + option]

    return result


def fit_frame_set(frame_filenames_configuration):
    """
    Perform a simultaneous fit of all frames included in frame_filenames.
    Args:
        frame_filenames ([str]):    The list of FITS file containting calibrated
            frames to fit. The files must include at least 3 extensions: the
            calibrated pixel values, estimated errors for the pixel values and
            the pixel quality mask.

        configuration(dict):    The configuration to use for PSF/PRF fitting,
            background extraction etc.

    Returns:
        None
    """


    def get_dr_fname(frame_fname):
        """Return the filename to saving a shape fit."""

        header = get_primary_header(frame_fname, True)
        return configuration['data_reduction_fname'].format_map(header)


    frame_filenames, configuration = frame_filenames_configuration
    print('Fitting frame set')

    get_sources = create_source_list_creator(configuration)
    print('Created source getter')

    shape_fitter_config = get_shape_fitter_config(configuration)
    star_shape_fitter = PiecewiseBicubicPSFMap()
    print('Created star shape fitter.')

    fit_sources = [get_sources(frame) for frame in frame_filenames]
    print('Fit sources: ' + repr(fit_sources))

    num_fit_groups = max(len(frame_sources) for frame_sources in fit_sources)

    for fit_group in range(num_fit_groups):
        shape_fitter_config['dr_path_substitutions']['fit_group'] = fit_group
        print('Fitting: '
              +
              '\tframe_filenames: ' + repr(frame_filenames)
              +
              '\tsources: ' + repr([sources[fit_group] for sources in
                                    fit_sources])
              +
              '\tdr_fnames: ' + repr([get_dr_fname(f) for f in
                                      frame_filenames]))
        star_shape_fitter.fit(
            fits_fnames=frame_filenames,
            sources=[sources[fit_group].to_records()
                     for sources in fit_sources],
            output_dr_fnames=[get_dr_fname(f) for f in frame_filenames],
            **shape_fitter_config
        )
        print('Done fitting')


def fit_star_shapes(image_collection, configuration):
    """Find the best-fit model for the PSF/PRF in the given images."""

    image_collection = list(image_collection)
    frame_list_splits = range(0,
                              len(image_collection),
                              configuration['num_simultaneous'])
    fit_arguments = [
        (
            image_collection[
                split
                :
                min(split + configuration['num_simultaneous'],
                    len(image_collection))
            ],
            configuration
        ) for split in frame_list_splits
    ]

    print('Using %d parallel processes'
          %
          configuration['num_parallel_processes'])
    if configuration['num_parallel_processes'] == 1:
        for args in fit_arguments:
            fit_frame_set(args)
    else:
        pool = Pool(processes=configuration['num_parallel_processes'])
        pool.imap_unordered(
            fit_frame_set,
            fit_arguments
        )
        pool.close()
        pool.join()


if __name__ == '__main__':
    cmdline_config = parse_command_line()
    fit_star_shapes(
        find_fits_fnames(cmdline_config.pop('calibrated_images'),
                         cmdline_config.pop('shapefit_only_if')),
        cmdline_config
    )
