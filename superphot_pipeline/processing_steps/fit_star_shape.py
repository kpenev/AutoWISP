#!/usr/bin/env python3

"""Fit a model for the shape of stars (PSF or PRF) in images."""

from superphot_pipeline.processing_steps.manual_util import get_cmdline_parser

def parse_grid_arg(grid_str):
    """Parse the string specifying the grid on which to model PSF/PRF."""

    if ';' not in grid_str:
        grid_str = ';'.join([grid_str, grid_str])
    grid = [
        [
            float(value)
            for value in sub_grid.split(',')
        ]
        for sub_grid in parsed_args.shape_grid.strip('"\'').split(';')
    ]


def parse_command_line():
    """Return the parsed command line arguments."""

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
            dest='shapefit_src_cover_grid'
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
            '--shapefit-max-sat-frac',
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
        parser.add_argument(
            '--num-parallel-processes',
            type=int,
            default=12,
            help='The number of simultaneous fitpsf/fitprf processes to run.'
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
            '--shape-terms',
            default='O0{(x-1991.5)/1991.5, (y-1329.5)/1329.5}',
            help='The term in the PSF shape parameter dependence.'
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


    parser = get_cmdline_parser(__doc__)
    parser.add_argument(
        'calibrated_images',
        nargs='+',
        help='A combination of individual images and image directories to '
        'process. Directories are not searched recursively.'
    )
    parser.add_argument(
        '--shapefit-only-if',
        default='True',
        help='Expression involving the header of the input images that '
        'evaluates to True/False if a particular image from the specified '
        'image collection should/should not be processed.'
    )
    parser.add_argument(
        '--shapefit-version',
        type=int,
        default=0,
        help='The version to assign to this star shape fit in the DR files.'
    )
    parser.add_argument(
        '--data-reduction-fname',
        default='DR/{RAWFNAME}.h5',
        help='Format string to generate the filename(s) of the data reduction '
        'files where extracted sources are saved. Replacement fields can be '
        'anything from the header of the calibrated image.'
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


class SourceListCreator:
    """Class for creating PRF fitting source lists for a single frame."""

    def _project_sources(self, header):
        """
        Get the frame projected positions of the catalogue sources.

        Args:
            header:    The header of the FITS frame currently being processed.
        Returns:
            numpy.array(dtype=[('x', float), ('y', float)]):
                The projected sources to use for PRF fitting.
        """

        for key in self.catalogue_columns:
            header.pop(key, None)
        self._evaluate_variable_expression.symtable.update(header)
        print('Reading transformation: '
              +
              repr(self.trans_fname_pattern % header))
        return Transformation(
            self.trans_fname_pattern % header
        )(
            self._evaluate_variable_expression.symtable
        )

    def _get_fit_sources(self, header):
        """
        Return all PRF fit variables for the frame as structured numpy array.
        Args:
            frame_fname:    The filename of the frame to get PRF fitting
                sources of.
        Returns:
            numpy.array(dtype=[('ID', 'S#'),\
                               ('x', float), ('y', float),\
                               ('enabled', numpy.int8),\
                               ...]):
                A numpy field array with fields named with all variables to
                include in the PRF fit and containing the corresponding values,
                except enabled, which is left uninitialized.
        """

        print('Projecting sources')
        projected_sources = self._project_sources(header)

        print('Grouping')
        if callable(self._grouping):
            grouping, in_frame = self._grouping(
                projected_sources,
                self._evaluate_variable_expression.symtable,
                (header['NAXIS2'], header['NAXIS1'])
            )
        else:
            grouping = self._grouping
            in_frame = numpy.logical_and(
                numpy.logical_and(
                    projected_sources['x'] > 0,
                    projected_sources['x'] < header['NAXIS1'],
                ),
                numpy.logical_and(
                    projected_sources['y'] > 0,
                    projected_sources['y'] < header['NAXIS2'],
                )
            )

        print('Creating final source dataset')
        #pylint false positive.
        #pylint: disable=no-member
        fit_sources = numpy.empty(
            projected_sources.size,
            dtype=(
                [
                    ('ID', 'S' + str(self._id_length)),
                    ('x', numpy.float64),
                    ('y', numpy.float64),
                    ('enabled', numpy.float64)
                ]
                +
                [(fit_var[0], numpy.float64) for fit_var in self.fit_variables]
            )
        )
        #pylint: enable=no-member
        fit_sources['x'] = projected_sources['x']
        fit_sources['y'] = projected_sources['y']
        fit_sources['ID'] = self._evaluate_variable_expression.symtable['ID']

        print('Evaluating variable expressions.')
        for var_name, var_expression in self.fit_variables:
            fit_sources[var_name] = self._evaluate_variable_expression(
                var_expression
            )

        return fit_sources[in_frame], grouping[in_frame]

    def __init__(self,
                 *,
                 trans_fname_pattern,
                 catalogue_fname,
                 fit_variables,
                 fit_fname_pattern,
                 grouping,
                 grouping_frame=None,
                 discard_faint=False,
                 remove_group_id=None):
        """
        Set up to evalute the given fit variable expressions.
        Args:
            trans_fname_pattern:    See --trans-fname-pattern command
                line argument.
            catalogue_fname:    See --catalogue-fname command line argument.
            fit_variables:    See --add-fit-variable command line argument.
            fit_fname_pattern:    See --output-fname-pattern command
                line argument.
            grouping:    A splitting of the input sources in groups, each of
                which is enabled separately during PRF fitting. Should be a
                callable taking the input projected sources and catalogue
                information and returning A numpy integer array indicating for
                each source the PRF fitting group it is in. Sources assigned to
                negative group IDs are never enabled.
            grouping_frame:    If None, grouping is derived for each input
                frames separately, potentially resulting in a different set of
                sources enabled for each frame. If not None, it should specify
                the filename of a FITS frame on which to derive the grouping
                once and all subsequent fits are based on the same exact
                sources, regardless of where on the frame they appear, as long
                as they are within the frame.
            discard_faint:    See `--discard-faint` command line argument.
            remove_group_id:    See '--remove-group-id' command line argument.
        Returns:
            None
        """

        catalogue_sources = parse_catalogue(catalogue_fname)
        if discard_faint is not None:
            discard_filter, faint_limit = discard_faint.split('>')
            faint_limit = float(faint_limit)
            catalogue_sources = catalogue_sources[
                catalogue_sources[discard_filter] <= faint_limit
            ]

        self._evaluate_variable_expression = asteval.Interpreter()
        for catalogue_column in catalogue_sources.dtype.names:
            self._evaluate_variable_expression.symtable[catalogue_column] = (
                catalogue_sources[catalogue_column]
            )

        self.fit_variables = fit_variables
        self.trans_fname_pattern = trans_fname_pattern
        self.catalogue_columns = catalogue_sources.dtype.names

        self._id_length = max(
            len(id_value) for id_value in catalogue_sources['ID']
        )

        self.fit_fname_pattern = fit_fname_pattern
        self.remove_group_id = remove_group_id

        if grouping_frame:
            header = get_header(grouping_frame)
            projected_sources = self._project_sources(header)
            self._grouping = grouping(
                projected_sources,
                self._evaluate_variable_expression.symtable,
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
        header = get_header(frame_fname)

        fit_sources, grouping = self._get_fit_sources(header)

        number_fit_groups = grouping.max() + 1

        if self.remove_group_id is not None:
            number_fit_groups = sorted(range(number_fit_groups))
            for remove_group_id in self.remove_group_id:
                print('Removing group_id: ' + repr(remove_group_id))
                del number_fit_groups[remove_group_id]
            result = [numpy.copy(fit_sources)
                      for group_id in number_fit_groups]
            for group_id in number_fit_groups:
                print(group_id)
                result[group_id]['enabled'] = (grouping == group_id)
            print(result)
        else:
            result = [numpy.copy(fit_sources)
                      for group_id in range(number_fit_groups)]
            for group_id in range(number_fit_groups):
                print(group_id)
                result[group_id]['enabled'] = (grouping == group_id)

            print(result)
        return result
#pylint: enable=too-few-public-methods
#pylint: enable=too-many-instance-attributes




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


    def get_dr_fname(frame_fname, fit_group):
        """Return the filename to saving a shape fit for single frame/group."""

        header = get_header(frame_fname)
        #False positive
        #pylint: disable=unsupported-assignment-operation
        header['FITGROUP'] = fit_group
        #pylint: enable=unsupported-assignment-operation

        return configuration['data_reduction_fname'].format(header)


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
        print('Fitting: '
              +
              '\tframe_filenames: ' + repr(frame_filenames)
              +
              '\tsources: ' + repr([sources[fit_group] for sources in
                                    fit_sources])
              +
              '\tdr_fnames: ' + repr([get_dr_fname(f, fit_group) for f in
                                      frame_filenames]))
        star_shape_fitter.fit(
            fits_fnames=frame_filenames,
            sources=[sources[fit_group] for sources in fit_sources],
            output_dr_fnames=[get_dr_fname(f, fit_group)
                              for f in frame_filenames],
            **shape_fitter_config
        )
        print('Done fitting')



def fit_star_shapes(image_collection, configuration):
    """Find the best-fit model for the PSF/PRF in the given images."""

    frame_list_splits = range(0,
                              len(configuration['frame_list']),
                              configuration['num_simultaneous'])
    fit_arguments = [
        (
            configuration['frame_list'][
                split
                :
                min(split + configuration['num_simultaneous'],
                    len(configuration['frame_list']))
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
    cmdline_config = vars(parse_command_line())
    del cmdline_config['config_file']
    fit_star_shapes(
        find_fits_fnames(cmdline_config.pop('calibrate_images')),
        find_fits_fnames(cmdline_config.pop('shape_fit_only_if')),
        cmdline_config
    )
