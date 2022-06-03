"""Collection of functions used by many processing steps."""

import logging

import numpy
import pandas
from astropy.io import fits

from configargparse import ArgumentParser, DefaultsFormatter

from superphot_pipeline.file_utilities import find_lc_fnames

class ManualStepArgumentParser(ArgumentParser):
    """Incorporate boiler plate handling of command line arguments."""

    def _add_version_args(self, components):
        """Add arguments to select versions of the given components."""

        version_arg_help = dict(
            srcextract=(
                'The version of the extracted sources to use/create.'
            ),
            catalogue=(
                'The version of the input catalogue of sources in the DR file '
                'to use/create.'
            ),
            skytoframe=(
                'The vesrion of the astrometry solution in the DR file.'
            ),
            srcproj=(
                'The version of the datasets containing projected photometry '
                'sources to use/create.'
            ),
            background=(
                'The version identifier of background measurements to '
                'use/create.'
            ),
            shapefit=(
                'The version identifier of PSF/PRF map fit to use/create.'
            ),
            apphot=(
                'The version identifier of aperture photometry to use/create.'
            ),
            magfit=(
                'The version of magnitude fitting to use/create.'
            )
        )
        for comp in components:
            self.add_argument(
                '--' + comp + '-version',
                type=int,
                default=0,
                help=version_arg_help[comp]
            )


    def __init__(self,
                 *,
                 input_type,
                 description,
                 add_component_versions=(),
                 inputs_help_extra='',
                 allow_parallel_processing=False,
                 convert_to_dict=True):
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

        self._convert_to_dict = convert_to_dict
        super().__init__(description=description,
                         default_config_files=[],
                         formatter_class=DefaultsFormatter,
                         ignore_unknown_config_file_keys=True)
        self.add_argument(
            '--config-file', '-c',
            is_config_file=True,
            help='Specify a configuration file in liu of using command line '
            'options. Any option can still be overriden on the command line.'
        )
        self.add_argument(
            '--extra-config-file',
            is_config_file=True,
            help='Hack around limitation of configargparse to allow for '
            'setting a second config file.'
        )

        if input_type == 'raw':
            input_name = 'raw_images'
        elif input_type.startswith('calibrated'):
            input_name = 'calibrated_images'
        elif input_type == 'dr':
            input_name = 'dr_files'
        elif input_type == 'lc':
            input_name = 'lc_files'
        else:
            input_name = None

        if input_name is not None:
            self.add_argument(
                input_name,
                nargs='+',
                help=(
                    (
                        'A combination of individual {0}s and {0} directories to '
                        'process. Directories are not searched recursively.'
                    ).format(input_name[:-1].replace('_', ' '))
                    +
                    inputs_help_extra
                )
            )
        if '+' in input_type and input_type.split('+')[1].strip() == 'dr':
            self.add_argument(
                '--data-reduction-fname',
                default='DR/{RAWFNAME}.h5',
                help='Format string to generate the filename(s) of the data '
                'reduction files where extracted sources are saved. Replacement'
                ' fields can be anything from the header of the calibrated '
                'image.'
            )
        if allow_parallel_processing:
            self.add_argument(
                '--num-parallel-processes',
                type=int,
                default=12,
                help='The number of simultaneous fitpsf/fitprf processes to '
                'run.'
            )

        self._add_version_args(add_component_versions)
        self.add_argument(
            '--verbose',
            default='info',
            choices=['debug', 'info', 'warning', 'error', 'critical'],
            help='The type of verbosity of logger.'
        )


    #pylint: disable=signature-differs
    def parse_args(self, *args, **kwargs):
        """Set-up logging and return cleaned up dict instead of namespace."""

        result = super().parse_args(*args, **kwargs)
        if self._convert_to_dict:
            result = vars(result)
            del result['config_file']
            del result['extra_config_file']
            logging.basicConfig(
                level=getattr(logging, result.pop('verbose').upper())
            )
        else:
            logging.basicConfig(
                level=getattr(logging, result.verbose.upper())
            )
            del result.config_file
            del result.extra_config_file
            del result.verbose

        return result
    #pylint: enable=signature-differs


class LCDetrendingArgumentParser(ManualStepArgumentParser):
    """Boiler plate handling of LC detrending command line arguments."""

    def _add_transit_parameters(self,
                                *,
                                timing=True,
                                duration=True,
                                geometry='',
                                limb_darkening=False,
                                fit_flags=False):
        """
        Add command line parameters to the current parser to specify a transit.

        Args:
            timing(bool):    Should arguments specifying the timing of the transit
                be included, i.e. period, ephemerides, duration?

            duration(bool):    Sholud a potentially redundant argument for transit
                duration be added?

            geometry(bool or str):     Should arguments specifying the geometry of
                the transit be included. If the value converts to False, nothing is
                included. Otherwise this argument should be either:

                    * 'circular': i.e. include radius ratio, inclination, and
                      semimajor axis.

                    * 'eccentric': i.e. in additition to the above, include
                      eccentricity and argument of periastron.

            limb_darkening(bool):    Should arguments for specifying the stellar
                limb-darkening be included?

        Returns:
            None
        """

        assert (not geometry) or (geometry in ['circular', 'eccentric'])

        if timing:
            self.add_argument(
                '--mid-transit',
                type=float,
                default=2455867.402743,
                help='The time of mit-transit in BJD. Default: %(default)s (from '
                'https://ui.adsabs.harvard.edu/abs/2019AJ....157...82W).'
            )
            self.add_argument(
                '--period',
                type=float,
                default=2.15000820,
                help='The orbital period (used to convert time to phase). Default: '
                '%(default)s (from '
                'https://ui.adsabs.harvard.edu/abs/2019AJ....157...82W).'
            )
        if duration:
            self.add_argument(
                '--transit-duration',
                type=float,
                default=3.1205,
                help='The transit duration in hours. Default: %(default)s (from '
                'https://ui.adsabs.harvard.edu/abs/2019AJ....157...82W).'
            )

        if geometry:
            self.add_argument(
                '--radius-ratio',
                type=float,
                default=0.14886,
                help='The planet to star radius ratio. Default: %(default)s '
                '(HAT-P-32).'
            )
            self.add_argument(
                '--inclination',
                type=float,
                default=88.98,
                help='The inclination angle between the orbital angular momentum '
                'and the line of sight in degrees. Default: %(default)s (HAT-P-32).'
            )
            self.add_argument(
                '--scaled-semimajor',
                type=float,
                default=5.344,
                help='The ratio of the semimajor axis to the stellar radius. '
                'Default: %(default)s (HAT-P-32).'
            )
            if geometry == 'eccentric':
                self.add_argument(
                    '--eccentricity', '-e',
                    type=float,
                    default=0.0,
                    help='The orbital eccentricity to assume for the transiting '
                    'planet. Default: %(default)s.'
                )
                self.add_argument(
                    '--periastron',
                    type=float,
                    default=0.0,
                    help='The argument of periastron. Default: %(default)s.'
                )

        if limb_darkening:
            self.add_argument(
                '--limb-darkening',
                nargs=2,
                type=float,
                default=(0.316, 0.303),
                help='The limb darkening coefficients to assume for the star '
                '(quadratic model). Default: %(default)s (HAT-P-32).'
            )

        if fit_flags:
            choices = []
            if timing:
                choices.extend(['mid_transit', 'period'])
            if geometry:
                choices.extend(['depth', 'inclination', 'semimajor'])
                if geometry == 'eccentric':
                    choices.extend(['eccentricity', 'periastron'])
            if limb_darkening:
                choices.extend(['limbdark'])
            self.add_argument(
                '--mutable-transit-params',
                nargs='+',
                choices=choices,
                default=[],
                help='List the transit parameters for which best-fit values should '
                'be found rather than assuming they are fixed at what is specified '
                'on the command line. Default: %(default)s.'
            )


    def __init__(self,
                 description,
                 add_reconstructive=True,
                 convert_to_dict=True):
        """
        Initialize the parser with options common to all LC detrending steps.

        Args:
            See ManualStepArgumentParser.__init__().

        Returns:
            None
        """

        super().__init__(
            input_type='lc',
            description=description,
            convert_to_dict=convert_to_dict
        )

        self.add_argument(
            '--fit-points-filter-expression',
            default=None,
            help='An expression using used_variables` which evaluates to either '
                 'True or False indicating if a given point in the lightcurve '
                 'should be fit and corrected.'
                 'Default: %(default)s'
        )
        self.add_argument(
            '--fit-datasets',
            type=parse_fit_datasets,
            action='append',
            help='A list of 3-tuples of pipeline keys'
            'corresponding to each variable identifying a dataset to fit and'
            'correct, an associated dictionary of path substitutions, and a'
            'pipeline key for the output dataset. Configurations of how the'
            'fitting was done and the resulting residual and non-rejected'
            'points are added to configuration datasets generated by removing'
            'the tail of the destination and adding `".cfg." + <parameter'
            'name>` for configurations and just ` + <parameter name>` for'
            'fitting statistics. For example, if the output dataset key is'
            '`"shapefit.epd.magnitude"`, the configuration datasets will look'
            'like `"shapefit.epd.cfg.fit_terms"`, and'
            '`"shapefit.epd.residual"`.'
            'Example: use tuple input '
            '(shapefit.magfit.magnitude:dict(magfit_iteration=5)'
            ':shapefit.epd.magnitude) using colon separators. If using a for loop '
            'indicate the variable to be looped such as "for ap_ind in range(39)", '
            'and include an empty space between the dictionary and the loop for '
            'example.'
        )
        self.add_argument(
            '--error-avg',
            default='nanmedian',
            help='How to average fitting residuals for outlier rejection. Default: '
            '%(default)s'
        )
        self.add_argument(
            '--rej-level',
            type=float,
            default=5.0,
            help='How far away from the fit should a point be before '
            'it is rejected in utins of error_avg. Default: %(default)s'
        )
        self.add_argument(
            '--max-rej-iter',
            type=int,
            default=20,
            help='The maximum number of rejection/re-fitting iterations to perform.'
            ' If the fit has not converged by then, the latest iteration is '
            'accepted. Default: %(default)s'
        )
    if add_reconstructive:
        target_args = self.add_argument_group(
            title='Followup Target',
            description='Arguments specific to processing followup observations '
            'where the target star is known to have a transit that occupies a '
            'significant fraction of the total collection of observations.'
        )
        target_args.add_argument(
            '--target-id',
            default=None,
            help='The lightcurve whose base filename starts with the given value '
            'will be fit using reconstructive detrending, starting the transit '
            'parameter fit with the values supplied, and fitting for the values '
            'allowed to vary. If not specified all LCs are fit in '
            'non-reconstructive way. Default: %(default)s (HAT-P-32).'
        )
        self._add_transit_parameters(target_args,
                                    timing=True,
                                    geometry='circular',
                                    limb_darkening=True,
                                    fit_flags=True)

    self.add_argument(
        '--epd-statistics-fname',
        default='epd_statistics.txt',
        help='The statistics filename for the results of the EPD fit. '
        'Default: %(default)s'
    )
    self.add_argument(
        '--magnitude-column',
        default='R',
        help='The magnitude column to use for the performance statistics. '
        'Default: %(default)s'
    )


def add_image_options(parser):
    """Add options specifying the properties of the image."""

    parser.add_argument(
        '--subpixmap',
        default=None,
        help='The sub-pixel sensitivity map to assume. If not specified '
        'uniform sensitivy is assumed.'
    )
    parser.add_argument(
        '--gain',
        type=float,
        default=1.0,
        help='The gain to assume for the input images.'
    )
    parser.add_argument(
        '--magnitude-1adu',
        type=float,
        default=10.0,
        help='The magnitude which corresponds to a source flux of 1ADU'
    )


def read_catalogue(catalogue_fname):
    """Return the catalogue parsed to pandas.DataFrame."""

    catalogue = pandas.read_csv(catalogue_fname,
                                sep = r'\s+',
                                header=0,
                                index_col=0)
    catalogue.columns = [colname.lstrip('#').split('[', 1)[0]
                         for colname in catalogue.columns]
    catalogue.index.name = (
        catalogue.index.name.lstrip('#').split('[', 1)[0]
    )
    return catalogue


def read_subpixmap(fits_fname):
    """Read the sub-pixel sensitivity map from a FITS file."""

    if fits_fname is None:
        return numpy.ones((1, 1), dtype=float)
    with fits.open(fits_fname, 'readonly') as subpixmap_file:
        #False positive, pylint does not see data member.
        #pylint: disable=no-member
        return numpy.copy(subpixmap_file[0].data).astype('float64')
        #pylint: enable=no-member
