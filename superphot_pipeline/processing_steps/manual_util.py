"""Collection of functions used by many processing steps."""

import logging
import re

import numpy
import pandas
from astropy.io import fits
from asteval import Interpreter

from configargparse import ArgumentParser, DefaultsFormatter

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
                        'A combination of individual {0}s and {0} directories '
                        'to process. Directories are not searched recursively.'
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


class DetrendDatasetIter:
    """Iterate over the detrending datasets specified in a cmdline argument."""

    def __init__(self, argument):
        """Set up the iterator for the given argument."""

        self._splits = argument.split(';')


    def __iter__(self):
        return self


    def __next__(self):
        """Return the next dataset specification."""

        if len(self._splits) == 0:
            raise StopIteration

        result = self._splits.pop(0)
        while len(self._splits) > 0 and self._splits[0] == '':
            self._splits.pop(0)
            result += ';'
            if len(self._splits) > 0:
                result += self._splits.pop(0)


class LCDetrendingArgumentParser(ManualStepArgumentParser):
    """Boiler plate handling of LC detrending command line arguments."""

    @staticmethod
    def _split_delimited_string(argument, separator):
        """Split the detrending dataset argument at `;`, handling `;;`."""

        splits = argument.split(separator)
        while splits:
            result = splits.pop(0)
            while splits and splits[0] == '':
                splits.pop(0)
                result += separator
                if splits:
                    result += splits.pop(0)
            yield result.strip()

    @staticmethod
    def _parse_detrend_datasets(argument):
        """Parse the detrend datasets argument (see help for details)."""

        dset_specfication_rex = re.compile(
            r'^(?P<from>[\w.]+)'
            r'\s*->\s*'
            r'(?P<to>[\w.]+)'
            r'\s*'
            r'(:\s*(?P<substitutions>.*))?'
            r'$'
        )
        for specification in (
                LCDetrendingArgumentParser._split_delimited_string(argument,
                                                                   ';')
        ):
            match = dset_specfication_rex.match(specification)
            print('Detrend dataset specification: ' + repr(specification))
            for component in ['from', 'to', 'substitutions']:
                print('\t{!s}: {!r}'.format(component, match[component]))
                if component == 'substitutions':
                    for substitution in (
                            LCDetrendingArgumentParser._split_delimited_string(
                                match['substitutions'],
                                '&'
                            )
                    ):
                        print('\t\t' + substitution)
        return None

        aeval = Interpreter()
        if 'for' in argument:
            tuples, loop = argument.split(' ', 1)
            dataset, dict_substitution, key = tuples.split(':')
            dataset = dataset.strip('(')
            key = key.strip(')')
            loop_pieces = loop.split()
            assert loop_pieces[0] == 'for'
            assert loop_pieces[2] == 'in'
            if ';' in dict_substitution:
                dict_substitution = dict_substitution.replace(';', ', ')
                if loop_pieces[1] in dict_substitution:
                    dict_substitution = [
                        aeval(
                            dict_substitution.replace(
                                loop_pieces[1],
                                str(x)
                            )
                        ) for x in aeval(loop_pieces[3])
                    ]
                else:
                    print('Error:'
                          +
                          loop_pieces[1]
                          +
                          ' not found within dictionary of path substitutions')
            else:
                dict_substitution = aeval(dict_substitution)
                if loop_pieces[1] in dict_substitution:
                    dict_substitution = [
                        aeval(
                            dict_substitution.replace(
                                loop_pieces[1],
                                str(x)
                            )
                        ) for x in aeval(loop_pieces[3])
                    ]
                else:
                    print('Error:'
                          +
                          loop_pieces[1]
                          +
                          ' not found within dictionary of path substitutions')
            return [(dataset, i, key) for i in dict_substitution]

        dataset, dict_substitution, key = argument.split(':')
        dataset = dataset.strip('(')
        key = key.strip(')')
        if ';' in dict_substitution:
            dict_substitution = aeval(dict_substitution.replace(';', ', '))
        else:
            dict_substitution = aeval(dict_substitution)
        return [(dataset, dict_substitution, key)]



    @staticmethod
    def _add_transit_parameters(parser,
                                *,
                                timing=True,
                                duration=True,
                                geometry='',
                                limb_darkening=False,
                                fit_flags=False):
        """
        Add command line parameters to the current parser to specify a transit.

        Args:
            timing(bool):    Should arguments specifying the timing of the
                transit be included, i.e. period, ephemerides, duration?

            duration(bool):    Sholud a potentially redundant argument for
                transit duration be added?

            geometry(bool or str):     Should arguments specifying the geometry
                of the transit be included. If the value converts to False,
                nothing is included. Otherwise this argument should be either:

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
            parser.add_argument(
                '--mid-transit',
                type=float,
                default=2455867.402743,
                help='The time of mit-transit in BJD. Default: %(default)s '
                '(from https://ui.adsabs.harvard.edu/abs/2019AJ....157...82W).'
            )
            parser.add_argument(
                '--period',
                type=float,
                default=2.15000820,
                help='The orbital period (used to convert time to phase). '
                'Default: %(default)s (from '
                'https://ui.adsabs.harvard.edu/abs/2019AJ....157...82W).'
            )
        if duration:
            parser.add_argument(
                '--transit-duration',
                type=float,
                default=3.1205,
                help='The transit duration in hours. Default: %(default)s (from'
                ' https://ui.adsabs.harvard.edu/abs/2019AJ....157...82W).'
            )

        if geometry:
            parser.add_argument(
                '--radius-ratio',
                type=float,
                default=0.14886,
                help='The planet to star radius ratio. Default: %(default)s '
                '(HAT-P-32).'
            )
            parser.add_argument(
                '--inclination',
                type=float,
                default=88.98,
                help='The inclination angle between the orbital angular '
                'momentum and the line of sight in degrees. '
                'Default: %(default)s (HAT-P-32).'
            )
            parser.add_argument(
                '--scaled-semimajor',
                type=float,
                default=5.344,
                help='The ratio of the semimajor axis to the stellar radius. '
                'Default: %(default)s (HAT-P-32).'
            )
            if geometry == 'eccentric':
                parser.add_argument(
                    '--eccentricity', '-e',
                    type=float,
                    default=0.0,
                    help='The orbital eccentricity to assume for the transiting'
                    ' planet. Default: %(default)s.'
                )
                parser.add_argument(
                    '--periastron',
                    type=float,
                    default=0.0,
                    help='The argument of periastron. Default: %(default)s.'
                )

        if limb_darkening:
            parser.add_argument(
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
            parser.add_argument(
                '--mutable-transit-params',
                nargs='+',
                choices=choices,
                default=[],
                help='List the transit parameters for which best-fit values '
                'should be found rather than assuming they are fixed at what is'
                ' specified on the command line. Default: %(default)s.'
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
            help='An expression using epd_variables` which evaluates to either'
            ' True or False indicating if a given point in the lightcurve '
            'should be fit and corrected. Default: %(default)s'
        )
        self.add_argument(
            '--detrend-datasets',
            type=self._parse_detrend_datasets,
            help='A `;` separated list of the datasets to detrend. Each entry '
            'should be formatted as: `<input-key> -> <output-key> '
            '[: <substitution> (= <value>| in <expression>) '
            '[& <substitution> (= <value> | in <expression>) ...]]. '
            'For example: `"apphot.magfit.magnitude -> '
            'apphot.epd.magnitude : magfit_iteration = 5 & aperture_index in '
            'range(39)"`. White space is ignored. Literal `;` and `&` can be '
            'used in the specification of a dataset as `;;` and `&&` '
            'respectively. Configurations of how the fitting was done and the '
            'resulting residual and non-rejected points are added to '
            'configuration datasets generated by removing the tail of the '
            'destination and adding `".cfg." + <parameter name>` for '
            'configurations and just ` + <parameter name>` for '
            'fitting statistics. For example, if the output dataset key is'
            '`"shapefit.epd.magnitude"`, the configuration datasets will look'
            'like `"shapefit.epd.cfg.fit_terms"`, and'
            '`"shapefit.epd.residual"`.'
        )
        self.add_argument(
            '--detrend-error-avg',
            default='nanmedian',
            help='How to average fitting residuals for outlier rejection. '
            'Default: %(default)s'
        )
        self.add_argument(
            '--detrend-rej-level',
            type=float,
            default=5.0,
            help='How far away from the fit should a point be before '
            'it is rejected in utins of detrend_error_avg. Default: %(default)s'
        )
        self.add_argument(
            '--detrend-max-rej-iter',
            type=int,
            default=20,
            help='The maximum number of rejection/re-fitting iterations to '
            'perform. If the fit has not converged by then, the latest '
            'iteration is accepted. Default: %(default)s'
        )
        if add_reconstructive:
            target_args = self.add_argument_group(
                title='Followup Target',
                description='Arguments specific to processing followup '
                'observations where the target star is known to have a transit '
                'that occupies a significant fraction of the total collection '
                'of observations.'
            )
            target_args.add_argument(
                '--target-id',
                default=None,
                help='The lightcurve of the given source (any one of the '
                'catalogue identifiers stored in the LC file) will be fit using'
                ' reconstructive detrending, starting the transit parameter fit'
                ' with the values supplied, and fitting for the values allowed '
                'to vary. If not specified all LCs are fit in '
                'non-reconstructive way.'
            )
            self._add_transit_parameters(target_args,
                                        timing=True,
                                        geometry='circular',
                                        limb_darkening=True,
                                        fit_flags=True)

        self.add_argument(
            '--detrending-catalogue', '--detrending-catalog', '--cat',
            default=None,
            help='The name of a catalogue file containing the sources in the '
            'frame. Used only to generate performance statistics reports. If '
            'not specified, performance is not reported.'
        )
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
