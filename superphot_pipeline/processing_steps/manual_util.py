"""Collection of functions used by many processing steps."""

import logging

import numpy
from astropy.io import fits

from configargparse import ArgumentParser, DefaultsFormatter

class ManualStepArgumentParser(ArgumentParser):
    """Incorporate boiler plate handling of command line arguments."""

    def _add_version_args(self, components):
        """Add arguments to select versions of the given components."""

        version_arg_help = {
            'srcextract': 'The version of the extracted sources to use/create.',
            'catalogue': (
                'The version of the input catalogue of sources in the DR file '
                'to use/create.'
            ),
            'skytoframe': (
                'The vesrion of the astrometry solution in the DR file.'
            ),
            'srcproj': (
                'The version of the datasets containing projected photometry '
                'sources to use/create.'
            ),
            'background': (
                'The version identifier of background measurements to '
                'use/create.'
            ),
            'shapefit': (
                'The version identifier of PSF/PRF map fit to use/create.'
            ),
            'apphot': (
                'The version identifier of aperture photometry to use/create.'
            ),
            'magfit': (
                'The version of magnitude fitting to use/create.'
            )
        }
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
                 processing_step,
                 add_component_versions=(),
                 inputs_help_extra='',
                 allow_parallel_processing=False,
                 convert_to_dict=True,
                 add_lc_fname_arg=False):
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

        self.argument_descriptions = {}
        self.argument_defaults = {}

        self._convert_to_dict = convert_to_dict
        super().__init__(description=description,
                         default_config_files=[],
                         formatter_class=DefaultsFormatter,
                         ignore_unknown_config_file_keys=True)
        self.add_argument(
            '--config-file', '-c',
            is_config_file=True,
            # default=config_file,
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
                    #Would not work with calculated arngument
                    #pylint: disable=consider-using-f-string
                    (
                        'A combination of individual {0}s and {0} directories '
                        'to process. Directories are not searched recursively.'
                    ).format(input_name[:-1].replace('_', ' '))
                    #pylint: enable=consider-using-f-string
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
        if add_lc_fname_arg:
            self.add_argument(
                '--lc-fname',
                default='LC/{0!d}-{1:03d}-{2:07d}.h5',
                help='The light curve dumping filename pattern to use.'
            )

        self.add_argument(
            '--std-out-err-fname',
            default=(processing_step + '_{task:s}_{now:s}_pid{pid:d}.outerr'),
            help='The filename pattern to redirect stdout and stderr during'
            'multiprocessing. Should include substitutions to distinguish '
            'output from different multiprocessing processes. May include '
            'substitutions for any configuration arguments for a given '
            'processing step.'
        )
        self.add_argument(
            '--fname-datetime-format',
            default='%Y%m%d%H%M%S',
            help='How to format date and time as part of filenames (e.g. when '
            'creating output files for multiprocessing.'
        )
        self.add_argument(
            '--logging-fname',
            default=(processing_step + '_{task:s}_{now:s}_pid{pid:d}.log'),
            help='The filename pattern to use for log files. Should include'
            ' substitutions to distinguish logs from different '
            'multiprocessing processes. May include substitutions for any '
            'configuration arguments for a given processing step.'
        )
        self.add_argument(
            '--verbose',
            default='info',
            choices=['debug', 'info', 'warning', 'error', 'critical'],
            help='The type of verbosity of logger.'
        )
        self.add_argument(
            '--logging-message-format',
            default=('%(levelname)s %(asctime)s %(name)s: %(message)s | '
                     '%(pathname)s.%(funcName)s:%(lineno)d'),
            help='The format string to use for log messages. See python logging'
            ' module for details.'
        )
        self.add_argument(
            '--logging-datetime-format',
            default=None,
            help='How to format date and time as part of filenames (e.g. when '
            'creating output files for multiprocessing.'
        )


    def add_argument(self, *args, **kwargs):
        """Store each argument's description in self.argument_descriptions."""

        argument_name = args[0].lstrip('-').replace('-', '_')
        if kwargs.get('action', None) == 'store_false':
            self.argument_descriptions[kwargs['dest']] = {
                'rename': argument_name,
                'help': kwargs['help']
            }
        else:
            self.argument_descriptions[argument_name] = kwargs['help']

        if 'default' in kwargs:
            nargs = kwargs.get('nargs', 1)
            if isinstance(kwargs['default'], str) or kwargs['default'] is None:
                self.argument_defaults[argument_name] = kwargs['default']
            else:
                if kwargs.get('action', None) == 'store_true':
                    assert kwargs.get('default', False) is False
                    self.argument_defaults[argument_name] = 'False'
                elif kwargs.get('action', None) == 'store_false':
                    assert kwargs.get('default', True) is True
                    self.argument_defaults[argument_name] = repr(
                        kwargs['dest'] == argument_name
                    )
                else:
                    self.argument_defaults[argument_name] = repr(
                        kwargs['default']
                    )
                if (
                        'type' not in kwargs
                        and
                        kwargs.get('action', None) not in ['store_true',
                                                           'store_false']
                        and
                        nargs == 1
                ):
                    raise ValueError(
                        f'Non-string default value ({kwargs["default"]}) and '
                        f'no type specified for {argument_name}.'
                    )
                if kwargs.get('action', None) not in ['store_true',
                                                      'store_false']:
                    if nargs == '+' or nargs > 1:
                        self.argument_defaults[argument_name] = repr(
                            list(kwargs['default'])
                        )
                    elif (
                        kwargs['type'](self.argument_defaults[argument_name])
                        !=
                        kwargs['default']
                    ):
                        raise ValueError(
                            'Could not convert default value of '
                            f'{argument_name} for DB: {kwargs["default"]}'
                        )

        return super().add_argument(*args, **kwargs)


    #pylint: disable=signature-differs
    def parse_args(self, *args, **kwargs):
        """Set-up logging and return cleaned up dict instead of namespace."""

        result = super().parse_args(*args, **kwargs)
        if self._convert_to_dict:
            result = vars(result)
            del result['config_file']
            del result['extra_config_file']
            logging.basicConfig(
                level=getattr(logging, result['verbose'].upper()),
                format='%(levelname)s %(asctime)s %(name)s: %(message)s | '
                       '%(pathname)s.%(funcName)s:%(lineno)d'
            )
        else:
            logging.basicConfig(
                level=getattr(logging, result.verbose.upper()),
                format='%(levelname)s %(asctime)s %(name)s: %(message)s | '
                       '%(pathname)s.%(funcName)s:%(lineno)d'
            )
            del result.config_file
            del result.extra_config_file
            del result.verbose

        if args or kwargs:
            result['argument_descriptions'] = self.argument_descriptions
            result['argument_defaults'] = self.argument_defaults

        return result
    #pylint: enable=signature-differs


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


def read_subpixmap(fits_fname):
    """Read the sub-pixel sensitivity map from a FITS file."""

    if fits_fname is None:
        return numpy.ones((1, 1), dtype=float)
    with fits.open(fits_fname, 'readonly') as subpixmap_file:
        #False positive, pylint does not see data member.
        #pylint: disable=no-member
        return numpy.copy(subpixmap_file[0].data).astype('float64')
        #pylint: enable=no-member
