"""Collection of functions used by many processing steps."""

import logging

import numpy
import pandas
from astropy.io import fits

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
                 allow_parallel_processing=False):
        """
        Initialize the praser with options common to all manual steps.

        Args:
            input_type(str):    What kind of files does the step process.
                Possible values are ``'raw'``, ``'calibrated'``, ``'dr'``,
                or ``'calibrated + dr'``.

            add_component_versions(str iterable):    A list of DR file version
                numbers the step needs to know. For example ``('srcextract',)``.

            description(str):    The description of the processing step to add
                to the help message.

            inputs_help_extra(str):    Additional text to append to the help
                string for the input files. Usually describing what requirements
                they must satisfy.

            allow_parallel_processing(bool):    Should an argument be added to
                specify the number of paralllel processes to use.

        Returns:
            None
        """

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

        if input_type == 'raw':
            input_name = 'raw_images'
        elif input_type.startswith('calibrated'):
            input_name = 'calibrated_images'
        elif input_type == 'dr':
            input_name = 'dr_files'
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

        result = vars(super().parse_args(*args, **kwargs))
        del result['config_file']
        logging.basicConfig(
            level=getattr(logging, result.pop('verbose').upper())
        )
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
