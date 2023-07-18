"""Collection of functions used by many processing steps."""

import logging

import numpy
import pandas
from astropy.io import fits

from configargparse import ArgumentParser, DefaultsFormatter

from superphot_pipeline import Evaluator

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
        if add_lc_fname_arg:
            self.add_argument(
                '--lc-fname',
                default='LC/{0!d}-{1:03d}-{2:07d}.h5',
                help='The light curve dumping filename pattern to use.'
            )

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
        print('result: ' + repr(result))
        if self._convert_to_dict:
            result = vars(result)
            print('As dict: ' + repr(result))
            del result['config_file']
            del result['extra_config_file']
            logging.basicConfig(
                level=getattr(logging, result.pop('verbose').upper()),
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


def read_catalogue(catalogue_fname,
                   filter_expr=None,
                   sort_expr='V'):
    """
    Return the catalogue parsed to pandas.DataFrame.

    Args:
        catalogue_fname(str):    The filename of the catalogue to read.

        filter_expr(str):    The expression to evaluate for each source in the
            catalogue, keeping only those for which conversion to boolean is
            True.

        sort_expr(str):    The expression to evaluate for each source in the
            catalogue sorting by the result.

    Returns:
        pandas.DataFrame:
            The columns in the catalogue fistered and sourted as specified
    """

    catalogue = pandas.read_csv(catalogue_fname,
                                sep=r'\s+',
                                header=0,
                                index_col=0)
    catalogue.columns = [colname.lstrip('#').split('[', 1)[0]
                         for colname in catalogue.columns]
    catalogue.index.name = (
        catalogue.index.name.lstrip('#').split('[', 1)[0]
    )
    cat_eval = Evaluator(catalogue)
    sort_val = cat_eval(sort_expr)
    print('Sort val: ' + repr(sort_val))

    if filter_expr is not None:
        print('Filter expression: ' + repr(filter_expr))
        filter_val = cat_eval(filter_expr)
        print('Filter val: ' + repr(filter_val))
        filter_val = filter_val.astype(bool)
        catalogue = catalogue.loc[filter_val]
        sort_val = sort_val[filter_val]

    return catalogue.iloc[numpy.argsort(sort_val)]


def read_subpixmap(fits_fname):
    """Read the sub-pixel sensitivity map from a FITS file."""

    if fits_fname is None:
        return numpy.ones((1, 1), dtype=float)
    with fits.open(fits_fname, 'readonly') as subpixmap_file:
        #False positive, pylint does not see data member.
        #pylint: disable=no-member
        return numpy.copy(subpixmap_file[0].data).astype('float64')
        #pylint: enable=no-member


if __name__ == '__main__':
    catalogue = read_catalogue(
        '/Users/kpenev/tmp/PANOPTES/R_astrometry_catalogue.ucac4',
        filter_expr='R<8.0',
        sort_expr='R-V'
    )
    print(repr(catalogue[['R', 'V']]))
    print(repr(catalogue['R'] - catalogue['V']))
