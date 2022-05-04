"""Collection of functions used by many processing steps."""

import pandas

from configargparse import ArgumentParser, DefaultsFormatter

def get_cmdline_parser(description, input_type, help_extra=''):
    """Return a command line parser with only a config file option defined."""

    parser = ArgumentParser(
        description=description,
        default_config_files=[],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=True
    )
    parser.add_argument(
        '--config-file', '-c',
        is_config_file=True,
        help='Specify a configuration file in liu of using command line '
        'options. Any option can still be overriden on the command line.'
    )

    if input_type == 'raw':
        input_name = 'raw_images'
    elif input_type == 'calibrated':
        input_name = 'calibrated_images'
    elif input_type == 'dr':
        input_name = 'dr_files'

    parser.add_argument(
        input_name,
        nargs='+',
        help=(
            (
                'A combination of individual {0}s and {0} directories to '
                'process. Directories are not searched recursively.'
            ).format(input_name[:-1].replace('_', ' '))
            +
            help_extra
        )
    )


    return parser


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
