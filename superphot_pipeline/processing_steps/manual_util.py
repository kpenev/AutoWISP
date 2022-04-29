"""Collection of functions used by many processing steps."""

from configargparse import ArgumentParser, DefaultsFormatter

def get_cmdline_parser(description):
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
    return parser
