#!/usr/bin/env python3

"""Calibrate image(s)."""

import re

from configargparse import Action

from superphot_pipeline.image_utilities import get_fits_fnames
from superphot_pipeline.image_calibration import Calibrator, overscan_methods
from superphot_pipeline.processing_steps.manual_util import get_cmdline_parser

def parse_area_str(area_str):
    """Parse a string formatted as <xmin>,<xmax>,<ymin>,<ymax> to dict."""

    return dict(
        zip(
            ['xmin', 'xmax', 'ymin', 'ymax'],
            [int(s) for s in area_str.split(',')]
        )
    )


#Interface defined by argparse
#pylint: disable=too-few-public-methods
class ParseChannelsAction(Action):
    """Parse chanel command line arguments as described in the help."""

    def __call__(self, parser, namespace, values, option_string=None):
        """Parse all channels into a dictionary as required by Calibrator."""

        result = dict()
        rex = re.compile(r'(?P<name>\w+)\('
                         r'(?P<x_start>\d+)(,(?P<x_step>\d+))?;'
                         r'(?P<y_start>\d+)(,(?P<y_step>\d+))?\)')
        channel_str = re.sub(r'\s+', '', ''.join(values))
        while channel_str:
            parsed = rex.match(channel_str)
            if not parsed or parsed.span()[0] != 0:
                raise ValueError('Malformatted channel specification: '
                                 +
                                 repr(channel_str))
            channel_str = channel_str[parsed.span()[1]:]

            parsed = parsed.groupdict()
            for direction in 'xy':
                if parsed[direction + '_step'] is None:
                    parsed[direction + '_step'] = 2
            result[parsed['name']] = tuple(
                slice(
                    int(parsed[direction + '_start']),
                    None,
                    int(parsed[direction + '_step'])
                )
                for direction in 'xy'
            )
        setattr(namespace, self.dest, result)


class ParseOverscanAction(Action):
    """Parse overscan command line argument per the help."""

    def __call__(self, parser, namespace, values, option_string=None):
        """Parse the --overscans argument to dict as required by Calibrator."""

        try:
            method, areas_str = values[0].split(':')
            setattr(
                namespace,
                self.dest,
                dict(
                    method=getattr(overscan_methods, method.title()),
                    areas=[parse_area_str(area)
                           for area in areas_str.split(';')]
                )
            )
        except Exception as orig_exception:
            raise ValueError(
                'Malformatted overscan specification: '
                +
                repr(values)
            ) from orig_exception
#pylint: enable=too-few-public-methods


def parse_command_line():
    """Return the parsed command line arguments."""

    parser = get_cmdline_parser(__doc__)
    parser.add_argument(
        'raw_images',
        nargs='+',
        help='A combination of individual images and image directories to '
        'process. Directories are not searched recursively.'
    )
    parser.add_argument(
        '--calibrate-only-if',
        default='True',
        help='Expression involving the header of the input images that '
        'evaluates to True/False if a particular image from the specified '
        'image collection should/should not be processed.'
    )
    parser.add_argument(
        '--raw-hdu',
        type=int,
        default=None,
        help='Which hdu of the raw frames contains the image to calibrate. If '
        'not specified, the first HDU with non-zero NAXIS value is used.'
    )
    parser.add_argument(
        '--split-channels',
        default=False,
        nargs='+',
        action=ParseChannelsAction,
        help='Allows processing images from color detectors which have pixels '
        'sensitive to different colors staggered on the image. Each entry '
        'specifies a sub-set of pixels that together form an image of just one '
        'color. If at least one channel is specified, any pixels not assigned '
        'to a channel are ignored. The format of each entry is: '
        '<channel name>(<x start>[,<x step>=2]];<y start>[,<y step>=2])'
        '[<channel name>(<x start>[,<x step>=2]];<y start>[,<y step>=2]).'
    )
    parser.add_argument(
        '--saturation-threshold',
        type=float,
        default=None,
        help='A value above which a pixel is considered saturated. If '
        r'uspecified, the value is set to 80%% of the highest pixel value in '
        'EACH image.'
    )
    parser.add_argument(
        '--overscans',
        action=ParseOverscanAction,
        default=dict(areas=None, method=None),
        help='Overscan correction to apply. The format is: '
        '<method>:<xmin1>,<xmax1>,<ymin1>,<ymax1>'
        '[;<xmin2>,<xmax2>,<ymin2>,<ymax2>;...].'
    )
    parser.add_argument(
        '--image-area',
        default=None,
        type=parse_area_str,
        help='The area of the image to process (the rest is discarded). The '
        'format is <xmin>,<xmax>,<ymin>,<ymax>'
    )
    parser.add_argument(
        '--gain',
        default=None,
        type=float,
        help='The gain to assume for the input image (electrons/ADU). If not '
        'specified, it must be defined in the header as GAIN keyword.'
    )
    parser.add_argument(
        '--compress-calibrated',
        default=None,
        type=int,
        help='Specify a quantization level for compressing the calibrated '
        'image.'
    )
    for master in ['bias', 'dark', 'flat']:
        parser.add_argument(
            '--master-' + master,
            default=None,
            help='The master ' + master + ' to apply. No ' + master +
            ' correction is applied of not specified.'
        )
    parser.add_argument(
        '--master-mask',
        default=None,
        nargs='+',
        help='Mask(s) to apply, indicating pixel quality. All pixels are '
        'considered "good" if no mask is specified.'
    )
    parser.add_argument(
        '--calibrated-fname',
        default='CAL/{RAWFNAME}.fits.fz',
        help='Format string to generate the filenames for saving the '
        'calibrated images. Replacement fields can be anything from the header '
        'of the generated image (including {CLRCHNL} - name of channel if '
        'channel splitting is done, and {RAWFNAME} - the name of the '
        'corresponding input FITS file without directories or `.fits` and '
        '`.fits.fz` extension).'
    )
    return parser.parse_args()


def calibrate(image_collection, configuration):
    """Calibrate the images from the specified collection."""

    calibrate_image = Calibrator(**configuration)
    for image_fname in image_collection:
        calibrate_image(image_fname)


if __name__ == '__main__':
    cmdline_config = vars(parse_command_line())
    del cmdline_config['config_file']
    calibrate(
        get_fits_fnames(
            cmdline_config.pop('raw_images'),
            cmdline_config.pop('calibrate_only_if')
        ),
        cmdline_config
    )
