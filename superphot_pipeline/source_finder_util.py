"""Some general purpose low level tools for source extraction."""

from subprocess import Popen, PIPE

def start_hatphot(unpacked_fits_fname, threshold, stdout=PIPE):
    """Find sources in the given frame using hatphot."""

    return Popen(
        [
            'hatphot',
            '--thresh', repr(threshold),
            '--sort', '0',
            unpacked_fits_fname
        ],
        stdout=stdout
    )

def start_fistar(unpacked_fits_fname, threshold, stdout=PIPE):
    """Find sources in the given frame using fistar."""

    return Popen(
        [
            'fistar', unpacked_fits_fname,
            '--sort', 'flux',
            '--format', ','.join(get_srcextract_columns('fistar')),
            '--algorithm', 'uplink',
            '--flux-threshold', repr(threshold)
        ],
        stdout=stdout
    )

def get_srcextract_columns(tool):
    """Return a list of the columns in the output source extraction file."""

    if tool == 'hatphot':
        return ('id', 'x', 'y', 'peak', 'flux', 'fwhm', 'npix', 'round', 'pa')
    if tool == 'fistar':
        return ('id',
                'x',
                'y',
                'bg',
                'amp',
                's',
                'd',
                'k',
                'flux',
                's/n',
                'npix')
    raise KeyError('Unrecognized sourc exatraction tool: ' + repr(tool))
