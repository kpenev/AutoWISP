"""Uniform interface for source extractoin using several methods."""

import numpy
from astropy.io import fits

from superphot_pipeline.file_utilities import\
    prepare_file_output,\
    get_unpacked_fits
from superphot_pipeline import source_finder_util

#This still makes sense as a class
#pylint: disable=too-few-public-methods
class SourceFinder:
    """Find sources in an image of the night sky and repor properties."""

    def __init__(self,
                 *,
                 tool='hatphot',
                 threshold=10,
                 allow_overwrite=False,
                 allow_dir_creation=False):
        """Prepare to use the specified tool and define faint limit."""

        self.configuration = dict(
            tool=tool,
            threshold=threshold,
            allow_overwrite=allow_overwrite,
            allow_dir_creation=allow_dir_creation
        )

    def __call__(self, fits_fname, source_fname=None, **configuration):
        """
        Extract the sources from the given frame and save or return them.

        Args:
            fits_fname(str):    The filename of the fits file to extract
                sources from. Can be packed or unpacked.

            source_fname(str or None):    If None, the extract sources are
                returned as numpy. field array. Otherwise, this specifies a file
                to save the source extractor output.

        Returns:
            field array or None:
                The extracted source from the image, if source_fname was None.
        """

        configuration = {**self.configuration, **configuration}
        if configuration['tool'] == 'mock':
            with fits.open(fits_fname) as fits_file:
                #False positive
                #pylint: disable=no-member
                hdu_index = 0 if fits_file[0].header['NAXIS'] else 1
                xresolution = fits_file[hdu_index].header['NAXIS1']
                yresolution = fits_file[hdu_index].header['NAXIS2']
                #pylint: disable=no-member
                med_pixel = numpy.median(fits_file[hdu_index].data)
            nsources = 1000
            result = numpy.empty(
                nsources,
                dtype=[
                    (name, (numpy.int32 if name in ['id', 'npix']
                            else numpy.float64))
                    for name in
                    source_finder_util.get_srcextract_columns('fistar')
                ]
            )

            result['id'] = numpy.arange(nsources)
            #False positive
            #pylint: disable=no-member
            result['x'] = numpy.random.random(nsources) * xresolution
            result['y'] = numpy.random.random(nsources) * yresolution

            result['bg'] = (
                (1.0 + 0.1 * numpy.random.random(nsources))
                *
                med_pixel
            )
            result['flux'] = (numpy.random.random(nsources)
                              *
                              configuration['threshold'])
            result['amp'] = 0.2 * result['flux']
            result['ston'] = result['flux'] / result['bg']

            result['s'] = 2.3 + 0.2 * numpy.random.random(nsources)
            result['d'] = 0.3 + 0.1 * numpy.random.random(nsources)
            result['k'] = 0.3 + 0.1 * numpy.random.random(nsources)


            result['npix'] = 20 + (5
                                   *
                                   numpy.random.random(nsources)).astype(int)
            #pylint: enable=no-member
            return result

        start_extraction = getattr(source_finder_util,
                                   'start_' + configuration['tool'])
        with get_unpacked_fits(fits_fname) as unpacked_fname:
            extraction_args = (unpacked_fname, configuration['threshold'])
            if source_fname:
                prepare_file_output(source_fname,
                                    configuration['allow_overwrite'],
                                    configuration['allow_dir_creation'])
                with open(source_fname, 'wb') as destination:
                    start_extraction(*extraction_args, destination).wait()
                return None

            extraction_process = start_extraction(*extraction_args)
            result = numpy.genfromtxt(
                extraction_process.stdout,
                names=source_finder_util.get_srcextract_columns(
                    configuration['tool']
                ),
                dtype=None
            )
            start_extraction.communicate()
            return result
#pylint: enable=too-few-public-methods
