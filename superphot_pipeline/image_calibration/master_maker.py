"""Define classes for creating master calibration frames."""

import numpy

from astropy.io import fits

from superphot_pipeline.image_utilities import read_image_components
from superphot_pipeline.general_purpose_stats import iterative_rejection_average
from superphot_pipeline.image_calibration.mask_utilities import mask_flags
from superphot_pipeline.pipeline_exceptions import\
    ImageMismatchError,\
    BadImageError

#pylint does not count __call__ but should.
#pylint: disable=too-few-public-methods
class MasterMaker:
    """
    Implement the simplest & fully generalizable procedure for making a master.

    Attrs:
        stacking_options:    A dictionary with the default configuration of how
            to perform the stacking. The expected keys exactly match the keyword
            only arguments of the `stack_to_master` method.

    Examples:

        >>> #Create an object for stacking frames to masters, overwriting the
        >>> #default outlier threshold
        >>> make_master = MasterMaker(outlier_threshold=10.0)

        >>> #Stack a set of frames to a master, allowing no more than 3
        >>> #averaging/outlier rejection iterations and allowing a minimum of 3
        >>> #valid source pixels to make a master, for this master only.
        >>> make_master(['f1.fits', 'f2.fits', 'f3.fits', 'f4.fits'],
        >>>             'master.fits',
        >>>             max_iter=3,
        >>>             min_valid_values=3)
    """

    @staticmethod
    def _update_stack_header(master_header,
                             frame_header,
                             filename):
        """
        Update the master header per header from one of the individual frames.

        Should be called once for each frame participating in the stack with the
        second argument being the header of that frame. The first argument
        should initially be an empty FITS header, which will get updated each
        time.

        Args:
            master_header:    The header to use for the stacked master frame,
                describing the frames being stacked. On exit contains only
                keywords shared with frame header and only if their
                corresponding values match.

            frame_header:    The header of an individual frame being added to
                the stack.

            filename:    The filename where the header was read from. Only used
                for reporting errors.

        Returns:
            None
        """

        if master_header:
            delete_indices = []
            for card_index, master_card in enumerate(master_header.cards):
                delete = True
                for frame_card in frame_header.cards:
                    if (
                            frame_card[0] == master_card[0]
                            and
                            frame_card[1] == master_card[1]
                    ):
                        delete = False
                        break
                if delete:
                    if master_card[0] == 'IMAGETYP':
                        raise ImageMismatchError(
                            (
                                'Attempting to combine images with '
                                'IMAGETYP = %s and IMAGETYP=%s into a master!'
                            )
                            %
                            (master_card[1], frame_header['IMAGETYP'])
                        )
                    delete_indices.insert(0, card_index)
            map(master_header.pop, delete_indices)
        else:
            master_header.extend(frame_header.cards)
            if 'IMAGETYP' not in master_header:
                raise BadImageError('Image %s does not define IMAGETYP'
                                    %
                                    filename)

    #Re-factoring to reduce locals will make things less readable.
    #pylint: disable=too-many-locals
    @staticmethod
    def stack(frame_list,
              *,
              outlier_threshold,
              average_func,
              min_valid_values,
              max_iter,
              exclude_mask):
        """
        Create a master by stacking a list of frames.

        Args:
            frame_list:    The frames to stack. Should be a list of
                FITS filenames.

            outlier_threshold:    See same name argument to
                `general_purpose_stats.iterative_rejection_average`

            average_func:    See same name argument to
                `general_purpose_stats.iterative_rejection_average`

            min_valid_values:    The minimum number of valid values to average
                for each pixel. If outlier rejection or masks results in fewer
                than this, the corresponding pixel gets a bad pixel mask.

            exclude_mask:    A bitwise or of mask flags, any of which result in
                the corresponding pixels being excluded from the averaging.
                Other mask flags in the input frames are ignored, treated
                as clean.

            max_iter:    See same name argument to
                `general_purpose_stats.iterative_rejection_average`

        Returns:
            master_values:    The best estimate for the values of the maseter at
                each pixel.

            master_stdev:    The best estimate of the standard deviation of the
                master pixels.

            master_mask:    The pixel quality mask for the master.

            master_header:    The header to use for the newly created
                master frame.
        """

        #pylint triggers on doxygen commands.
        #pylint: disable=anomalous-backslash-in-string
        @staticmethod
        def document_in_header(header):
            """
            Document how the stacking was done in the given header.

            The following extra keywords are added:
            \verbatim
                NUMFCOMB: The number of frames combined in this master.
                ORIGF%(04)d: The base filename of each original frame added. The
                    keyword will get %-substituted with the frame index.
                OUTLTHRS: The threshold for marking pixel values as outliers in
                    units of RMS deviation from final value.
                AVRGFUNC: The __name__ attribute of the averaging function used.
                MINAVGPX: The minimum number of valid pixel values
                    contributing to a pixel's average required to consider the
                    resulting master pixel valid.
                MAXREJIT: The maximum number of rejection/averaging iterations
                    allowed.
                XCLUDMSK: Pixels with masks or-ing to true with this value were
                    excluded from the average.
            \endverbatim

            Args:
                header:    The header to add the stacking configuration to.

            Returns:
                None
            """

            header['NUMFCOMB'] = (len(frame_list),
                                  'Number frames combined in master')
            for index, fname in enumerate(frame_list):
                header['ORIGF%(04)d' % index] = (
                    fname,
                    'Original frame contributing to master'
                )
            header['OUTLTHRS'] = (
                outlier_threshold,
                'The threshold for discarding outlier pixels'
            )
            header['AVRGFUNC'] = (
                average_func.__name__,
                'The averaging function used used for stacking'
            )
            header['MINAVGPX'] = (
                min_valid_values,
                'The minimum number of valid pixels required.'
            )
            header['MAXREJIT'] = (
                max_iter,
                'Max number of rejection/averaging iterations'
            )
            header['XCLUDMSK'] = (
                exclude_mask,
                'Pixels matching any of this mask were excluded'
            )
            header['IMAGETYP'] = 'master' + header['IMAGETYP']
        #pylint: enable=anomalous-backslash-in-string

        pixel_values = None
        master_header = fits.Header()
        for frame_index, frame_fname in enumerate(frame_list):
            image, mask, header = read_image_components(frame_fname,
                                                        read_error=False,
                                                        read_header=True)
            MasterMaker._update_stack_header(master_header, header, frame_fname)

            if pixel_values is None:
                pixel_values = numpy.empty((len(frame_list),) + image.shape)

            pixel_values[frame_index] = image
            pixel_values[frame_index][
                numpy.bitwise_and(mask, exclude_mask).astype(bool)
            ] = numpy.nan

        master_values, master_stdev, master_num_averaged = (
            iterative_rejection_average(pixel_values,
                                        outlier_threshold=outlier_threshold,
                                        average_func=average_func,
                                        max_iter=max_iter,
                                        axis=0,
                                        mangle_input=True,
                                        keepdims=False)
        )

        master_mask = numpy.full(pixel_values.shape(),
                                 mask_flags['CLEAR'],
                                 dtype='int8')
        master_mask[master_num_averaged < min_valid_values] = mask_flags['BAD']

        document_in_header(master_header)

        return master_values, master_stdev, master_mask, master_header
    #pylint: enable=too-many-locals

    def __init__(self,
                 *,
                 outlier_threshold=5.0,
                 average_func=numpy.nanmedian,
                 min_valid_values=5,
                 max_iter=numpy.inf,
                 exclude_mask=(
                     mask_flags['FAULT']
                     | mask_flags['HOT']
                     | mask_flags['COSMIC']
                     | mask_flags['OUTER']
                     | mask_flags['OVERSATURATED']
                     | mask_flags['LEAKED']
                     | mask_flags['SATURATED']
                     | mask_flags['BAD']
                     | mask_flags['NAN']
                 )):
        """
        Create a master maker with the given default stacking configuration.

        Args:
            See the keyword only arguments to the `stack` method.

        Returns:
            None
        """

        print('Number positional arguments: '
              +
              repr(self.__init__.__code__.co_argcount))
        self.stacking_options = dict(
            outlier_threshold=outlier_threshold,
            average_func=average_func,
            min_valid_values=min_valid_values,
            max_iter=max_iter,
            exclude_mask=exclude_mask
        )

    def __call__(self,
                 frame_list,
                 output_fname,
                 compress=True,
                 **stacking_options):
        """
        Create a master by stacking the given frames.

        The header of the craeted frame contains all keywords that are
        common and with consistent value from the input frames. In addition the
        following keywords are added:

        Args:
            frame_list:    A list of the frames to stack (FITS filenames).

            output_fname:    The name of the output file to create.

            compress:    Should the final result be compressed?

            stacking_options:    Keyword only arguments allowing overriding the
                stacking configuration specified at construction for this
                stack only.

        Returns:
            None
        """

        for option_name, default_value in self.stacking_options.items():
            if option_name not in stacking_options:
                stacking_options[option_name] = default_value

        #pylint false positive
        #pylint: disable=missing-kwoa
        values, stdev, mask, header = self.stack(frame_list, **stacking_options)
        #pylint: enable=missing-kwoa

        header_list = [header,
                       fits.Header([('IMAGETYP', 'error')]),
                       fits.Header([('IMAGETYP', 'mask')])]
        if compress:
            image_list = [values, stdev, mask]
            hdulist = fits.HDUList(
                fits.CompImageHDU(data, header)
                for data, header in zip(image_list, header_list)
            )
        else:
            hdulist = [
                fits.PrimaryHDU(data=values, header=header),
                fits.ImageHDU(data=stdev, header=header_list[1]),
                fits.ImageHDU(data=mask, header=header_list[2]),
            ]
        hdulist.writeto(output_fname)
#pylint: enable=too-few-public-methods

if __name__ == '__main__':
    make_master = MasterMaker()
    print(repr(make_master.__dict__))
