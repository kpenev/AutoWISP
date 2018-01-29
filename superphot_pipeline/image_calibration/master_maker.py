"""Define classes for creating master calibration frames."""

import numpy

from astropy.io import fits

from superphot_pipeline.image_utilities import read_image_components
from superphot_pipeline.iterative_rejection_util import iterative_rejection_average
from superphot_pipeline.image_calibration.mask_utilities import mask_flags
from superphot_pipeline.image_calibration.fits_util import create_result

from superphot_pipeline.pipeline_exceptions import\
    ImageMismatchError,\
    BadImageError
from superphot_pipeline import Processor

#pylint does not count __call__ but should.
#pylint: disable=too-few-public-methods
class MasterMaker(Processor):
    """
    Implement the simplest & fully generalizable procedure for making a master.

    Attrs:
        default_exclude_mask:    The default bit-mask indicating all flags which
            should result in a pixel being excluded from the averaging.

        stacking_options:    A dictionary with the default configuration of how
            to perform the stacking. The expected keys exactly match the keyword
            only arguments of the `stack_to_master` method.

    Examples:

        >>> #Create an object for stacking frames to masters, overwriting the
        >>> #default outlier threshold and requiring at least 10 frames be
        >>> #stacked
        >>> make_master = MasterMaker(outlier_threshold=10.0,
        >>>                           min_valid_frames=10)

        >>> #Stack a set of frames to a master, allowing no more than 3
        >>> #averaging/outlier rejection iterations and allowing a minimum of 3
        >>> #valid source pixels to make a master, for this master only.
        >>> make_master(
        >>>     ['f1.fits.fz', 'f2.fits.fz', 'f3.fits.fz', 'f4.fits.fz'],
        >>>     'master.fits.fz',
        >>>     max_iter=3,
        >>>     min_valid_values=3
        >>> )
    """

    default_exclude_mask = (mask_flags['FAULT']
                            | mask_flags['HOT']
                            | mask_flags['COSMIC']
                            | mask_flags['OUTER']
                            | mask_flags['OVERSATURATED']
                            | mask_flags['LEAKED']
                            | mask_flags['SATURATED']
                            | mask_flags['BAD']
                            | mask_flags['NAN'])

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
            print('Checking master header against ' + filename)

            delete_indices = []
            for card_index, master_card in enumerate(master_header.cards):
                delete = next(
                    (False for frame_card in frame_header.cards
                     if frame_card[:2] == master_card[:2]),
                    True
                )
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
            print('Deleting:\n'
                  +
                  '\n'.join([
                      repr(master_header.cards[i][0]) for i in delete_indices
                  ]))
            print('Starting with %d cards' % len(master_header.cards))
            for index in delete_indices:
                del master_header[index]
            print('%d cards remain' % len(master_header.cards))
        else:
            master_header.extend(
                filter(lambda c: tuple(c) != ('', '', ''), frame_header.cards)
            )
            if 'IMAGETYP' not in master_header:
                raise BadImageError('Image %s does not define IMAGETYP'
                                    %
                                    filename)

    def __init__(self,
                 *,
                 outlier_threshold=5.0,
                 average_func=numpy.nanmedian,
                 min_valid_frames=10,
                 min_valid_values=5,
                 max_iter=numpy.inf,
                 exclude_mask=default_exclude_mask):
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
            min_valid_frames=min_valid_frames,
            min_valid_values=min_valid_values,
            max_iter=max_iter,
            exclude_mask=exclude_mask
        )

    #This method is intended to be overriden.
    #pylint: disable=no-self-use
    def prepare_for_stacking(self, image):
        """
        Override with any useful pre-processing of images before stacking.

        Args:
            image:    One of the images to include in the stack.

        Returns:
            stack_image:    The image to actually include in the stack. Return
                None if the image should be excluded.
        """

        return image
    #pylint: enable=no-self-use

    #Re-factoring to reduce locals will make things less readable.
    #pylint: disable=too-many-locals
    def stack(self,
              frame_list,
              *,
              min_valid_frames,
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

            min_valid_frames:    The smallest number of frames from which to
                create a master. This could be broken if either the input list
                is not long enough or if too many frames are discarded by
                self.prepare_for_stacking().

            outlier_threshold:    See same name argument to
                `iterative_rejection_util.iterative_rejection_average`

            average_func:    See same name argument to
                `iterative_rejection_util.iterative_rejection_average`

            min_valid_values:    The minimum number of valid values to average
                for each pixel. If outlier rejection or masks results in fewer
                than this, the corresponding pixel gets a bad pixel mask.

            exclude_mask:    A bitwise or of mask flags, any of which result in
                the corresponding pixels being excluded from the averaging.
                Other mask flags in the input frames are ignored, treated
                as clean.

            max_iter:    See same name argument to
                `iterative_rejection_util.iterative_rejection_average`

        Returns:
            master_values:    The best estimate for the values of the maseter at
                each pixel. None if stacking failed.

            master_stdev:    The best estimate of the standard deviation of the
                master pixels. None if stacking failed.

            master_mask:    The pixel quality mask for the master. None if
                stacking failed.

            master_header:    The header to use for the newly created
                master frame. None if stacking failed.

            discarded_frames:    List of the frames that were excluded by
                self.prepare_for_stacking().
        """

        #pylint triggers on doxygen commands.
        #pylint: disable=anomalous-backslash-in-string
        def document_in_header(header):
            """
            Document how the stacking was done in the given header.

            The following extra keywords are added:
            \verbatim
                NUMFCOMB: The number of frames combined in this master.
                ORIGF%04d: The base filename of each original frame added. The
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
                header['ORIGF%03d' % index] = (
                    fname,
                    'Original frame contributing to master'
                )
            header['OUTLTHRS'] = (
                repr(outlier_threshold),
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
                max_iter if numpy.isfinite(max_iter) else str(max_iter),
                'Max number of rejection/averaging iterations'
            )
            header['XCLUDMSK'] = (
                exclude_mask,
                'Pixels matching any of this mask were excluded'
            )
            header['IMAGETYP'] = 'master' + header['IMAGETYP']
        #pylint: enable=anomalous-backslash-in-string

        if len(frame_list) < min_valid_frames:
            return None, None, None, None, []

        pixel_values = None
        master_header = fits.Header()
        frame_index = 0
        discarded_frames = []
        for frame_fname in frame_list:
            image, mask, header = read_image_components(frame_fname,
                                                        read_error=False,
                                                        read_header=True)
            stack_image = self.prepare_for_stacking(image)
            if stack_image is None:
                discarded_frames.append(image)
            else:
                MasterMaker._update_stack_header(master_header,
                                                 header,
                                                 frame_fname)

                if pixel_values is None:
                    pixel_values = numpy.empty((len(frame_list),) + image.shape)

                pixel_values[frame_index] = stack_image
                pixel_values[frame_index][
                    numpy.bitwise_and(mask, exclude_mask).astype(bool)
                ] = numpy.nan
                frame_index += 1

        if frame_index < min_valid_frames:
            return None, None, None, None, discarded_frames

        pixel_values = pixel_values[:frame_index]

        master_values, master_stdev, master_num_averaged = (
            iterative_rejection_average(pixel_values,
                                        outlier_threshold=outlier_threshold,
                                        average_func=average_func,
                                        max_iter=max_iter,
                                        axis=0,
                                        mangle_input=True,
                                        keepdims=False)
        )

        master_mask = numpy.full(pixel_values[0].shape,
                                 mask_flags['CLEAR'],
                                 dtype='int8')
        master_mask[master_num_averaged < min_valid_values] = mask_flags['BAD']

        document_in_header(master_header)

        return (master_values,
                master_stdev,
                master_mask,
                master_header,
                discarded_frames)
    #pylint: enable=too-many-locals



    def __call__(self,
                 frame_list,
                 output_fname,
                 *,
                 compress=True,
                 allow_overwrite=False,
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

            allow_overwrite:    See same name argument
                to superphot_pipeline.image_calibration.fits_util.create_result.

            stacking_options:    Keyword only arguments allowing overriding the
                stacking configuration specified at construction for this
                stack only.

        Returns:
            discarded_frames:    Frames which were discarded during stacking.
        """

        for option_name, default_value in self.stacking_options.items():
            if option_name not in stacking_options:
                stacking_options[option_name] = default_value

        #pylint false positive
        #pylint: disable=missing-kwoa
        values, stdev, mask, header, discarded_frames = self.stack(
            frame_list,
            **stacking_options
        )
        #pylint: enable=missing-kwoa

        if values is not None:
            create_result(image_list=[values, stdev, mask],
                          header=header,
                          result_fname=output_fname,
                          compress=compress,
                          allow_overwrite=allow_overwrite)

        return discarded_frames
#pylint: enable=too-few-public-methods

if __name__ == '__main__':
    make_master = MasterMaker()
    print(repr(make_master.__dict__))
