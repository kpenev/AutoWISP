"""Define classes for creating master calibration frames."""

from functools import reduce
import logging
from os.path import exists

import numpy

from astropy.io import fits

from autowisp.fits_utilities import read_image_components, update_stack_header
from autowisp.iterative_rejection_util import iterative_rejection_average
from autowisp.image_calibration.mask_utilities import mask_flags
from autowisp.image_calibration.fits_util import create_result

from autowisp import Processor


# pylint does not count __call__ but should.
# pylint: disable=too-few-public-methods
class MasterMaker(Processor):
    """
    Implement the simplest & fully generalizable procedure for making a master.

    Attributes:
        default_exclude_mask:    The default bit-mask indicating all flags which
            should result in a pixel being excluded from the averaging.

        stacking_options:    A dictionary with the default configuration of how
            to perform the stacking. The expected keys exactly match the keyword
            only arguments of the :meth:`stack` method.

    Examples:

        >>> #Create an object for stacking frames to masters, overwriting the
        >>> #default outlier threshold and requiring at least 10 frames be
        >>> #stacked
        >>> make_master = MasterMaker(outlier_threshold=10.0,
        >>>                           min_valid_frames=10)
        >>>
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

    _logger = logging.getLogger(__name__)

    default_exclude_mask = ("BAD",)

    def __init__(
        self,
        *,
        outlier_threshold=5.0,
        average_func=numpy.nanmedian,
        min_valid_frames=10,
        min_valid_values=5,
        max_iter=numpy.inf,
        exclude_mask=default_exclude_mask,
        compress=16,
        add_averaged_keywords=(),
    ):
        """
        Create a master maker with the given default stacking configuration.

        Keyword Args:
            compress:    If None or False, the final result is not
                compressed. Otherwise, this is the quantization level used for
                compressing the image (see `astropy.io.fits` documentation).

            all oothers:    See the keyword only arguments to the :meth:`stack`
                method.

        Returns:
            None
        """

        super().__init__()
        self.stacking_options = {
            "outlier_threshold": outlier_threshold,
            "average_func": average_func,
            "min_valid_frames": min_valid_frames,
            "min_valid_values": min_valid_values,
            "max_iter": max_iter,
            "exclude_mask": exclude_mask,
            "add_averaged_keywords": add_averaged_keywords,
        }
        self.compress = compress

    def prepare_for_stacking(self, image):
        """
        Override with any useful pre-processing of images before stacking.

        Args:
            image:    One of the images to include in the stack.

        Returns:
            same type as image:
                The image to actually include in the stack. Return None if the
                image should be excluded.
        """

        return image

    # Re-factoring to reduce locals will make things less readable.
    # pylint: disable=too-many-locals
    def stack(
        self,
        frame_list,
        *,
        min_valid_frames,
        outlier_threshold,
        average_func,
        min_valid_values,
        max_iter,
        exclude_mask,
        add_averaged_keywords,
        custom_header=None,
    ):
        # No way to avoid
        # pylint: disable=line-too-long
        """
        Create a master by stacking a list of frames.

        Args:
            frame_list:    The frames to stack. Should be a list of
                FITS filenames.

            min_valid_frames:    The smallest number of frames from which to
                create a master. This could be broken if either the input list
                is not long enough or if too many frames are discarded by
                :meth:`prepare_for_stacking`.

            outlier_threshold:    See same name argument to
                :func:`autowisp.iterative_rejection_util.iterative_rejection_average`
                .

            average_func:    See same name argument to
                :func:`autowisp.iterative_rejection_util.iterative_rejection_average`
                .

            min_valid_values:    The minimum number of valid values to average
                for each pixel. If outlier rejection or masks results in fewer
                than this, the corresponding pixel gets a bad pixel mask.

            max_iter:    See same name argument to
                :func:`autowisp.iterative_rejection_util.iterative_rejection_average`
                .

            exclude_mask:    A bitwise or of mask flags, any of which result in
                the corresponding pixels being excluded from the averaging.
                Other mask flags in the input frames are ignored, treated
                as clean.

            add_averaged_keywords:    Any keywords listed will be added to the
                master header with a value calculated using the same averaging
                procedure as the image pixels.

            custom_header:    See same name argument to __call__().

        Returns:
            (tuple):
                2-D array:
                    The best estimate for the values of the maseter at
                    each pixel. None if stacking failed.

                2-D array:
                    The best estimate of the standard deviation of the
                    master pixels. None if stacking failed.

                2-D array:
                    The pixel quality mask for the master. None if
                    stacking failed.

                fits.Header:
                    The header to use for the newly created master frame. None
                    if stacking failed.

                [<FITS filenames>]:
                    List of the frames that were excluded by
                    self.prepare_for_stacking().
        """
        # pylint: enable=line-too-long

        def document_in_header(header):
            """
            Document how the stacking was done in the given header.

            The following extra keywords are added::

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

            Args:
                header:    The header to add the stacking configuration to.

            Returns:
                None
            """

            header["NUMFCOMB"] = (
                len(frame_list),
                "Number frames combined in master",
            )
            for index, fname in enumerate(frame_list):
                header[f"ORIGF{index:03d}"] = (
                    fname,
                    "Original frame contributing to master",
                )
            header["OUTLTHRS"] = (
                repr(outlier_threshold),
                "The threshold for discarding outlier pixels",
            )
            header["AVRGFUNC"] = (
                average_func.__name__,
                "The averaging function used used for stacking",
            )
            header["MINAVGPX"] = (
                min_valid_values,
                "The minimum number of valid pixels required.",
            )
            header["MAXREJIT"] = (
                max_iter if numpy.isfinite(max_iter) else str(max_iter),
                "Max number of rejection/averaging iterations",
            )
            header["XCLUDMSK"] = (
                ",".join(exclude_mask),
                "Pixels matching any of this mask were excluded",
            )
            header["IMAGETYP"] = "master" + header["IMAGETYP"]

        # pylint: enable=anomalous-backslash-in-string

        if custom_header is None:
            custom_header = {}

        if len(frame_list) < min_valid_frames:
            return None, None, None, None, []

        pixel_values = None
        header_values = None
        master_header = fits.Header(custom_header.items())
        frame_index = 0
        discarded_frames = []
        first_frame = True
        exclude_mask_bits = reduce(
            lambda bits, pix_condition: numpy.bitwise_or(
                bits, mask_flags[pix_condition]
            ),
            exclude_mask,
            0,
        )
        for frame_fname in frame_list:
            # False positive
            # pylint: disable=unbalanced-tuple-unpacking
            image, mask, header = read_image_components(
                frame_fname, read_error=False, read_header=True
            )
            # pylint: enable=unbalanced-tuple-unpacking

            stack_image = self.prepare_for_stacking(image)
            if stack_image is None:
                discarded_frames.append(frame_fname)
            else:
                update_stack_header(
                    master_header, header, frame_fname, first_frame
                )
                first_frame = False

                if pixel_values is None:
                    pixel_values = numpy.empty((len(frame_list),) + image.shape)
                if header_values is None:
                    header_values = numpy.empty(
                        (len(frame_list), len(add_averaged_keywords))
                    )

                pixel_values[frame_index] = stack_image
                exclude_pix = numpy.bitwise_and(mask, exclude_mask_bits).astype(
                    bool
                )
                if exclude_pix.any():
                    pixel_values[frame_index][exclude_pix] = numpy.nan
                    self._logger.warning(
                        "Excluding %d masked pixels from %s",
                        exclude_pix.sum(),
                        frame_fname,
                    )
                for kw_index, keyword in enumerate(add_averaged_keywords):
                    # False positive
                    # pylint: disable=unsubscriptable-object
                    header_values[frame_index, kw_index] = header[keyword]
                    # pylint: enable=unsubscriptable-object
                frame_index += 1

        if frame_index < min_valid_frames:
            return None, None, None, None, discarded_frames

        pixel_values = pixel_values[:frame_index]

        master_values, master_stdev, master_num_averaged = (
            iterative_rejection_average(
                pixel_values,
                outlier_threshold=outlier_threshold,
                average_func=average_func,
                max_iter=max_iter,
                axis=0,
                mangle_input=True,
                keepdims=False,
            )
        )
        averaged_header, _, _ = iterative_rejection_average(
            header_values,
            outlier_threshold=outlier_threshold,
            average_func=average_func,
            max_iter=max_iter,
            axis=0,
            mangle_input=True,
            keepdims=False,
        )

        master_mask = numpy.full(
            pixel_values[0].shape, mask_flags["CLEAR"], dtype="int8"
        )
        master_mask[master_num_averaged < min_valid_values] = mask_flags["BAD"]

        document_in_header(master_header)
        for keyword, value in zip(add_averaged_keywords, averaged_header):
            assert keyword not in master_header
            master_header[keyword] = value

        return (
            master_values,
            master_stdev,
            master_mask,
            master_header,
            discarded_frames,
        )

    # pylint: enable=too-many-locals

    def __call__(
        self,
        frame_list,
        output_fname,
        *,
        allow_overwrite=False,
        custom_header=None,
        compress=None,
        **stacking_options,
    ):
        """
        Create a master by stacking the given frames.

        The header of the craeted frame contains all keywords that are
        common and with consistent value from the input frames. In addition the
        following keywords are added:

        Args:
            frame_list:    A list of the frames to stack (FITS filenames).

            output_fname:    The name of the output file to create. Can involve
                substitutions of any header keywords of the generated file.

            compress:    See :meth:`__init__`

            allow_overwrite:    See same name argument
                to autowisp.image_calibration.fits_util.create_result.

            custom_header(dict):    A collection of keywords to use in addition
                to/instead of what is in the input frames header.

            stacking_options:    Keyword only arguments allowing overriding the
                stacking configuration specified at construction for this
                stack only.

        Returns:
            bool:
                Whether creating the master succeeded.

            [<FITS filenames>]:
                Frames which were discarded during stacking.
        """

        if custom_header is None:
            custom_header = {}

        for option_name, default_value in self.stacking_options.items():
            if option_name not in stacking_options:
                stacking_options[option_name] = default_value

        # pylint false positive
        # pylint: disable=missing-kwoa
        values, stdev, mask, header, discarded_frames = self.stack(
            frame_list, custom_header=custom_header, **stacking_options
        )
        # pylint: enable=missing-kwoa

        if values is None:
            self._logger.error(
                "Failed to create master %s!", repr(output_fname)
            )
        else:
            create_result(
                image_list=[values, stdev, mask],
                header=header,
                result_fname=output_fname,
                compress=compress or self.compress,
                allow_overwrite=allow_overwrite,
            )
            assert exists(output_fname)

        return values is not None, discarded_frames


# pylint: enable=too-few-public-methods

if __name__ == "__main__":
    make_master = MasterMaker()
    print(repr(make_master.__dict__))
