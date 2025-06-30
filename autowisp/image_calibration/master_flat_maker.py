"""Define classes for creating master flat frames."""

import logging

import numpy

from autowisp.image_calibration.master_maker import MasterMaker
from autowisp.fits_utilities import read_image_components
from autowisp.image_utilities import get_pointing_from_header
from autowisp.image_calibration.mask_utilities import mask_flags
from autowisp.image_smoothing import ImageSmoother
from autowisp.iterative_rejection_util import (
    iterative_rejection_average,
    iterative_rej_polynomial_fit,
)


class MasterFlatMaker(MasterMaker):
    r"""
    Specialize MasterMaker for making master flat frames.

    Attributes:
        stamp_statistics_config:    Dictionary configuring how stamps statistics
            for stamp-based selection are extracted from the frames.

        stamp_select_config:    Dictionary configuring how stamp-based selection
            is performed. See keyword only arguments of
            :meth:`configure_stamp_selection` for details.

        large_scale_smoother:    A
            :class:`autowisp.image_smoothing.ImageSmoother`
            instance applied  to the ratio of a frame to the reference large
            scale structure before applying it to the frame.

        cloud_check_smoother:
            :class:`autowisp.image_smoothing.ImageSmoother`
            instance used for cloud detection performed on the full flat frames
            after smoothing to the master large scale structure.

        master_stack_config:    Dictionary configuring how to stack the
            individual frames into a master.

        _master_large_scale:    The large scale structure imposed on all master
            flats. Dictionary with keys ``values``, ``stdev``, ``mask`` and
            ``header``. If empty dictionary, nothing is imposed.

    Examples:

        >>> import scipy.ndimage.filters
        >>> from autowisp.image_smoothing import\
        >>>     PolynomialImageSmoother,\
        >>>     SplineImageSmoother,\
        >>>     ChainSmoother,\
        >>>     WrapFilterAsSmoother
        >>>
        >>> #Stamp statistics configuration:
        >>> #  * stamps span half the frame along each dimension
        >>> #  * stamps are detrended by a bi-quadratic polynomial with at most
        >>> #    one rejection iteration, discarding more than 3-sigma outliers.
        >>> #  * for each stamp a iterative rejection mean and variance are
        >>> #    calculated with up to 3 iterations rejecting three or more
        >>> #    sigma outliers.
        >>> stamp_statistics_config = dict(
        >>>     fraction=0.5,
        >>>     smoother=PolynomialImageSmoother(num_x_terms=3,
        >>>                                      num_y_terms=3,
        >>>                                      outlier_threshold=3.0,
        >>>                                      max_iterations=3),
        >>>     average='mean',
        >>>     outlier_threshold=3.0,
        >>>     max_iter=3
        >>> )
        >>>
        >>> #Stamp statistics based selection configuration:
        >>> #  * Stamps with more than 0.1% of their pixels saturated are
        >>> #    discarded
        >>> #  * if variance vs mean quadratic fit has residual of more than 5k
        >>> #    ADU^2, the entire night is considered cloudy.
        >>> #  * individual frames with stamp mean and variance deviating more
        >>> #    than 2*(fit_residual) are discarded as cloudy.
        >>> #  * high master flat will be generated from frames with stamp mean
        >>> #    > 25 kADU, and low master flat from frames with stamp mean < 15
        >>> #    kADU (intermediate frames are discarded).
        >>> stamp_select_config = dict(max_saturated_fraction=1e-4,
        >>>                            var_mean_fit_threshold=2.0,
        >>>                            var_mean_fit_iterations=2,
        >>>                            cloudy_night_threshold=5e3,
        >>>                            cloudy_frame_threshold=2.0,
        >>>                            min_high_mean=2.5e4,
        >>>                            max_low_mean=1.5e4)
        >>>
        >>> #Large scale structure smoothing configuration. For each frame, the
        >>> #large scale struture is corrected by taking the ratio of the frame
        >>> #to the reference (median of all input frames), smoothing this ratio
        >>> #and then dividing by it. The following defines how the smoothing is
        >>> #performed:
        >>> #  * shrink by a factor of 4 in each direction (16 pixels gets
        >>> #    averaged to one).
        >>> #  * Performa a box-filtering with a half-size of 6-pixels using
        >>> #    median averaging
        >>> #  * Perform a bi-cubic spline interpolation smoothing of the
        >>> #    box-filtered image.
        >>> #  * Discard more than 5-sigma outliers if any and re-smooth (no
        >>> #    further iterations allowed)
        >>> #  * Re-interpolate the image back to its original size, using
        >>> #    bicubic interpolation (see zoom_image()).
        >>> #  * The resulting image is scaled to have a mean of 1 (no
        >>> #    configuration for that).
        >>> large_scale_smoother = ChainSmoother(
        >>>     WrapFilterAsSmoother(scipy.ndimage.filters.median_filter,
        >>>                          size=12),
        >>>     SplineImageSmoother(num_x_nodes=3,
        >>>                         num_y_nodes=3,
        >>>                         outlier_threshold=5.0,
        >>>                         max_iter=1),
        >>>     bin_factor=4,
        >>>     zoom_interp_order=3
        >>> )
        >>>
        >>> #Configuration for smoothnig when checking for clouds.
        >>> #After smoothing to the master large scale structure:
        >>> #  * extract a central stamp is extracted from each flat covering
        >>> #    3/4 of the frame along each dimension
        >>> #  * shrink the fractional deviation of that stamp from the master
        >>> #    by a factor of 4 in each dimension
        >>> #  * smooth by median box-filtering with half size of 4 shrunk
        >>> #    pixels
        >>> #  * zoom the frame back out by a factor of 4 in each dimension
        >>> #    (same factor as shrinking, no separater config), using
        >>> #    bi-quadratic interpolation.
        >>> cloud_check_smoother = WrapFilterAsSmoother(
        >>>     scipy.ndimage.filters.median_filter,
        >>>     size=8,
        >>>     bin_factor=4,
        >>>     zoom_interp_order=3
        >>> )
        >>>
        >>> #When stacking masters require:
        >>> #  * At least 10 input frames for a high intensity master and at
        >>> #    least 5 for a low intensity one.
        >>> #  * When creating the stack used to match large-scale structure,
        >>> #    use median averaging with outlier rejection of more than
        >>> #    4-sigma outliers with at most one reject/re-fit iteration.
        >>> #  * When creating the final master use median averaging with
        >>> #    outlier rejection of more than 2-sigma outliers in the positive
        >>> #    and more than 3-sigma in the negative direction with at most 2
        >>> #    reject/re-fit iterations. Create the master compressed and
        >>> #    raise an exception if a file with that name already exists.
        >>> master_stack_config = dict(
        >>>     min_pointing_separation=150.0,
        >>>     large_scale_deviation_threshold=0.05,
        >>>     min_high_combine=10,
        >>>     min_low_combine=5,
        >>>     large_scale_stack_options=dict(
        >>>         outlier_threshold=4,
        >>>         average_func=numpy.nanmedian,
        >>>         min_valid_values=3,
        >>>         max_iter=1,
        >>>         exclude_mask=MasterMaker.default_exclude_mask
        >>>     ),
        >>>     master_stack_options=dict(
        >>>         outlier_threshold=(2, -3),
        >>>         average_func=numpy.nanmedian,
        >>>         min_valid_values=3,
        >>>         max_iter=2,
        >>>         exclude_mask=MasterMaker.default_exclude_mask,
        >>>         compress=True,
        >>>         allow_overwrite=False
        >>>     )
        >>> )
        >>>
        >>> #Create an object for stacking calibrated flat frames to master
        >>> #flats. In addition to the stamp-based rejections:
        >>> #  * reject flats that point within 40 arcsec of each other on the
        >>> #    sky.
        >>> #  * Require at least 10 frames to be combined into a high master
        >>> #    and at least 5 for a low master.
        >>> #  * if the smoothed cloud-check image contains pixels with absolute
        >>> #    value > 5% the frame is discarded as cloudy.
        >>> make_master_flat = MasterFlatMaker(
        >>>     stamp_statistics_config=stamp_statistics_config,
        >>>     stamp_select_config=stamp_select_config,
        >>>     large_scale_smoother=large_scale_smoother,
        >>>     cloud_check_smoother=cloud_check_smoother,
        >>>     master_stack_config=master_stack_config
        >>> )
        >>>
        >>> #Create master flat(s) from the given raw flat frames. Note that
        >>> #zero, one or two master flat frames can be created, depending on
        >>> #the input images. Assume that the raw flat frames have names like
        >>> #10-<fnum>_2.fits.fz, with fnum ranging from 1 to 30 inclusive.
        >>> make_master_flat(
        >>>     ['10-%d_2.fits.fz' % fnum for fnum in range(1, 31)],
        >>>     high_master_fname='high_master_flat.fits.fz',
        >>>     low_master_fname='low_master_flat.fits.fz'
        >>> )
    """

    _logger = logging.getLogger(__name__)

    def _get_stamp_statistics(self, frame_list, **stamp_statistics_config):
        """
        Get relevant information from the stamp of a single input flat frame.

        Args:
            frame_list:    The list of frames for which to extract stamp
                statistics.

        Returns:
            numpy field array:
                An array with fields called ``mean``, ``variance`` and
                ``num_averaged`` with the obvious meanings.
        """

        stamp_statistics = numpy.empty(
            len(frame_list),
            dtype=[
                ("mean", numpy.float64),
                ("variance", numpy.float64),
                ("num_averaged", numpy.uint),
            ],
        )
        for frame_index, fname in enumerate(frame_list):
            # False positive
            # pylint: disable=unbalanced-tuple-unpacking
            image, mask = read_image_components(
                fname, read_error=False, read_header=False
            )
            # pylint: enable=unbalanced-tuple-unpacking

            y_size = int(image.shape[0] * stamp_statistics_config["fraction"])
            x_size = int(image.shape[1] * stamp_statistics_config["fraction"])
            x_off = (image.shape[1] - x_size) // 2
            y_off = (image.shape[0] - y_size) // 2
            num_saturated = (
                numpy.bitwise_and(
                    mask[y_off : y_off + y_size, x_off : x_off + x_size],
                    numpy.bitwise_or(
                        mask_flags["OVERSATURATED"],
                        numpy.bitwise_or(
                            mask_flags["LEAKED"], mask_flags["SATURATED"]
                        ),
                    ),
                )
                .astype(bool)
                .sum()
            )

            if num_saturated > (
                self.stamp_select_config["max_saturated_fraction"]
                * (x_size * y_size)
            ):
                stamp_statistics[frame_index] = numpy.nan, numpy.nan, numpy.nan
            else:
                smooth_stamp = stamp_statistics_config["smoother"].detrend(
                    # False positive
                    # pylint: disable=unsubscriptable-object
                    image[y_off : y_off + y_size, x_off : x_off + x_size]
                    # pylint: enable=unsubscriptable-object
                )

                stamp_statistics[frame_index] = iterative_rejection_average(
                    smooth_stamp.flatten(),
                    average_func=stamp_statistics_config["average"],
                    max_iter=stamp_statistics_config["max_iter"],
                    outlier_threshold=(
                        stamp_statistics_config["outlier_threshold"]
                    ),
                    mangle_input=True,
                )

        stamp_statistics["variance"] = numpy.square(
            stamp_statistics["variance"]
        ) * (stamp_statistics["num_averaged"] - 1)
        return stamp_statistics

    # Splitting will not improve readability
    # pylint: disable=too-many-locals
    def _classify_from_stamps(
        self, frame_list, stamp_statistics_config, stamp_select_config
    ):
        """
        Classify frames by intensity: (high/low/ntermediate), or flag as cloudy.

        Args:
            frame_list:    The list of frames to create masters from.

        Returns:
            (tuple):
                filename list:
                    The list of high intensity non-cloudy frames.

                filename list:
                    The list of low intensity non-cloudy frames.

                filename list:
                    The list of medium intensity non-cloudy frames.

                filename list:
                    The list of frames suspected of containing clouds in
                    their central stamps.
        """

        stamp_statistics = self._get_stamp_statistics(
            frame_list, **stamp_statistics_config
        )

        if (
            stamp_select_config["cloudy_night_threshold"] is not None
            or stamp_select_config["cloudy_frame_threshold"] is not None
        ):
            (fit_coef, residual, best_fit_variance) = (
                iterative_rej_polynomial_fit(
                    x=stamp_statistics["mean"],
                    y=stamp_statistics["variance"],
                    order=2,
                    outlier_threshold=stamp_select_config[
                        "var_mean_fit_threshold"
                    ],
                    max_iterations=stamp_select_config[
                        "var_mean_fit_iterations"
                    ],
                    return_predicted=True,
                )
            )

            stat_msg = (
                f'\t{"frame":50s}|{"mean":10s}|{"std":10s}|{"fitstd":10s}\n'
                + "\t"
                + 92 * "_"
                + "\n"
            )
            for fname, stat, fitvar in zip(
                frame_list, stamp_statistics, best_fit_variance
            ):
                stat_msg += (
                    f"\t{fname:50s}|{stat['mean']:10g}|"
                    f"{stat['variance']**0.5:10g}|{fitvar**0.5:10g}\n"
                )
            self._logger.debug("Flat stamp pixel statistics:\n")

            self._logger.debug(
                "Best fit quadratic: " "%f + %f*m + %f*m^2; residual=%f",
                *fit_coef,
                residual,
            )

            if (stamp_select_config["cloudy_night_threshold"] is not None) and (
                residual > stamp_select_config["cloudy_night_threshold"]
            ):
                return [], [], [], frame_list

        high = stamp_statistics["mean"] > stamp_select_config["min_high_mean"]
        low = stamp_statistics["mean"] < stamp_select_config["max_low_mean"]

        if self.stamp_select_config.get("cloudy_frame_threshold") is not None:
            cloudy = (
                numpy.abs(stamp_statistics["variance"] - best_fit_variance)
                > stamp_select_config["cloudy_frame_threshold"] * residual
            )
        else:
            cloudy = False

        medium = numpy.arange(len(frame_list))[
            numpy.logical_and(
                numpy.logical_and(
                    numpy.logical_not(high), numpy.logical_not(low)
                ),
                numpy.logical_not(cloudy),
            )
        ]

        high = numpy.arange(len(frame_list))[
            numpy.logical_and(high, numpy.logical_not(cloudy))
        ]
        low = numpy.arange(len(frame_list))[
            numpy.logical_and(low, numpy.logical_not(cloudy))
        ]
        cloudy = numpy.arange(len(frame_list))[cloudy]
        return (
            [frame_list[i] for i in high],
            [frame_list[i] for i in low],
            [frame_list[i] for i in medium],
            [frame_list[i] for i in cloudy],
        )

    # pylint: disable=too-many-locals

    def _find_colocated(self, frame_list):
        """
        Split the list of frames into well isolated ones and co-located ones.

        Args:
            frame_list:    A list of the frames to create the masters from
                (FITS filenames).

        Returns:
            (tuple):
                filename list:
                    The ist of frames sufficiently far in pointing from all
                    other frames.

                filename list:
                    The list of frames which have at least one other
                    frame too close in pointing to them.
        """

        if self.master_stack_config["min_pointing_separation"] is None:
            return frame_list, []

        frame_pointings = [get_pointing_from_header(f) for f in frame_list]
        colocated = numpy.full(len(frame_list), False)
        for reference_index, reference_pointing in enumerate(frame_pointings):
            for index, pointing in enumerate(
                frame_pointings[reference_index + 1 :]
            ):
                if (
                    reference_pointing.separation(pointing).to("arcsec").value
                    < self.master_stack_config["min_pointing_separation"]
                ):
                    colocated[reference_index] = True
                    colocated[index + reference_index + 1] = True

        isolated = numpy.arange(len(frame_list))[numpy.logical_not(colocated)]
        colocated = numpy.arange(len(frame_list))[colocated]
        return (
            [frame_list[i] for i in isolated],
            [frame_list[i] for i in colocated],
        )

    def __init__(
        self,
        *,
        stamp_statistics_config=None,
        stamp_select_config=None,
        large_scale_smoother=None,
        cloud_check_smoother=None,
        master_stack_config=None,
    ):
        """
        Create object for creating master flats out of calibrated flat frames.

        Args:
            stamp_statistics_config:    A dictionary mith arguments to pass
                to :meth:`configure_stamp_statistics`.

            stamp_select_cofig:    A dictionary with arguments to pass
                to :meth:`configure_stamp_selection`.

            large_scale_smoother:    An ImageSmoother instance used when
                matching large scale structure of individual flats to master.

            cloud_check_smoother:    An ImageSmoother instance used when checkng
                the full frames for clouds (after stamps are checked).

            master_stack_config:    Configuration of how to stack frames to a
                master after matching their large scale structure. See
                example in class doc-string for details.

        Returns:
            None
        """

        super().__init__()

        self.stamp_statistics_config = {}
        self.stamp_select_config = {}
        self.large_scale_smoother = large_scale_smoother
        self.cloud_check_smoother = cloud_check_smoother
        self.master_stack_config = master_stack_config

        if stamp_statistics_config is not None:
            self.configure_stamp_statistics(**stamp_statistics_config)

        if stamp_select_config is not None:
            self.configure_stamp_selection(**stamp_select_config)

        self._master_large_scale = {}

    def configure_stamp_statistics(
        self,
        *,
        fraction=None,
        smoother=None,
        outlier_threshold=None,
        max_iter=None,
        average=None,
    ):
        """
        Configure extraction of stamp satistics for rejection & high/low split.

        Any arguments left as None are not updated.

        Args:
            fraction:    The fraction of the frame size that is included in the
                stamp along each dimension (i.e. fraction=0.5 means 1/4 of all
                frame pixels will be incruded in the stamp).

            smoother:    An ImageSmoother instance used used for
                de-trending the stamps before extracting statistics

            outlier_threshold:    The threshold in units of RMS deviation from
                the average above which pixels are considered outliers from the
                de-trending function and discarded from its fit.

            max_iter:    The maximum number of fit/reject iterations to perform
                before declaring the de-trending function final.

            average:    How to compute the average. Should be either ``'mean'``
                or ``'median'``.

        Returns:
            None
        """

        if fraction is not None:
            assert isinstance(fraction, (int, float))
            self.stamp_statistics_config["fraction"] = fraction

        if smoother is not None:
            assert isinstance(smoother, ImageSmoother)
            self.stamp_statistics_config["smoother"] = smoother

        if outlier_threshold is not None:
            self.stamp_statistics_config["outlier_threshold"] = (
                outlier_threshold
            )

        if max_iter is not None:
            assert isinstance(max_iter, int)
            self.stamp_statistics_config["max_iter"] = max_iter

        if average is not None:
            assert average in [numpy.nanmean, numpy.nanmedian]
            self.stamp_statistics_config["average"] = average

    def configure_stamp_selection(
        self,
        *,
        max_saturated_fraction=None,
        var_mean_fit_threshold=None,
        var_mean_fit_iterations=None,
        cloudy_night_threshold=None,
        cloudy_frame_threshold=None,
        min_high_mean=None,
        max_low_mean=None,
    ):
        """
        Configure stamp-based frame selection and high/low split.

        Args:
            max_saturated_fraction:    The maximum fraction of stamp pixels
                allowed to be saturated before discarding the frame.

            cloudy_night_threshold:    The maximum residual of the variance vs
                mean quadratic fit before a night is declared cloudy. If None,
                this check is disabled.

            cloudy_frame_threshold:    The maximum deviation in units of the RMS
                residual of the fit an individual frame's variance vs mean from
                the var(mean) quadratic fit before the frame is declared cloudy.

            min_high_mean:    The minimum mean of the stamp pixels in order to
                consider the frame high intensity.

            max_low_mean:    The maximum mean of the stamp pixels in order to
                consider the frame low intensity. Must not overlap
                with min_high_mean.

        Returns:
            None
        """

        if max_saturated_fraction is not None:
            assert isinstance(max_saturated_fraction, (int, float))
            self.stamp_select_config["max_saturated_fraction"] = (
                max_saturated_fraction
            )

        if var_mean_fit_threshold is not None:
            assert isinstance(var_mean_fit_threshold, (int, float))
            self.stamp_select_config["var_mean_fit_threshold"] = (
                var_mean_fit_threshold
            )

        if var_mean_fit_iterations is not None:
            assert not numpy.isfinite(var_mean_fit_iterations) or isinstance(
                var_mean_fit_iterations, int
            )
            self.stamp_select_config["var_mean_fit_iterations"] = (
                var_mean_fit_iterations
            )

        if cloudy_night_threshold is not None:
            assert isinstance(cloudy_night_threshold, (int, float))
            self.stamp_select_config["cloudy_night_threshold"] = (
                cloudy_night_threshold
            )

        if cloudy_frame_threshold is not None:
            assert isinstance(cloudy_frame_threshold, (int, float))
            self.stamp_select_config["cloudy_frame_threshold"] = (
                cloudy_frame_threshold
            )

        if min_high_mean is not None:
            assert isinstance(min_high_mean, (int, float))
            self.stamp_select_config["min_high_mean"] = min_high_mean

        if max_low_mean is not None:
            assert isinstance(max_low_mean, (int, float))
            self.stamp_select_config["max_low_mean"] = max_low_mean

    def prepare_for_stacking(self, image):
        r"""
        Match image large scale to `self._master_large_scale` if not empty.

        Args:
            image:    The image to transform large scale structure of.

        Returns:
            2-D array:
                If :attr:`_master_large_scale` is an empty dictionary, this is
                just :attr:`image`\ . Otherwise, :attr:`image` is transformed to
                have the same large scale structure as
                :attr:`_master_large_scale`\ , while the small scale
                structure is preserved. :attr:`image` is also checked for
                clouds, and discarded if cloudy.
        """

        if not self._master_large_scale:
            return image

        corrected_image = image * self.large_scale_smoother.smooth(
            self._master_large_scale["values"] / image
        )
        max_abs_deviation = numpy.abs(
            self.cloud_check_smoother.smooth(
                corrected_image / self._master_large_scale["values"] - 1.0
            )
        ).max()
        if (
            max_abs_deviation
            > self.master_stack_config["large_scale_deviation_threshold"]
        ):
            return None

        return corrected_image / corrected_image.mean()

    # TODO: implement full header documentation.
    # More configuration can be overwritten for master flats.
    # pylint: disable=arguments-differ
    def __call__(
        self,
        frame_list,
        high_master_fname,
        low_master_fname,
        *,
        compress=True,
        allow_overwrite=False,
        stamp_statistics_config=None,
        stamp_select_config=None,
        master_stack_config=None,
        custom_header=None,
    ):
        # No good way to avoid
        # pylint: disable=line-too-long
        """
        Attempt to create high & low master flat from the given frames.

        Args:
            frame_list:    A list of the frames to create the masters from
                (FITS filenames).

            high_master_fname:    The filename to save the generated high
                intensity master flat if one is successfully created.

            low_master_fname:    The filename to save the generated low
                intensity master flat if one is successfully created.

            compress:    Should the final result be compressed?

            allow_overwrite:    See same name argument
                to
                :func:`autowisp.image_calibration.fits_util.create_result`
                .

            stamp_statistics_config(dict):    Overwrite the configuration
                for extracting stamp statistics for this set of frames only.

            stamp_select_config(dict):    Overwrite the stamp selection
                configuration for this set of frames only.

            master_stack_config(dict):    Overwrite the configuration for how to
                stack frames to a master for this set of frames only.

            custom_header:    See same name argument to
                :func:`autowisp.image_calibration.master_maker.MasterMaker.__call__`
                .

        Returns:
            dict:
                A dictionary splitting the input list of frames into

                    high:
                        All entries from :attr:`frame_list` which were deemed
                        suitable for inclusion in a master high flat.

                    low:
                        All entries from :attr:`frame_list` which were deemed
                        suitable for inclusion in a master low flat.

                    medium:
                        All entries from :attr:`frame_list` which were of
                        intermediate intensity and thus not included in any
                        master, but for which no issues were detected.

                    colocated:
                        All entries from :attr:`frame_list` which were excluded
                        because they were not sufficiently isolated from their
                        closest neighbor to guarantee that stars do not overlap.

                    cloudy:
                        All entries from :attr:`frame_list` which were flagged
                        as cloudy either based on their stamps or on the final
                        full-frame cloud check.
        """
        # pylint: enable=line-too-long

        if stamp_statistics_config is None:
            stamp_statistics_config = {}
        if stamp_select_config is None:
            stamp_select_config = {}
        if master_stack_config is None:
            master_stack_config = {}
        if custom_header is None:
            custom_header = {}

        stamp_statistics_config = {
            **self.stamp_statistics_config,
            **stamp_statistics_config,
        }
        stamp_select_config = {
            **self.stamp_select_config,
            **stamp_select_config,
        }
        master_stack_config["large_scale_stack_options"] = {
            **self.master_stack_config["large_scale_stack_options"],
            "custom_header": custom_header,
            **master_stack_config.get("large_scale_stack_options", {}),
        }
        master_stack_config["master_stack_options"] = {
            **self.master_stack_config["master_stack_options"],
            "custom_header": custom_header,
            **master_stack_config.get("master_stack_options", {}),
        }

        self._logger.debug(
            "Creating master flats (high:%s, low:%s) with:\n"
            "\n\tstamp statistics config:\n\t\t%s"
            "\n\tstamp selection config:\n\t\t%s"
            "\n\tmaster stacking config:\n\t\t%s",
            high_master_fname,
            low_master_fname,
            "\n\t\t".join(
                f"{k}: {v!r}" for k, v in stamp_statistics_config.items()
            ),
            "\n\t\t".join(
                f"{k}: {v!r}" for k, v in stamp_select_config.items()
            ),
            "\n\t\t".join(
                f"{k}: {v!r}" for k, v in master_stack_config.items()
            ),
        )

        frames = {}
        isolated_frames, frames["colocated"] = self._find_colocated(frame_list)

        self._logger.debug(
            "Isolated flats:\n\t%s", "\n\t".join(isolated_frames)
        )
        self._logger.debug(
            "Colocated flats:\n\t%s" "\n\t".join(frames["colocated"])
        )

        frames["high"], frames["low"], frames["medium"], frames["cloudy"] = (
            self._classify_from_stamps(
                isolated_frames, stamp_statistics_config, stamp_select_config
            )
        )

        log_msg = "Stamp frame classification:\n"
        for key, filenames in frames.items():
            log_msg += f"\t{key:s} ({len(filenames):d}):\n\t\t" + "\n\t\t".join(
                filenames
            )
        self._logger.debug(log_msg)

        min_combine = {
            "high": master_stack_config.get(
                "min_high_combine", self.master_stack_config["min_high_combine"]
            ),
            "low": master_stack_config.get(
                "min_low_combine", self.master_stack_config["min_low_combine"]
            ),
        }

        success = {"high": False, "low": False}
        if len(frames["high"]) >= min_combine["high"]:
            self._master_large_scale = {}
            # False positive
            # pylint: disable=missing-kwoa
            (
                self._master_large_scale["values"],
                self._master_large_scale["stdev"],
                self._master_large_scale["mask"],
                self._master_large_scale["header"],
                discarded_frames,
            ) = self.stack(
                frames["high"],
                min_valid_frames=min_combine["high"],
                **master_stack_config["large_scale_stack_options"],
            )
            # pylint: enable=missing-kwoa
            assert not discarded_frames

            success["high"], more_cloudy_frames = super().__call__(
                frames["high"],
                high_master_fname,
                min_valid_frames=min_combine["high"],
                **master_stack_config["master_stack_options"],
            )
            frames["cloudy"].extend(more_cloudy_frames)
            for frame in more_cloudy_frames:
                frames["high"].remove(frame)

            if len(frames["low"]) > min_combine["low"]:
                success["low"], more_cloudy_frames = super().__call__(
                    frames["low"],
                    low_master_fname,
                    min_valid_frames=min_combine["low"],
                    **master_stack_config["master_stack_options"],
                )
                frames["cloudy"].extend(more_cloudy_frames)
                for frame in more_cloudy_frames:
                    frames["low"].remove(frame)
            else:
                self._logger.warning(
                    "Skipping low master flat since only %d frames remain, "
                    "but %d are required",
                    len(frames["low"]),
                    min_combine["low"],
                )

        log_msg = "Final frame classification:\n"
        for key, filenames in frames.items():
            log_msg += f"\t{key:s} ({len(filenames):d}):\n\t\t" + "\n\t\t".join(
                filenames
            )
        self._logger.info(log_msg)
        return success, frames

    # pylint: disable=arguments-differ
