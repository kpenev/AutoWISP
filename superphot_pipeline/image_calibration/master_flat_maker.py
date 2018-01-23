"""Define classes for creating master flat frames."""

#TODO: pythonize

from superphot_pipeline.image_calibration import MasterMaker

class MasterFlatMaker(MasterMaker):
    """
    Specialize MasterMaker for making master flat frames.

    Attrs:
        min_pointing_offset:    The minimum offset in pointing between flats to
            avoid overlapping stars.

        stamp_config:    Dictionary configuring how stamps statistics for
            stamp-based selection are extracted from the frames.

        stamp_select_config:    Dictionary configuring how stamp-based selection
            is performed. See keyword only arguments of _check_central_stamp for
            details.

        smoothing_config:    Dictionary configuring how to smooth the ratio of a
            frame to the reference large scale structure before applying it to
            the frame. See keyword only arguments of
            _smooth_image for details.

        cloud_check_smoothing_config:    Dictionary configuring how smoothing
            for the purposes of cloud detection is performed on the full flat
            frames after smoothing to the master large scale structure. See
            keyword only arguments of _smooth_image for details.

        large_scale_deviation_threshold:    The threshold on the cloud-check
            image for considering the frame cloudy (in addition to stamp-based
            cloud detection).

    Examples:

        >>> #Stamp statistics configuration:
        >>> #  * stamps span half the frame along each dimension
        >>> #  * stamps are detrended by a bi-quadratic polynomial with at most
        >>> #    one rejection iteration, discarding more than 3-sigma outliers.
        >>> #  * for each stamp a iterative rejection mean and variance are
        >>> #    calculated with up to 3 iterations rejecting three or more
        >>> #    sigma outliers.
        >>> stamp_statistics_config = dict(fraction=0.5,
        >>>                                detrend_order=2,
        >>>                                average='mean',
        >>>                                outlier_threshold=3.0,
        >>>                                max_iter=3)

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
        >>>                            cloudy_night_threshold=5e3,
        >>>                            cloudy_frame_threshold=2.0,
        >>>                            min_high_mean=2.5e4,
        >>>                            max_low_mean=1.5e4)

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
        >>> #  * The resulting image is scaled to have a mean of 1 (no
        >>> #    configuration for that).
        >>> large_scale_smoothing_config = dict(shrink_factor=4,
        >>>                                     box_filter='median',
        >>>                                     box_half_size=6,
        >>>                                     spline_order=3,
        >>>                                     outlier_threshold=5.0,
        >>>                                     max_iter=1)

        >>> #Configuration for smoothnig for the purposes of checking for clouds.
        >>> #After smoothing to the master large scale structure:
        >>> #  * extract a central stamp is extracted from each flat covering
        >>> #    3/4 of the frame along each dimension
        >>> #  * shrink the fractional deviation of that stamp from the master
        >>> #    by a factor of 4 in each dimension
        >>> #  * smooth by median box-filtering with half size of 4 shrunk
        >>> #    pixels
        >>> #  * Perform at most 1 rejection re-smoothing iteration rejecting
        >>> #    outliers of more than 3-sigma.
        >>> #  * zoom the frame back out by a factor of 4 in each dimension
        >>> #    (same factor as shrinking, no separater config), using
        >>> #    bi-quadratic interpolation.
        >>> cloud_check_smoothing_config = dict(shrink_factor=4,
        >>>                                     box_filter='median',
        >>>                                     box_half_size=4,
        >>>                                     spline_order=None,
        >>>                                     outlier_threshold=3.0,
        >>>                                     max_iter=1)

        >>> #Create an object for stacking calibrated flat frames to master
        >>> #flats. In addition to the stamp-based rejections:
        >>> #  * reject flats that point within 40 arcsec of each other on the sky.
        >>> #  * if the smoothed cloud-check image contains pixels with absolute
        >>> #    value > 5% the frame is discarded as cloudy.
        >>> make_master_flat = MasterFlatMaker(
        >>>     min_pointing_offset=40.0,
        >>>     large_scale_deviation_threshold=0.05,
        >>>     stamp_statistics_config=stamp_statistics_config,
        >>>     stamp_select_config=stamp_select_config,
        >>>     large_scale_smoothing_config=large_scale_smoothing_config,
        >>>     cloud_check_smoothing_config=cloud_check_smoothing_config
        >>> )

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

    @staticmethod
    def _configure_smoothing(smoothing_config,
                             *,
                             shrink_factor=None,
                             box_filter=None,
                             box_half_size=None,
                             spline_order=None,
                             outlier_threshold=None,
                             max_iter=None):
        """
        Verify that a set of smoothing args make sense and update configuration.

        Args:
            smoothing_config:    The dictionary with the smoothing configuration
                to update.

            shrink_factor:    The factor by which to shrink the frame in each
                dimension before smoothing.

            box_filter:    The averaging to use for the box-filter smoothing.
                Should be either mean or median.

            box_half_size:    The half-size in pixels of the box-filter.

            spline_order:    After box filtering smooth using this order spline.

            outlier_threshold:    Pixels that are outliers of more than this
                from the corresponding smoothed value are discarded and
                smoothing is repeated.

            max_iter:    The maximum number of rejection/smoothing iterations
                to perform.

        Returns:
            None
        """

        if shrink_factor is not None:
            assert isinstance(shrink_factor, int)
            smoothing_config['shrink_factor'] = shrink_factor

        if box_filter is not None:
            assert box_filter in ['mean', 'median']
            smoothing_config['box_filter'] = box_filter

        if box_half_size is not None:
            assert isinstance(box_half_size, int)
            smoothing_config['box_half_size'] = box_half_size

        if spline_order is not None:
            assert isinstance(spline_order, int)
            smoothing_config['spline_order'] = spline_order

        if outlier_threshold is not None:
            assert isinstance(outlier_threshold, (float, int))
            smoothing_config['outlier_threshold'] = outlier_threshold

        if max_iter is not None:
            assert isinstance(max_iter, int)
            smoothing_config['max_iter'] = max_iter

    def __init__(self,
                 *,
                 stacking_config=dict(),
                 min_pointing_offset=None,
                 large_scale_deviation_threshold=None,
                 stamp_statistics_config=None,
                 stamp_select_config=None,
                 large_scale_smoothing_config=None,
                 cloud_check_smoothing_config=None):
        """
        Create object for creating master flats out of calibrated flat frames.

        Args:
            stacking_config:    The arguments to pass to MasterMake.__init__
                configuring how stacking of the final set of selected and
                prepared frames is perfcormed.

            min_pointing_offset:    The minimum distance between individual flat
                pointings required for stellar PSFs not to overlap.

            large_scale_deviation_threshold:    The maxmimu allowed fractional
                deviation from the largel scale structure after smoothing before
                a frame is declared cloudy.

            stamp_statistics_config:    A dictionary mith arguments to pass
                to configure_stamp_statistics.

            stamp_select_cofig:    A dictionary with arguments to pass
                to configure_stamp_selection.

            large_scale_smoothing_config:    A dictionary with arguments to pass
                to configure_large_scale_smoothing.

            cloud_check_smoothing_config:    A dictionary with arguments to pass
                to configure_cloud_check_smoothing.

        Returns:
            None
        """

        super().__init__(**stacking_config)

        self.min_pointing_offset = min_pointing_offset
        self.large_scale_deviation_threshold = large_scale_deviation_threshold
        self.stamp_statistics_config = dict()
        self.stamp_select_config = dict()
        self.large_scale_smoothing_config = dict()
        self.cloud_check_smoothing_config = dict()

        if stamp_statistics_config is not None:
            self.configure_stamp_statistics(**stamp_statistics_config)

        if stamp_select_config is not None:
            self.configure_stamp_selection(**stamp_select_config)

        if large_scale_smoothing_config is not None:
            self.configure_large_scale_smoothing(**large_scale_smoothing_config)

        if cloud_check_smoothing_config is not None:
            self.configure_cloud_check_smoothing(**cloud_check_smoothing_config)

    def configure_stamp_statistics(self,
                                   *,
                                   fraction=None,
                                   detrend_order=None,
                                   outlier_threshold=None,
                                   max_iter=None,
                                   average=None):
        """
        Configure extraction of stamp satistics for rejection & high/low split.

        Any arguments left as None are not updated.

        Args:
            fraction:    The fraction of the frame size that is included in the
                stamp along each dimension (i.e. fraction=0.5 means 1/4 of all
                frame pixels will be incruded in the stamp).

            detrend_order:    The order of the bi-polynomial used for
                de-trending the stamps before extracting statistics

            outlier_threshold:    The threshold in units of RMS deviation from
                the average above which pixels are considered outliers from the
                de-trending function and discarded from its fit.

            max_iter:    The maximum number of fit/reject iterations to perform
                before declaring the de-trending function final.

            average:    How to compute the average. Should be either 'mean'
                or 'median'.

        Returns:
            None
        """

        if fraction is not None:
            assert isinstance(fraction, (int, float))
            self.stamp_statistics_config['fraction'] = fraction

        if detrend_order is not None:
            assert isinstance(detrend_order, int)
            self.stamp_statistics_config['detrend_order'] = detrend_order

        if outlier_threshold is not None:
            assert isinstance(outlier_threshold, (int, float))
            self.stamp_statistics_config['outlier_threshold'] = (
                outlier_threshold
            )

        if max_iter is not None:
            assert isinstance(max_iter, int)
            self.stamp_statistics_config['max_iter'] = max_iter

        if average is not None:
            assert average in ['mean', 'median']
            self.stamp_statistics_config.average = average

    def configure_stamp_selection(self,
                                  *,
                                  max_saturated_fraction=None,
                                  cloudy_night_threshold=None,
                                  cloudy_frame_threshold=None,
                                  min_high_mean=None,
                                  max_low_mean=None):
        """
        Configure stamp-based frame selection and high/low split.

        Args:
            max_saturated_fraction:    The maximum fraction of stamp pixels
                allowed to be saturated before discarding the frame.

            cloudy_night_threshold:    The maximum residual of the variance vs
                mean quadratic fit before a night is declared cloudy.

            cloudy_frame_threshold:    The maximum deviation of an individual
                frame's variance vs mean from the var(mean) quadratic fit before
                the frame is declared cloudy.

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
            self.stamp_select_config['max_saturated_fraction'] = (
                max_saturated_fraction
            )

        if cloudy_night_threshold is not None:
            assert isinstance(cloudy_night_threshold, (int, float))
            self.stamp_select_config['cloudy_night_threshold'] = (
                cloudy_night_threshold
            )

        if cloudy_frame_threshold is not None:
            assert isinstance(cloudy_frame_threshold, (int, float))
            self.stamp_select_config['cloudy_frame_threshold'] = (
                cloudy_frame_threshold
            )

        if min_high_mean is not None:
            assert isinstance(min_high_mean, (int, float))
            self.stamp_select_config['min_high_mean'] = min_high_mean

        if max_low_mean is not None:
            assert isinstance(max_low_mean, (int, float))
            self.stamp_select_config['max_low_mean'] = max_low_mean

    def configure_large_scale_smoothing(self, **kwargs):
        """
        Configure the smoothnig for matching large scale structure to master.

        Args:
            See _configure_smoothing.

        Returns:
            None
        """

        self._configure_smoothing(self.large_scale_smoothing_config, **kwargs)

    def configure_cloud_check_smoothing(self, **kwargs):
        """
        Configure the smoothnig for final cloud check.

        Args:
            See _configure_smoothing.

        Returns:
            None
        """

        self._configure_smoothing(self.cloud_check_smoothing_config, **kwargs)
