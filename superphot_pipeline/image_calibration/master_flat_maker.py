"""Define classes for creating master flat frames."""

#TODO: pythonize

import numpy

from superphot_pipeline.image_calibration.master_maker import MasterMaker
from superphot_pipeline.image_utilities import read_image_components
from superphot_pipeline.image_calibration.mask_utilities import mask_flags
from superphot_pipeline.image_smoothing import\
    ImageSmoother
from superphot_pipeline.iterative_rejection_util import\
    iterative_rejection_average,\
    iterative_rej_polynomial_fit

class MasterFlatMaker(MasterMaker):
    """
    Specialize MasterMaker for making master flat frames.

    Attrs:
        min_pointing_offset:    The minimum offset in pointing between flats to
            avoid overlapping stars.

        stamp_statistics_config:    Dictionary configuring how stamps statistics
            for stamp-based selection are extracted from the frames.

        stamp_select_config:    Dictionary configuring how stamp-based selection
            is performed. See keyword only arguments of _check_central_stamp for
            details.

        large_scale_smoother:    An ImageSmoother instance applied  to the ratio
            of a frame to the reference large scale structure before applying it
            to the frame. See keyword only arguments of _smooth_image for
            details.

        cloud_check_smoother:    ImageSmoother instance used for cloud detection
            performed on the full flat frames after smoothing to the master
            large scale structure. See keyword only arguments of _smooth_image
            for details.

        large_scale_deviation_threshold:    The threshold on the cloud-check
            image for considering the frame cloudy (in addition to stamp-based
            cloud detection).

    Examples:

        >>> import scipy.ndimage.filters
        >>> from superphot_pipeline.image_smoothing import\
        >>>     PolynomialImageSmoother,\
        >>>     SplineImageSmoother,\
        >>>     ChainSmoother,\
        >>>     WrapFilterAsSmoother

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

        >>> #Configuration for smoothnig for the purposes of checking for clouds.
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
        >>>     large_scale_smoother=large_scale_smoother,
        >>>     cloud_check_smoother=cloud_check_smoother
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

    def _get_stamp_statistics(self, frame_list):
        """
        Get relevant information from the stamp of a single input flat frame.

        Args:
            frame_list:    The list of frames for which to extract stamp
                statistics.

        Returns:
            stamp_statistics:    A structure numpy array with fields called
                `'mean'`, `'variance'` and `'num_averaged'` with the obvious
                meanings.
        """

        stamp_statistics = numpy.empty(
            len(frame_list),
            dtype=[('mean', numpy.float),
                   ('variance', numpy.float),
                   ('num_averaged', numpy.int)]
        )
        for frame_index, fname in enumerate(frame_list):
            image, mask = read_image_components(fname,
                                                read_error=False,
                                                read_header=False)

            y_size = int(image.shape[0]
                         *
                         self.stamp_statistics_config['fraction'])
            x_size = int(image.shape[1]
                         *
                         self.stamp_statistics_config['fraction'])
            x_off = (image.shape[1] - x_size) // 2
            y_off = (image.shape[0] - y_size) // 2
            num_saturated = numpy.bitwise_and(
                mask[y_off : y_off + y_size, x_off : x_off + x_size],
                numpy.bitwise_or(
                    mask_flags['OVERSATURATED'],
                    numpy.bitwise_or(
                        mask_flags['LEAKED'],
                        mask_flags['SATURATED']
                    )
                )
            ).astype(bool).sum()

            if (
                    num_saturated
                    >
                    (
                        self.stamp_select_config['max_saturated_fraction']
                        *
                        (x_size * y_size)
                    )
            ):
                stamp_statistics[frame_index] = numpy.nan, numpy.nan, numpy.nan
            else:
                smooth_stamp = self.stamp_statistics_config['smoother'].detrend(
                    image[y_off : y_off + y_size, x_off : x_off + x_size]
                )
                stamp_statistics[frame_index] = iterative_rejection_average(
                    smooth_stamp.flatten(),
                    average_func=self.stamp_statistics_config['average'],
                    max_iter=self.stamp_statistics_config['max_iter'],
                    outlier_threshold=(
                        self.stamp_statistics_config['outlier_threshold']
                    ),
                    mangle_input=True
                )

        stamp_statistics['variance'] = (
            numpy.square(stamp_statistics['variance'])
            *
            (stamp_statistics['num_averaged'] - 1)
        )
        return stamp_statistics

    def __init__(self,
                 *,
                 stacking_config=dict(),
                 min_pointing_offset=None,
                 large_scale_deviation_threshold=None,
                 stamp_statistics_config=None,
                 stamp_select_config=None,
                 large_scale_smoother=None,
                 cloud_check_smoother=None):
        """
        Create object for creating master flats out of calibrated flat frames.

        Args:
            stacking_config:    The arguments to pass to MasterMaker.__init__()
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

            large_scale_smoother:    An ImageSmoother instance used when
                matching large scale structure of individual flats to master.

            cloud_check_smoother:    An ImageSmoother instance used when checkng
                the full frames for clouds (after stamps are checked).

        Returns:
            None
        """

        super().__init__(**stacking_config)

        self.min_pointing_offset = min_pointing_offset
        self.large_scale_deviation_threshold = large_scale_deviation_threshold
        self.stamp_statistics_config = dict()
        self.stamp_select_config = dict()
        self.large_scale_smoother = large_scale_smoother
        self.cloud_check_smoother = cloud_check_smoother

        if stamp_statistics_config is not None:
            self.configure_stamp_statistics(**stamp_statistics_config)

        if stamp_select_config is not None:
            self.configure_stamp_selection(**stamp_select_config)

    def configure_stamp_statistics(self,
                                   *,
                                   fraction=None,
                                   smoother=None,
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

            smoother:    An ImageSmoother instance used used for
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

        if smoother is not None:
            assert isinstance(smoother, ImageSmoother)
            self.stamp_statistics_config['smoother'] = smoother

        if outlier_threshold is not None:
            assert isinstance(outlier_threshold, (int, float))
            self.stamp_statistics_config['outlier_threshold'] = (
                outlier_threshold
            )

        if max_iter is not None:
            assert isinstance(max_iter, int)
            self.stamp_statistics_config['max_iter'] = max_iter

        if average is not None:
            assert average in [numpy.nanmean, numpy.nanmedian]
            self.stamp_statistics_config['average'] = average

    def configure_stamp_selection(self,
                                  *,
                                  max_saturated_fraction=None,
                                  var_mean_fit_threshold=None,
                                  var_mean_fit_iterations=None,
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
            self.stamp_select_config['max_saturated_fraction'] = (
                max_saturated_fraction
            )

        if var_mean_fit_threshold is not None:
            assert isinstance(var_mean_fit_threshold, (int, float))
            self.stamp_select_config['var_mean_fit_threshold'] = (
                var_mean_fit_threshold
            )

        if var_mean_fit_iterations is not None:
            assert (
                not numpy.isfinite(var_mean_fit_iterations)
                or
                isinstance(var_mean_fit_iterations, int)
            )
            self.stamp_select_config['var_mean_fit_iterations'] = (
                var_mean_fit_iterations
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

    def __call__(self,
                 frame_list,
                 high_master_fname,
                 low_master_fname,
                 *,
                 compress=True,
                 allow_overwrite=False,
                 **stacking_options):
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
                to superphot_pipeline.image_calibration.fits_util.create_result.

            stacking_options:    Keyword only arguments allowing overriding the
                stacking configuration specified at construction for this
                stack only.

        Reutrns:
            None
        """

        stamp_statistics = self._get_stamp_statistics(frame_list)

        if(
                self.stamp_select_config['cloudy_night_threshold'] is not None
                or
                self.stamp_select_config['cloudy_frame_threshold'] is not None
        ):
            fit_coef, residual, best_fit_variance = iterative_rej_polynomial_fit(
                x=stamp_statistics['mean'],
                y=stamp_statistics['variance'],
                order=2,
                outlier_threshold=self.stamp_select_config[
                    'var_mean_fit_threshold'
                ],
                max_iterations=self.stamp_select_config[
                    'var_mean_fit_iterations'
                ],
                return_predicted=True
            )

            print("Flat stamp pixel statistics:\n\t%50s|%10s|%10s|%10s"
                  %
                  ("frame", "mean", "std", "fitstd"))
            print("\t" + 92 * '_')
            for fname, stat, fitvar in zip(frame_list,
                                           stamp_statistics,
                                           best_fit_variance):
                print("\t%50s|%10g|%10g|%10g"
                      %
                      (fname,
                       stat['mean'],
                       stat['variance']**0.5,
                       fitvar**0.5))
            print("Best fit quadratic: %f + %f*m + %f*m^2; residue=%f"
                  %
                  (tuple(fit_coef) + (residual,)))

            if(
                    (
                        self.stamp_select_config['cloudy_night_threshold']
                        is not None
                    )
                    and
                    (
                        residual
                        >
                        self.stamp_select_config['cloudy_night_threshold']
                    )
            ):
                return

            if self.stamp_select_config['cloudy_frame_threshold'] is not None:
                cloudy_stamps = (
                    numpy.abs(stamp_statistics['variance'] - best_fit_variance)
                    >
                    self.stamp_select_config['cloudy_frame_threshold']
                )
