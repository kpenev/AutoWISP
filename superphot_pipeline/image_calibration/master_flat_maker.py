"""Define classes for creating master flat frames."""

import numpy

from astropy.io import fits

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
        >>> stamp_config = dict(fraction=0.5,
        >>>                     detrend_order=2,
        >>>                     average='mean',
        >>>                     outlier_threshold=3.0,
        >>>                     max_iter=3)

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
        >>>     stamp_config=stamp_config,
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
