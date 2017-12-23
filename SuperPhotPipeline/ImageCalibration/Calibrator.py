"""Define a class (Calibrator) for low-level image calibration."""

from astropy.io import fits
from numpy import isfinite, sqrt

calibrator_sha = $Id$

class Calibrator :
    """
    Provide basic calibration operations (bias/dark/flat) fully tracking noise.

    Example:

        from SuperPhotPipeline.ImageCalibration import Calibrator,\
                                                       OverscanMethods

        #Create a calibrator callable instance
        calibrate = Calibrator(
            #The first 20 lines of the image are overscan area.
            overscans = [dict(xmin = 0, xmax = 4096, ymin = 0, ymax = 20)],

            #Overscan corrections should subtract the median of the values.
            overscan_method = OverscanMethos.median,

            #The master bias frame to use.
            master_bias = 'masters/master_bias1.fits',

            #The gain (electrons/ADU) to assume for the input images.
            gain = 16.0,

            #The area within the raw frame containing the image:
            image_area = dict(xmin = 0, xmax = 4096, ymin = 20, ymax = 4116
        )

        #Specify a master dark after construction.
        calibrate.set_masters(dark = 'masters/master_dark3.fits')

        #Calibrate an image called 'raw1.fits', producing (or overwriting a
        #calibrated file called 'calib1.fits' using the previously specified
        #calibration parameters. Note that no flat correction is going to be
        #applied, since a master flat was never specified.
        calibrate(raw = 'raw1.fits', calibrated = 'calib1.fits')

        #Calibrate an image, changing the gain assumed for this image only.
        calibrate(raw = 'raw2.fits', calibrated = 'calib2.fits', gain = 8.0)

    Attributes:
        master_bias: numpy 2-D array like
            The currently set default master bias to use if not overwritten when
            calling the calibrator.

        master_dark: numpy 2-D array like
            The currently set default master dark to use if not overwritten when
            calling the calibrator.

        master_flat: numpy 2-D array like
            The currently set default master flat to use if not overwritten when
            calling the calibrator.

        oversans: [dict(xmin = <int>, xmax = <int>, ymin = <int>, ymax = <int>),
                   ...]
            The areas in the raw image to use for overscan corrections.

        overscan_method: callable
            A callable which returns the overscan correction (a 2-D numpy array)
            to apply to the input image, and an estimate of its variance, given
            the image and the overscan area.

        image_area: dict(xmin = <int>, xmax = <int>, ymin = <int>, ymax = <int>)
            The area in the raw image which actually contains the useful image
            of the night sky. The dimensions must match the dimensions of the
            masters to apply an of the overscan correction retured by
            overscan_method.
    """

    def _check_calib_params(self, raw_image, calib_params) :
        """
        Check if calibration parameters are consistent mutually and with image.

        Raises an exception if some problem is detected.

        Problems checked for:
            - Overscan or image areas outside the image.
            - Any master resolution does not match image area.
            - Image or overscan areas have inverted boundaries, e.g. xmin > xmax
            - gain is not a finite positive number

        Args:
            raw_image:
                The raw image to calibrate (numpy 2-D array).
            calib_params:
                The current set of calibration parameters to apply.

        Returns: None
        """

        def check_area(area, area_name) :
            """
            Make sure the given area is within raw_image, raise excption if not.

            Args:
                area:
                    The area to check. Must have the same format as an overscan
                    entry or the image_area argument to __init__.

                area_name:
                    The name identifying the problematic area. Used if an
                    exception is raised.
            
            Returns: None
            """

            for direction, resolution in zip(['x', 'y'], raw_image.shape) :
                area_min, area_max = area[direction + 'min'], area[direction + 'max']
                if area_min > area_max :
                    raise ValueError(
                        area_name + ' area has '
                        +
                        direction + 'min(' + str(area_min) + ') > '
                        +
                        direction + 'max (' + str(area_max) + ')'
                    )
                if area_min < 0 :
                    raise OutsideImageError(
                        area_name + ' area (' + str(area) + ') '
                        +
                        'extends below ' + direction + ' = 0'
                    )
                if area_max > resolution :
                    raise OutsideImageError(
                        area_name + ' area (' + str(area) + ') '
                        +
                        'extends beyond the image ' + direction + ' resolution '
                        +
                        'of ' + str(resolution)
                    )
        def check_master_resolution() :
            """
            Raise exception if any master has resolution not that of image area.

            Args: None

            Returns: None
            """

            expected_master_shape = (
                (
                    calib_params['image_area']['ymax']
                    -
                    calib_params['image_area']['ymin']
                ),
                (
                    calib_params['image_area']['xmax']
                    -
                    calib_params['image_area']['xmin']
                )
            )
            for master_type in ['bias', 'dark', 'flat'] :
                for master_component in ['correction', 'variance'] :
                    master_shape = calib_params[
                        'master_' + master_type + master_component
                    ].shape
                    if master_shape != expected_master_shape :
                        raise ImageMismatchError(
                            'Master ' + master_type
                            +
                            ' ' + master_component
                            +
                            (
                                ' shape (%dx%d) does not match the image area '
                                'specified during calibration '
                                '(%d < x < %d, %d < y < %d).'
                            )
                            %
                            (master_shape + (calib_params['image_area']['xmin'],
                                             calib_params['image_area']['xmax'],
                                             calib_params['image_area']['ymin'],
                                             calib_params['image_area']['ymax']))
                        )


        check_area(calib_params['image_area'], image_area)
        if calib_params['overscans'] is not None :
            for overscan_area in calib_params['overscans'] :
                check_area(overscan_area, 'overscan')

        check_master_resolution()
        if gain < 0 or not isfinite(gain) :
            raise ValueError('Invalid gain specified during calibration: '
                             +
                             repr(gain))

    def __init__(self,
                 master_bias = None,
                 master_dark = None,
                 master_flat = None,
                 overscans = None,
                 overscan_method = None,
                 image_area = None,
                 gain = 1.0) :
        """
        Create a calibrator and define some default calibration parameters.

        Args:
            master_bias: 
                The name of the file containing the master bias frame to use. If
                None, no master bias correction is applied.

            master_dark: 
                The name of the file containing the master dark frame to use. If
                None, no master dark correction is applied.

            master_flat: 
                The name of the file containing the master flat frame to use. If
                None, no master flat correction is applied.

            overscans:
                List of the overscan areas to use. Each overscan area is
                dict(xmin = <int>, xmax = <int>, ymin = <int>, ymax = <int>).
               
            overscan_method:
                A callable that given an overscan area and an image returns the
                overscan correction to apply as a numpy 2-D array (or
                compatible). None or emply list disables overscan corrections).
                See OverscanMethods for functions that can be used for this
                argument.
            
            image_area:
                The area on the image which will be made part of the final
                image. The same format as an overscan region. If None, defaults
                to the entire image.

            gain:
                The gain to assume for the input image (electrons/ADU). Only
                useful when estimating errors and could be used by the
                overscan_method.

        Returns: None
        """

        self.set_masters(bias = master_bias,
                         dark = master_dark,
                         flat = master_flat)
        self.overscans = overscans
        self.overscan_method = overscan_method
        self.image_area = image_area
        self.gain = gain

    def set_masters(self, **kwargs) :
        """
        Define the default masterts to use for calibrations.

        Args:
            **kwargs:
                Keyword only arguments defining the master to use for each kind
                of calibratin step. Must be one of 'bias', 'dark' or 'flat'.
                Everything else is ignored.

        Returns: None
        """

        for master_type in ['bias', 'dark', 'flat'] :
            if master_type in kwargs :
                master_fname = kwargs[master_type]
                setattr(self, 'master_' + master_type, master_fname)
                if master_fname is None :
                    setattr(self, 'master_' + master_type + '_correction', None)
                    setattr(self, 'master_' + master_type + '_variance', None)
                else :
                    with fits.open(master_fname, mode = 'readonly') as image :
                        setattr(self, 'master_' + master_type + '_correction',
                                image[0].data)
                        setattr(self, 'master_' + master_type + '_variance',
                                image[1].data**2)

    def __call__(self,
                 raw,
                 calibrated,
                 **calibration_params) :
        """
        Calibrate the raw frame, save result to calibrated.

        Args:
           raw:
               The filename of the raw frame to calibrate.
            calibrated:
                The filename under which to save the calibrated frame.
            **calibration_params:
                Keyword only arguments allowing one of the calibration
                parameters to be switched for this calibration only. Should be
                one of the arguments of __init__, with the same meaning.

        Returns: None
        """

        def aplly_subtractive_correction(correction_image,
                                         correction_variance,
                                         calibrated_image,
                                         variance_image) :
            """
            Subtract a correction from calibrated_image, also update variance. 

            Args:
                correction_image:
                    The correction to subtract from the image.
                correction_variance:
                    An estimate of the variance of the correction_image entries.
                calibrated_image:
                    The image to subtract correction_image from.
                variance_image:
                    An estimate of the variance in the calibrated image. Also
                    updated accordingly.

            Returns: None
            """

            calibrated_image -= correction_image
            variance_image += correction_variance

        def apply_flat_correction(flat_image,
                                  flat_variance,
                                  calibrated_image,
                                  calibrated_variance) :
            """
            Apply flat-field correction to calibrated image and update variance.

            Args:
                flat_image:
                    The flat-field correction to divide the image by.
                correction_variance:
                    An estimate of the variance of the flat_image entries.
                calibrated_image:
                    The image to apply the correction to.
                variance_image:
                    An estimate of the variance in the calibrated image. Also
                    updated accordingly.

            Returns: None
            """

            calibrated_image /= flat_image
            variance_image = (
                variance_image / flat_image**2
                +
                flat_variance * calibrated_image**2 / flat_correction**4
            )

        def create_result() :
            """
            Create the calibarted FITS file documenting calibration in header.

            Args: None

            Returns: None
            """

            <++>

        for master_type in ['bias', 'dark', 'flat'] :
            if 'master_' + master_type in calibration_params :
                fname = calibration_params['master_' + master_type]
                with fits.open(fname, mode = 'readonly') as image :
                    calibration_params[
                        'master_' + master_type + '_correction'
                    ] = image[0].data
                    calibration_params[
                        'master_' + master_type + '_variance'
                    ] = image[1].data

        param_list = (
            [
                'master_' + master_type + suffix
                for master_type in 'bias', 'dark', 'flat'
                for suffix in ['_correction', '_variance', '']
            ]
            +
            ['overscans', 'overscan_method', 'image_area', 'gain']
        )
        for param in param_list :
            if param not in calibration_params :
                calibration_params[param] = getattr(self, param)

        with fits.open(raw, 'readonly') as raw_image :
            self._check_calib_params(raw_image[0].data, calibration_params)

            calibrated_image = raw_image[0].data[
                calibration_params['ymin'] : calibration_params['ymax'],
                calibration_params['xmin'] : calibration_params['xmax'],
            ]

            variance_image = calibrated_image / calibration_params['gain']

            if (
                    calibration_params['overscan_method'] is not None
                    and
                    calibration_params['overscans'] is not None
            ) :
                apply_subtractive_correction(
                    *calibration_params['overscan_method'](
                        raw_image,
                        calibration_params['overscans']
                    ),
                    calibrated_image,
                    variance_image
                )

            for mastert_type in ['master_bias', 'master_dark'] :
                if (
                        calibration_params[
                            master_type + '_correction'
                        ] is not None
                ) :
                    assert(calibration_params[master_type + '_variance']
                           is not None)
                    apply_subtractive_correction(
                        calibration_params[master_type + '_correction'],
                        calibration_params[master_type + '_variance']
                    )

            if calibration_params['master_flat_correction'] is not None :
                assert(calibration_params['master_flat_variance'] is not None)
                apply_flat_correction(calibration_params['master_flat_correction'],
                                      calibration_params['master_flat_variance'],
                                      calibrated_image,
                                      calibrated_variance)
