"""Define a class (Calibrator) for low-level image calibration."""

from hashlib import sha1
from numpy import isfinite
from astropy.io import fits

from superphot_pipeline.pipeline_exceptions import\
    OutsideImageError,\
    ImageMismatchError

calibrator_sha = '$Id$'

class Calibrator:
    """
    Provide basic calibration operations (bias/dark/flat) fully tracking noise.

    Attrs:

        Public attributes provide defaults for calibration parameters that can
        be overwritten on a one-time basis for each frame being calibrated by
        passing arguments to __call__.

        master_bias:    The filename of the master bias frame used in the
            calibrations.

        master_bias_correction:    numpy 2-D array like containing the currently
            set default master bias to use if not overwritten when calling the
            calibrator.

        master_bias_variance:    numpy 2-D array like, containing the variance
            estimate of the default master bias correction.

        master_dark:    The filename of the master dark frame used in the
            calibrations.

        master_dark_correction:    numpy 2-D array like containing the currently
            set default master dark to use if not overwritten when calling the
            calibrator.

        master_dark_variance:    numpy 2-D array like, containing the variance
            estimate of the default master dark correction.

        master_flat:    The filename of the master flat frame used in the
            calibrations.

        master_flat_correction:    numpy 2-D array like containing the currently
            set default master flat to use if not overwritten when calling the
            calibrator.

        master_flat_variance:    numpy 2-D array like, containing the variance
            estimate of the default master flat correction.

        oversans:    The areas in the raw image to use for overscan corrections.
            The format is

            [dict(xmin = <int>, xmax = <int>, ymin = <int>, ymax = <int>), ...]

        overscan_method:    See OverscanMethods.OverscanMethodBase.

        image_area:    The area in the raw image which actually contains the
            useful image of the night sky. The dimensions must match the
            dimensions of the masters to apply an of the overscan correction
            retured by overscan_method. The format is:

            dict(xmin = <int>, xmax = <int>, ymin = <int>, ymax = <int>)

        gain:
            The gain to assume for the input image (electrons/ADU). Only useful
            when estimating errors and could be used by the overscan_method.

    Examples:

        >>> from SuperPhotPipeline.ImageCalibration import Calibrator,\
        >>> OverscanMethods

        >>> #Create a calibrator callable instance
        >>> calibrate = Calibrator(
        >>>    #The first 20 lines of the image are overscan area.
        >>>    overscans = [dict(xmin = 0, xmax = 4096, ymin = 0, ymax = 20)],

        >>>    #Overscan corrections should subtract the median of the values.
        >>>    overscan_method = OverscanMethods.Median(),

        >>>    #The master bias frame to use.
        >>>    master_bias = 'masters/master_bias1.fits',

        >>>    #The gain (electrons/ADU) to assume for the input images.
        >>>    gain = 16.0,

        >>>    #The area within the raw frame containing the image:
        >>>    image_area = dict(xmin = 0, xmax = 4096, ymin = 20, ymax = 4116
        >>> )

        >>> #Specify a master dark after construction.
        >>> calibrate.set_masters(dark = 'masters/master_dark3.fits')

        >>> #Calibrate an image called 'raw1.fits', producing (or overwriting a
        >>> #calibrated file called 'calib1.fits' using the previously specified
        >>> #calibration parameters. Note that no flat correction is going to be
        >>> #applied, since a master flat was never specified.
        >>> calibrate(raw = 'raw1.fits', calibrated = 'calib1.fits')

        >>> #Calibrate an image, changing the gain assumed for this image only and
        >>> #disabling overscan correction for this image only.
        >>> calibrate(raw = 'raw2.fits',
        >>>          calibrated = 'calib2.fits',
        >>>          gain = 8.0,
        >>>          overscans = None)
    """

    @staticmethod
    def check_calib_params(raw_image, calib_params):
        """
        Check if calibration parameters are consistent mutually and with image.

        Raises an exception if some problem is detected.

        Args:
            raw_image: The raw image to calibrate (numpy 2-D array).

            calib_params: The current set of calibration parameters to apply.

        Returns:
            None

        Notes:
            Checks for:
              * Overscan or image areas outside the image.
              * Any master resolution does not match image area.
              * Image or overscan areas have inverted boundaries, e.g. xmin > xmax
              * gain is not a finite positive number


        """

        def check_area(area, area_name):
            """
            Make sure the given area is within raw_image, raise excption if not.

            Args:

                area:    The area to check. Must have the same format as an
                    overscan ntry or the image_area argument to __init__.

                area_name:    The name identifying the problematic area. Used if
                    an exception is raised.

            Returns: None
            """

            for direction, resolution in zip(['x', 'y'], raw_image.shape):
                area_min, area_max = area[direction + 'min'], area[direction + 'max']
                if area_min > area_max:
                    raise ValueError(
                        area_name + ' area has '
                        +
                        direction + 'min(' + str(area_min) + ') > '
                        +
                        direction + 'max (' + str(area_max) + ')'
                    )
                if area_min < 0:
                    raise OutsideImageError(
                        area_name + ' area (' + str(area) + ') '
                        +
                        'extends below ' + direction + ' = 0'
                    )
                if area_max > resolution:
                    raise OutsideImageError(
                        area_name + ' area (' + str(area) + ') '
                        +
                        'extends beyond the image ' + direction + ' resolution '
                        +
                        'of ' + str(resolution)
                    )
        def check_master_resolution():
            """
            Raise exception if any master has resolution not that of image area.

            Args:
                None

            Returns:
                None
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
            for master_type in ['bias', 'dark', 'flat']:
                for master_component in ['correction', 'variance']:
                    master_shape = calib_params[
                        'master_' + master_type + master_component
                    ].shape
                    if master_shape != expected_master_shape:
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


        check_area(calib_params['image_area'], 'image_area')
        if calib_params['overscans'] is not None:
            for overscan_area in calib_params['overscans']:
                check_area(overscan_area, 'overscan')

        check_master_resolution()
        if calib_params['gain'] < 0 or not isfinite(calib_params['gain']):
            raise ValueError('Invalid gain specified during calibration: '
                             +
                             repr(calib_params['gain']))

    #pylint: disable=anomalous-backslash-in-string
    #Triggers on doxygen commands.
    @staticmethod
    def _document_in_header(calibration_params, header):
        """
        Update the given header to document how the calibration was done.

        The following keywords are added or overwritten:

        \verbatim
            MBIASFNM: Filename of the master bias used
            MBIASSHA: Sha-1 checksum of the master bias frame used.
            MDARKFNM: Filename of the master dark used
            MDARKSHA: Sha-1 checksum of the master dark frame used.
            MFLATFNM: Filename of the master flat used
            MFLATSHA: Sha-1 checksum of the master flat frame used.
            OVRSCN%02d: The overscan regions get consecutive IDs and for
                each one, the id gets substituted in the keyword as given.
                The value describes the overscan region like:
                %(xmin)d < x < %(xmax)d, %(ymin)d < y < %(ymax)d with the
                sibustition dictionary given directly by the corresponding
                overscan area.
            IMAGAREA: '%(xmin)d < x < %(xmax)d, %(ymin)d < y < %(ymax)d'
                subsituted with the image are as specified during
                calibration.
            CALBGAIN: The gain assumed during calibration.
            CLIBSHA: Sha-1 chacksum of the Calibrator.py blob per Git.
        \endverbatim

        In addition, the overscan method describes itself in the header in
        any way it sees fit.

        Args:
            header:    The header to update with the
                calibration information.

        Returns:
            None
        """

        for master_type in 'bias', 'dark', 'flat':
            if calibration_params['master_' + master_type] is not None:
                header['M' + master_type.upper() + 'FNM'] = (
                    calibration_params['master_' + master_type],
                    'Master ' + master_type + ' frame applied'
                )
                with open(calibration_params['master_bias'], 'r') as master:
                    hasher = sha1()
                    hasher.update(master.read().encode('ascii'))
                    header['M' + master_type.upper() + 'SHA'] = (
                        hasher.hexdigest(),
                        'SHA-1 checksum of the master ' + master_type
                    )

        area_pattern = '%(xmin)d < x < %(xmax)d, %(ymin)d < y < %(ymax)d'
        for overscan_id, overscan_area in \
                enumerate(calibration_params['overscans']):
            header['OVRSCN%02d' % overscan_id] = (
                area_pattern % overscan_area,
                'Overscan area #' + str(overscan_id)
            )

        calibration_params['overscan_method'].document_in_fits_header(header)

        header['IMAGAREA'] = (
            area_pattern % calibration_params['image_area'],
            'Image region in raw frame'
        )

        header['CALIBGAIN'] = (calibration_params['gain'],
                               'Electrons/ADU assumed during calib')

        assert calibrator_sha[:4] == '$Id:'
        assert calibrator_sha[-1] == '$'
        header['CALIBSHA'] = (calibrator_sha[4:-1].strip(),
                              'Git Id of Calibrator.py used in calib')
    #pylint: enable=anomalous-backslash-in-string

    @staticmethod
    def _create_result(calibrated_image,
                       variance_image,
                       header,
                       calibrated_fname,
                       compressed):
        """
        Create the calibarted FITS file documenting calibration in header.

        Args:
            calibrated_image:    The calibrated image data to write to the
                output file.

            variance_image:    An estiamte of the variance of each pixel of
                the calibrated image. Saved as a second extension.

            header:    The header of the raw image. Everything gets copied
                ot the reduced image and further keywords are added to
                document the calibration.

            calibrated_fname:    The filename under which to save the
                craeted image.

            compressed:    Should the created image be compressed?

        Returns:
            None
        """

        if compressed:
            hdu_list = fits.HDUList([
                fits.CompImageHDU(calibrated_image, header),
                fits.CompImageHDU(variance_image)
            ])
        else:
            hdu_list = fits.HDUList([
                fits.PrimaryHDU(calibrated_image, header),
                fits.ImageHDU(variance_image)
            ])
        hdu_list.writeto(calibrated_fname)

    #pylint: disable=too-many-arguments
    #Makes sense to allow all calibration parameters to be specified.
    def __init__(self,
                 master_bias=None,
                 master_dark=None,
                 master_flat=None,
                 overscans=None,
                 overscan_method=None,
                 image_area=None,
                 gain=1.0):
        """
        Create a calibrator and define some default calibration parameters.

        Args:
            master_bias:    See same name attribute to Calibrator class.

            master_dark:    See same name attribute to Calibrator class.

            master_flat:    See same name attribute to Calibrator class.

            overscans:    See same name attribute to Calibrator class.

            overscan_method:    See same name attribute to Calibrator class.

            image_area:    See same name attribute to Calibrator class.

            gain:    See same name attribute to Calibrator class.

        Returns:
            None
        """

        self.set_masters(bias=master_bias,
                         dark=master_dark,
                         flat=master_flat)
        self.overscans = overscans
        self.overscan_method = overscan_method
        self.image_area = image_area
        self.gain = gain
    #pylint: enable=too-many-arguments

    def set_masters(self, **kwargs):
        """
        Define the default masterts to use for calibrations.

        Kwargs:
            kwargs:    Keyword only arguments defining the master to use for
                each kind of calibratin step. Must be one of 'bias', 'dark' or
                'flat'. Everything else is ignored.

        Returns:
            None
        """

        for master_type in ['bias', 'dark', 'flat']:
            if master_type in kwargs:
                master_fname = kwargs[master_type]
                setattr(self, 'master_' + master_type, master_fname)
                if master_fname is None:
                    setattr(self, 'master_' + master_type + '_correction', None)
                    setattr(self, 'master_' + master_type + '_variance', None)
                else:
                    with fits.open(master_fname, mode='readonly') as image:
                        #pylint: disable=no-member
                        #pylint false positive.
                        setattr(self, 'master_' + master_type + '_correction',
                                image[0].data)
                        setattr(self, 'master_' + master_type + '_variance',
                                image[1].data**2)
                        #pylint: enable=no-member

    def __call__(self,
                 raw,
                 calibrated,
                 compress_calibrated=True,
                 **calibration_params):
        """
        Calibrate the raw frame, save result to calibrated.

        Args:
            raw:    The filename of the raw frame to calibrate.

            calibrated:    The filename under which to save the
                calibrated frame.

            compress_calibrated:    Should the calibrated image be compressed.

        Kwargs:

            calibration_params:    Keyword only arguments allowing one of the
                calibration parameters to be switched for this calibration only.
                Should be one of the arguments of __init__, with the
                same meaning.

        Returns:
            None
        """

        def apply_subtractive_correction(correction_image,
                                         correction_variance,
                                         calibrated_image,
                                         variance_image):
            """
            Subtract a correction from calibrated_image, also update variance.

            Args:
                correction_image:    The correction to subtract from the image.

                correction_variance:    An estimate of the variance of the
                    correction_image entries.

                calibrated_image:    The image to subtract
                    correction_image from.

                variance_image:    An estimate of the variance in the calibrated
                    image. Also updated accordingly.

            Returns:
                None
            """

            calibrated_image -= correction_image
            variance_image += correction_variance

        def apply_flat_correction(flat_image,
                                  flat_variance,
                                  calibrated_image,
                                  calibrated_variance):
            """
            Apply flat-field correction to calibrated image and update variance.

            Args:
                flat_image:    The flat-field correction to divide the image by.

                correction_variance:    An estimate of the variance of the
                    flat_image entries.

                calibrated_image:    The image to apply the correction to.

                variance_image:    An estimate of the variance in the calibrated
                    image. Also updated accordingly.

            Returns:
                None
            """

            calibrated_image /= flat_image
            calibrated_variance = (
                calibrated_variance / flat_image**2
                +
                flat_variance * calibrated_image**2 / flat_image**4
            )

        for master_type in ['bias', 'dark', 'flat']:
            if 'master_' + master_type in calibration_params:
                with fits.open(calibration_params['master_' + master_type],
                               mode='readonly') as image:
                    #pylint: disable=no-member
                    #pylint false positive.
                    calibration_params[
                        'master_' + master_type + '_correction'
                    ] = image[0].data
                    calibration_params[
                        'master_' + master_type + '_variance'
                    ] = image[1].data
                    #pylint: enable=no-member

        for param in (
                [
                    'master_' + master_type + suffix
                    for master_type in ['bias', 'dark', 'flat']
                    for suffix in ['_correction', '_variance', '']
                ]
                +
                ['overscans', 'overscan_method', 'image_area', 'gain']
        ):
            if param not in calibration_params:
                calibration_params[param] = getattr(self, param)

        with fits.open(raw, 'readonly') as raw_image:
            #pylint: disable=no-member
            #pylint false positive.
            self.check_calib_params(raw_image[0].data, calibration_params)

            calibrated_image = raw_image[0].data[
                calibration_params['ymin']: calibration_params['ymax'],
                calibration_params['xmin']: calibration_params['xmax'],
            ]
            #pylint: enable=no-member

            variance_image = calibrated_image / calibration_params['gain']

            if (
                    calibration_params['overscan_method'] is not None
                    and
                    calibration_params['overscans'] is not None
            ):
                apply_subtractive_correction(
                    *calibration_params['overscan_method'](
                        raw_image,
                        calibration_params['overscans'],
                        calibration_params['image_area'],
                        calibration_params['gain']
                    ),
                    calibrated_image,
                    variance_image
                )

            for master_type in ['master_bias', 'master_dark']:
                if (
                        calibration_params[
                            master_type + '_correction'
                        ] is not None
                ):
                    assert(calibration_params[master_type + '_variance']
                           is not None)
                    apply_subtractive_correction(
                        calibration_params[master_type + '_correction'],
                        calibration_params[master_type + '_variance'],
                        calibrated_image,
                        variance_image
                    )

            if calibration_params['master_flat_correction'] is not None:
                assert calibration_params['master_flat_variance'] is not None
                apply_flat_correction(calibration_params['master_flat_correction'],
                                      calibration_params['master_flat_variance'],
                                      calibrated_image,
                                      variance_image)

            #pylint: disable=no-member
            #pylint false positive.
            raw_header = raw_image[0].header
            #pylint: enable=no-member
            self._document_in_header(calibration_params, raw_header)
            self._create_result(calibrated_image,
                                variance_image,
                                raw_header,
                                calibrated,
                                compress_calibrated)
