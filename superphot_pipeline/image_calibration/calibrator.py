"""Define a class (Calibrator) for low-level image calibration."""

from hashlib import sha1
from astropy.io import fits

import numpy

from superphot_pipeline.image_calibration.mask_utilities import\
    combine_masks,\
    get_saturation_mask,\
    mask_flags
from superphot_pipeline.image_calibration.fits_util import\
    create_result

from superphot_pipeline.fits_utilities import\
    read_image_components
from superphot_pipeline.pipeline_exceptions import\
    OutsideImageError,\
    ImageMismatchError
from superphot_pipeline import Processor

from superphot_pipeline.image_calibration.mask_utilities import\
    git_id as mask_utilities_git_id
from superphot_pipeline.image_calibration.overscan_methods import\
    git_id as overscan_methods_git_id
from superphot_pipeline.image_calibration.fits_util import get_raw_header

git_id = '$Id$'

class Calibrator(Processor):
    #pylint: disable=anomalous-backslash-in-string
    #Triggers on doxygen commands.
    """
    Provide basic calibration operations (bias/dark/flat) fully tracking noise.

    Public attributes, set through :func:`__init__`, provide defaults for
    calibration parameters that can be overwritten on a one-time basis for
    each frame being calibrated by passing arguments to :func:`__call__`.

    Attributes:
        configuration:    Defines the following calibration parameters:
            raw_hdu:    Which hdu of the raw frame contains the image to
                calibrate. Note that the header is always taken from the first
                non-trivial hdu. For compressed frames, this should never be
                zero.

            split_channels:    If this is a color image (i.e. staggering 4 color
                channels), this argument should be a dictionary with keys the
                names to use for the color channels and values specifying slices
                in the image that isolate the pixels of each channel.

            saturation_threshold:    The value (in ADU) above which the pixel is
                considered saturated.

            overscan:   A dictionary containing:

                * areas:    The areas in the raw image to use for overscan
                  corrections.  The format is::

                        [
                            dict(xmin = <int>,
                                 xmax = <int>,
                                 ymin = <int>,
                                 ymax = <int>),
                            ...
                        ]

                * method:    See :class:`overscan_methods.Base`

            image_area:    The area in the raw image which actually contains the
                useful image of the night sky. The dimensions must match the
                dimensions of the masters to apply an of the overscan correction
                retured by overscan_method. The format is::

                    dict(xmin = <int>, xmax = <int>, ymin = <int>, ymax = <int>)

            gain:    The gain to assume for the input image (electrons/ADU).
                Only useful when estimating errors and could be used by the
                overscan_method.

            fnum:    Expression involving header keywords (with "-" replaced by
                "_") or `RAWFNAME` which takes the raw frame name without
                extension or path and returning an integer to use as a frame
                number (should be unique for each image).

            calibrated_fname:    The filename under which to save the craeted
                image. Treated as a format string that may involve replacement
                fields from the resulting header, including {CLRCHNL}, which
                gets replaced by the color channel if channel splitting is
                performed, as well as {RAWFNAME}, which gets replaced by the
                file name of the input FITS file with all directories and
                `.fits` or `.fits.fz` extension stripped and {FNUM} which gets
                replaced by the frame number (see above).

        master_bias:    A dictionary containing:

            * filename: The filename of the master bias frame to use in
              subsequent calibrations (if not overwritten).

            * correction:    The correction to apply (if not overwritten).

            * variance:    An estimate of the variance of the ``correction``
              entry.

            * mask:    The bitmask the pixel indicating the combination of
              flags raised for each pixel. The individual flages are defined
              by :data:`mask_utilities.mask_flags`.

        master_dark:    Analogous to :attr:`master_bias` but contaning the
            information about the default master dark.

        master_flat:    Analogous to :attr:`master_bias` but contaning the
            information about the default master flat.

        masks:    A dictionary contaning:

            * filenames: A list of the files from which this mask was
              constructed.

            * image: The combined mask image (bitwise OR) of all masks in
              ``filenames``.

        module_git_ids:    A collection of Git Id values (sha1 checksums of the
            git blobs) for each module used by the calibration. Those get added
            to the header to ensure reprobducability.

    Examples:

        >>> from superphot_pipeline.image_calibration import Calibrator,\
        >>>     overscan_methods
        >>>
        >>> #Create a calibrator callable instance
        >>> calibrate = Calibrator(
        >>>     #The first 20 lines of the image are overscan area and overscan
        >>>     #is applied by subtracting the median of all pixels in the
        >>>     #overscan area from the remaining image.
        >>>     overscans=dict(areas=[dict(xmin=0, xmax=4096, ymin=0, ymax=20)],
        >>>                    method=overscan_methods.Median())
        >>>
        >>>     #The master bias frame to use.
        >>>     master_bias='masters/master_bias1.fits',
        >>>
        >>>     #The gain (electrons/ADU) to assume for the input images.
        >>>     gain=16.0,
        >>>
        >>>     #The area within the raw frame containing the image:
        >>>     image_area=dict(xmin=0, xmax=4096, ymin=20, ymax=4116)
        >>> )
        >>>
        >>> #Specify a master dark after construction.
        >>> calibrate.set_masters(dark='masters/master_dark3.fits')
        >>>
        >>> #Calibrate an image called 'raw1.fits', producing (or overwriting a
        >>> #calibrated file called 'calib1.fits' using the previously specified
        >>> #calibration parameters. Note that no flat correction is going to be
        >>> #applied, since a master flat was never specified.
        >>> calibrate(raw='raw1.fits', calibrated='calib1.fits')
        >>>
        >>> #Calibrate an image, changing the gain assumed for this image only
        >>> #and disabling overscan correction for this image only.
        >>> calibrate(raw='raw2.fits',
        >>>           calibrated='calib2.fits',
        >>>           gain=8.0,
        >>>           overscans=None)
    """
    #pylint: enable=anomalous-backslash-in-string

    module_git_ids = {'calibrator': git_id,
                      'overscan_methods': overscan_methods_git_id,
                      'mask_utilities': mask_utilities_git_id}

    default_configuration = {
        'overscans': {'areas': None, 'method': None},
        'image_area': None,
        'bias_level_adu': 0.0,
        'gain': 1.0,
        'leak_directions': [],
        'split_channels': False,
        'compress_calibrated': 16,
        'allow_overwrite': False
    }

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

              * Image or overscan areas have inverted boundaries, e.g.
                xmin > xmax

              * gain is not a finite positive number

              * leak_directions is not an iterable of 2-element iterables
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

            for direction, resolution in zip(['y', 'x'], raw_image.shape):
                area_min = area[direction + 'min']
                area_max = area[direction + 'max']

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
                if calib_params[master_type] is None:
                    continue
                for master_component in ['correction', 'variance', 'mask']:
                    master_shape = calib_params[
                        master_type
                    ][
                        master_component
                    ].shape

                    if master_shape != expected_master_shape:
                        raise ImageMismatchError(
                            'Master ' + master_type
                            +
                            ' ' + master_component
                            +
                            (
                                ' shape '
                                f'({master_shape[0]:d}x{master_shape[0]:d}) '
                                'does not match the image area specified during'
                                ' calibration '
                                f"({calib_params['image_area']['xmin']:d} "
                                '< x < '
                                f"{calib_params['image_area']['xmax']:d}, "
                                f"{calib_params['image_area']['ymin']:d} "
                                '< y < '
                                f"{calib_params['image_area']['ymax']:d})."
                            )
                        )


        if calib_params['image_area'] is None:
            calib_params['image_area'] = {
                'xmin': 0,
                'ymin': 0,
                'xmax': raw_image.shape[1],
                'ymax': raw_image.shape[0]
            }
        else:
            check_area(calib_params['image_area'], 'image_area')
        if calib_params['overscans']['areas'] is not None:
            for overscan_area in calib_params['overscans']['areas']:
                check_area(overscan_area, 'overscan')

        check_master_resolution()
        if (
                calib_params['gain'] <= 0
                or
                not numpy.isfinite(calib_params['gain'])
        ):
            raise ValueError('Invalid gain specified during calibration: '
                             +
                             repr(calib_params['gain']))

        try:
            for x_offset, y_offset in calib_params['leak_directions']:
                if (
                        not isinstance(x_offset, int)
                        or
                        not isinstance(y_offset, int)
                ):
                    raise ValueError(
                        f'Invalid leak direction: ({x_offset:s}, {y_offset:s})'
                    )
        except Exception as leak_problem:
            raise ValueError(
                'Malformatted list of leak directions: '
                +
                repr(calib_params['leak_directions'])
            ) from leak_problem

    @staticmethod
    def _calib_mask_from_master(mask):
        """
        Overwrite a master mask with what should be used for a calibrated frame.

        Args:
            mask:    The mask of the master frame. On exit, gets overwritten by
                the mask with which the calibrated image mask should be combined
                to account for imperfections in the master.

        Returns:
            None
        """

        for flag in ['OVERSATURATED', 'LEAKED']:
            match_flag = numpy.bitwise_and(
                mask,
                mask_flags[flag]
            ).astype(bool)
            mask[match_flag] = numpy.bitwise_and(
                mask[match_flag],
                numpy.bitwise_not(mask_flags[flag])
            )
            mask[match_flag] = numpy.bitwise_or(
                mask[match_flag],
                mask_flags['FAULT']
            )

    #pylint: disable=anomalous-backslash-in-string
    #Triggers on doxygen commands.
    @staticmethod
    def _document_in_header(calibration_params, header):
        """
        Update the given header to document how the calibration was done.

        The following keywords are added or overwritten::

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

        In addition, the overscan method describes itself in the header in
        any way it sees fit.

        Args:
            calibration_params:    The parameters used when calibrating (see
                calibration_params argument to :meth:`__call__`\ .

            header:    The header to update with the
                calibration information.

        Returns:
            None
        """

        for master_type in 'bias', 'dark', 'flat':
            if calibration_params[master_type] is not None:
                header['M' + master_type.upper() + 'FNM'] = (
                    calibration_params[master_type]['filename'],
                    'Master ' + master_type + ' frame applied'
                )
                with open(calibration_params[master_type]['filename'],
                          'rb') as master:
                    hasher = sha1()
                    hasher.update(master.read())
                    header['M' + master_type.upper() + 'SHA'] = (
                        hasher.hexdigest(),
                        'SHA-1 checksum of the master ' + master_type
                    )

        area_pattern = '%(xmin)d < x < %(xmax)d, %(ymin)d < y < %(ymax)d'

        if (
                calibration_params['overscans']['method'] is not None
                and
                calibration_params['overscans']['areas'] is not None
        ):
            for overscan_id, overscan_area in \
                    enumerate(calibration_params['overscans']['areas']):
                header[f'OVRSCN{overscan_id:02d}'] = (
                    area_pattern % overscan_area,
                    'Overscan area #' + str(overscan_id)
                )

            calibration_params['overscans']['method'].document_in_fits_header(
                header
            )

        header['IMAGAREA'] = (
            area_pattern % calibration_params['image_area'],
            'Image region in raw frame'
        )

        header['CALBGAIN'] = (calibration_params['gain'],
                              'Electrons/ADU assumed during calib')

        for module_name, module_git_id in Calibrator.module_git_ids.items():
            assert module_git_id[:4] == '$Id:'
            assert module_git_id[-1] == '$'
            header['CALGITID'] = (
                module_name + ':' + module_git_id[4:-1].strip(),
                'Git ID of calib module.'
            )
    #pylint: enable=anomalous-backslash-in-string


    def _get_calibration_params(self, overwrite_params, raw_image):
        """Set all calibration parameters following overwrite rules."""

        calibration_params = super().__call__(**overwrite_params)
        calibration_params['combine_headers'] = (
            calibration_params['raw_hdu'] is not None
        )
        if calibration_params['raw_hdu'] is None:
            calibration_params['raw_hdu'] = (
                1 if raw_image[0].header['NAXIS'] == 0 else 0
            )

        if calibration_params['gain'] is None:
            calibration_params['gain'] = float(
                raw_image[calibration_params['raw_hdu']].header['GAIN']
            )

        for master_type in ['bias', 'dark', 'flat']:
            if master_type in calibration_params:
                image, error, mask = read_image_components(
                    calibration_params[master_type]
                )[:3]
                self._calib_mask_from_master(mask)
                calibration_params[master_type] = {
                    'filename': calibration_params[master_type],
                    'correction': image,
                    'variance': numpy.square(error),
                    'mask': mask
                }
            else:
                calibration_params[master_type] = getattr(
                    self,
                    'master_' + master_type
                )

        if 'masks' in calibration_params:
            if isinstance(calibration_params['masks'], str):
                calibration_params['masks'] = {
                    'filenames': [calibration_params['masks']],
                    'image': read_image_components(
                        calibration_params['masks']
                    )[2]
                }
            else:
                calibration_params['masks'] = {
                    'filenames': calibration_params['masks'],
                    'image': combine_masks(calibration_params['masks'])[2]
                }
        else:
            calibration_params['masks'] = self.masks
        return calibration_params


    def __init__(self,
                 *,
                 saturation_threshold,
                 **configuration):
        """
        Create a calibrator and define some default calibration parameters.

        Args:
            Any configuration parameters users wish to change from their default
            values (see class description for details).

            masters: See set_masters.

        Returns:
            None
        """

        self.set_masters(
            **{master_type: configuration.pop('master_' + master_type, None)
               for master_type in ['bias', 'dark', 'flat']}
        )
        configuration['saturation_threshold'] = saturation_threshold
        super().__init__(**configuration)

    def set_masters(self, *, bias=None, dark=None, flat=None, masks=None):
        """
        Define the default masterts to use for calibrations.

        Args:
            bias:    The filename of the master bias to use.

            dark:    The filename of the master dark to use.

            flat:    The filename of the master flat to use.

            masks:    Either a single filename or a list of the mask files
                to use.

        Returns:
            None
        """

        if masks is not None:
            if isinstance(masks, str):
                self.masks = {
                    'filenames': [masks],
                    'image': read_image_components(masks)[2]
                }
            else:
                self.masks = {'filenames': masks, 'image': combine_masks(masks)}
        else:
            self.masks = None

        for master_type, master_fname in [('bias', bias),
                                          ('dark', dark),
                                          ('flat', flat)]:
            if master_fname is not None:
                image, error, mask = read_image_components(master_fname)[:3]
                self._calib_mask_from_master(mask)
                setattr(
                    self,
                    'master_' + master_type,
                    {
                        'filename': master_fname,
                        'correction': image,
                        'variance': numpy.square(error),
                        'mask': mask
                    }
                )
            else:
                setattr(self, 'master_' + master_type, None)

    def __call__(self,
                 raw,
                 **calibration_params):
        r"""
        Calibrate the raw frame, save result to calibrated.

        Args:
            raw:    The filename of the raw frame to calibrate.

            calibration_params:    Keyword only arguments allowing one of the
                calibration parameters to be switched for this calibration only.

        Returns:
            None
        """

        def apply_subtractive_correction(correction,
                                         calibrated_images):
            """
            Subtract correction from calibrated_image & update variance & mask.

            Args:
                correction:    The correction to subtract from the image, same
                    format as the master_bias or master_dark class attributes.

                calibrated_images:    A list of the three images constituting
                    the reduced FITS file (image, error, mask).

            Returns:
                None
            """

            calibrated_images[0] -= correction['correction']
            calibrated_images[1] += correction['variance']
            if 'mask' in correction and correction['mask'] is not None:
                calibrated_images[2] = numpy.bitwise_or(calibrated_images[2],
                                                        correction['mask'])

        def apply_flat_correction(master_flat,
                                  calibrated_images):
            """
            Apply flat-field correction to calibrated image and update variance.

            Args:
                master_flat:    The master flat to divide the image by, same
                    format as the `self.master_flat` attribute.

                calibrated_images:    A list of the three images constituting
                    the reduced FITS file (image, error, mask).

            Returns:
                None
            """

            calibrated_images[0] /= master_flat['correction']
            calibrated_images[1] = (
                calibrated_images[1]
                /
                master_flat['correction']**2
                +
                master_flat['variance']
                *
                calibrated_images[0]**2
                /
                master_flat['correction']**4
            )
            calibrated_images[2] = numpy.bitwise_or(calibrated_images[2],
                                                    master_flat['mask'])

        with fits.open(raw, 'readonly') as raw_image:
            calibration_params = self._get_calibration_params(
                calibration_params,
                raw_image
            )

            #pylint: disable=no-member
            #pylint false positive.
            self.check_calib_params(
                raw_image[calibration_params['raw_hdu']].data,
                calibration_params
            )

            trimmed_image = raw_image[calibration_params['raw_hdu']].data[
                calibration_params['image_area']['ymin']
                :
                calibration_params['image_area']['ymax']
                ,
                calibration_params['image_area']['xmin']
                :
                calibration_params['image_area']['xmax'],
            ].astype('float64') - calibration_params['bias_level_adu']
            calibrated_images = [
                trimmed_image,
                numpy.zeros(shape=trimmed_image.shape),
                get_saturation_mask(trimmed_image,
                                    calibration_params['saturation_threshold'],
                                    calibration_params['leak_directions'])
            ]

            #pylint: enable=no-member

            if (
                    calibration_params['overscans']['method'] is not None
                    and
                    calibration_params['overscans']['areas'] is not None
            ):
                apply_subtractive_correction(
                    calibration_params['overscans']['method'](
                        raw_image,
                        calibration_params['overscans']['areas'],
                        calibration_params['image_area'],
                        calibration_params['gain']
                    ),
                    calibrated_images
                )

            for master_type in ['bias', 'dark']:
                if calibration_params[master_type] is not None:
                    apply_subtractive_correction(
                        calibration_params[master_type],
                        calibrated_images
                    )
                if master_type == 'bias':
                    calibrated_images[1] += (
                        numpy.abs(calibrated_images[0])
                        /
                        calibration_params['gain']
                        +
                        (
                            calibration_params['read_noise_electrons']
                            /
                            calibration_params['gain']
                        )**2

                    )

            if calibration_params['flat'] is not None:
                apply_flat_correction(calibration_params['flat'],
                                      calibrated_images)

            raw_header = get_raw_header(raw_image, calibration_params)

            self._document_in_header(calibration_params, raw_header)
            calibrated_images[1] = numpy.sqrt(calibrated_images[1])

            create_result(image_list=calibrated_images,
                          header=raw_header,
                          result_fname=calibration_params['calibrated_fname'],
                          split_channels=calibration_params['split_channels'],
                          compress=calibration_params['compress_calibrated'],
                          allow_overwrite=calibration_params['allow_overwrite'])
