Low Level Master Stack Interface {#LowLevelMasterStack_page}
================================

Given a user-specified set of calibrated FITS frames produced as described in
\ref LowLevelImageCalibration_page, this module stacks them to create master
frames. The stacking procedure and example usage for each master types is as
follows:

Master Bias/Dark
----------------

The procedure for generating one of these two types of masters is that each
pixels in the output image in generated from the corresponding pixels in the
individual input images by iterating between the following two steps:
    1. finding the median
    2. rejecting values differing from the median by more than some specified
       threshold times the root mean square deviation from the median

Master Flat
-----------

Since flat frames can be images of the sky or of perhaps a dome with changing
illumination, special care must be taken to compensate for changes in the large
scale structure from one calibrated flat frame to the other. Further, with sky
frames there is always the possibility of clouds, so there needs to be an
automated procedure for detecting clouds in individual frames and discarding
them, or of detecting cloudy flat collections and refusing to generate a master
flat altogether. The procedure used by HATSouth is as follows:
    1. For each flat a mean and standard devation are calculated:

        1.1. A central stamp is cut-off from each flat

        1.2. The stamp is smoothed by 
             \code{.sh} 
                 fitrans", stamp_fname, "--smooth",\
                 "polynomial,order=2,iterations=1,sigma=3,detrend"
             \endcode

        1.3. Iteratively rejected mean and standard deviation are calculated:
             \code{.sh}
                 "fiinfo", "--statistics", "mean,iterations=3,sigma=3"
             \endcode

        1.4. The number of non-saturated (I think really clean) pixels is
             calculated:
             \code{.sh}
                 fiinfo stamp_fname -m
             \endcode
             find a line starting with 8 dashes ('-') and use the number of pixels
             fiinfo reports for that line.

        1.5. if 1 - (number of pixels from step 4) / (total pixels in stamp) is bigger
             than some number reject the frame.

        1.6. If the frame is not rejected reuturn its mean and standard deviation
             from step 1.3.

    2. A check is performed for clouds:

        2.1. Fit a quadratic to the standard deviation vs mean from step 1
             above.
             \code{.sh}
                lfit -c 'm:1,s:2' -v 'a,b,c -f 'a*m^2+b*m+c' -y 's^2' -r\
                "%f"%rej_params.sigma_level -n 2 --residual
             \endcode

        2.2. If the fit residual as reported by lfit is larger than some
             critical value, the entire group of flats is discarded and no
             master is generated.

        2.3. If the fit is acceptable, but a frame is too far away from the
             best-fit line, the frame is discarded.

    3. Flats are split into low and high:

        3.1. The median (MEDMED below) and the median absolute deviation from the
             median (MADMED) of all means from step 1 is calculated.

        3.2. Frames with mean above MEDMED - (rej_params.min_level * MEDMED) and
             above some absolute threshold are considered high.

        3.3. Frames below a different threshold are considered low.

        3.4. Frames that are neither low nor high are discarded.

    4. Frames for which the pointing as described in the headir is within some
       critical arc-distance are discarded. So are frames missing pointing
       information in their headers.

    5. If after all rejection steps above, the number of flats is not at least
       some specified threshold, no master is generated.

    6. A preliminary master flat is created from all high flats using an
       iteratively rejected median:
       \code{.sh}
            ficombine --mode 'rejmed,sigma=4,iterations=1'\
            calib_flat1.fits\
            calib_flat2.fits\
            ...\
            --output preliminary_master.fits
       \endcode

    7. Scale each individual calibrated flat frame to the same large scale
       structure as the preliminary master flat from step 6. For
       calib_flat1.fits the commands are:
       \code{.sh}
           fiarith "'preliminary_master.fits'/'calib_flat1.fits'"\
           | fitrans --shrink 4\
           | fitrans --input - --smooth params.args --output -\
           | fitrans --zoom 4\
           | fiarith "'calib_flat1.fits'*'-'*4" --output scaled_flat1.fits
       \endcode

    8. Calculate the maximum deviation between each scaled frame and the
       preliminary master in a stamp near the center spanning 75% of each
       dimension of the input scaled flat. Assuming a frame resolution of
       4096x4096:
       \code{.sh}
            fiarith "'scaled_flat1.fits'/'preliminary_master.fits'-1"\
            | fitrans --shrink 4\
            | fitrans --offset '128,128' --size '768,768'\
            | fitrans --smooth 'median,hsize=4,iterations=1,sigma=3'\
            | fitrans --zoom 4\
            | fiinfo --data 'min,max'
       \endcode
       The deviation is the maximum in absolute value of the two values
       returned.

    9. If the deviation from step 8 is bigger than some critical value (0.05 for
       HATSouth) the frame is rejected as cloudy.

    10. If enough unrejected frames remain, a master flat is generated by median
        combining with rejecting outliers:
        \code{.sh}
            ficombine --mode 'rejmed,iterations=2,lower=3,upper=2'\
            scaled_flat1.fits\
            scaled_flat2.fits\
            ...\
            --output master_flat.fits
        \endcode
