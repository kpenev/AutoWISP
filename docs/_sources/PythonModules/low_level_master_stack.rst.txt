********************************
Low Level Master Stack Interface
********************************

Given a user-specified set of calibrated FITS frames produced as described in
:doc:`low_level_image_calibration`, this module stacks them to create master
frames. The stacking procedure and example usage for each master types is as
follows:

Master Bias/Dark
================

The procedure for generating one of these two types of masters is that each
pixels in the output image in generated from the corresponding pixels in the
individual input images by iterating between the following two steps:

    #. Finding the median.

    #. Rejecting values differing from the median by more than some specified
       threshold times the root mean square deviation from the median.

Possible modifications:
-----------------------

    #. Using a different averaging method than median. In order to support
       iterations, whatever the method is, should ignore NaN values in the input
       images.
    
    #. More general stacking, i.e. not one rejecting pixels based on deviation
       from average, perhaps using external informaiton etc.

Master Flat
===========

Since flat frames can be images of the sky or of perhaps a dome with changing
illumination, special care must be taken to compensate for changes in the large
scale structure from one calibrated flat frame to the other. Further, with sky
frames there is always the possibility of clouds, so there needs to be an
automated procedure for detecting clouds in individual frames and discarding
them, or of detecting cloudy flat collections and refusing to generate a master
flat altogether. The procedure used by HATSouth is as follows:

    1. For each flat a mean and standard devation are calculated:

        1.1. A central stamp is cut-off from each flat

        1.2. The stamp is smoothed by::

            fitrans stamp.fits --smooth polynomial,order=2,iterations=1,sigma=3,detrend

        Translation:

            * A second order polynomial is fit to the pixel values.

            * More than 3-sigma outliers are rejected and the fit is repeated.

            * The image is then divided by the best-fit surface.

        1.3. The number of non-saturated (actually completely clean) pixels is
        calculated::

            fiinfo stamp.fits -m

        find a line starting with 8 dashes ('-') and use the number of pixels
        fiinfo reports for that line.

        1.4. if 1 - (number of pixels from step 4) / (total pixels in stamp) is
        bigger than some number reject the frame.

        1.5. If the frame is not rejected, iteratively rejected mean and
        standard deviation are calculated::

            fiinfo --statistics mean,iterations=3,sigma=3

    2. A check is performed for clouds:

        2.1. Fit a quadratic to the standard deviation vs mean from step 1
        above. More text on this longe line::

            lfit -c 'm:1,s:2'\
                -v 'a,b,c'\
                -f 'a*m^2+b*m+c'\
                -y 's^2'\
                -r '2.0'\
                -n 2\
                --residual

        Translation:

            * Fit a quadratic to (standard deviation)\ :sup:`2` vs (mean) from
              step 1.
         
            * Discard all points more than two sigma away from the fit go back
              to 2.1.1, for up to two iterations.

            * Get the best fit coefficients and the residual from the last fit.

        2.2. If the fit residual as reported by lfit is larger than some
        critical value, the entire group of flats is discarded and no master is
        generated.

        2.3. If the fit is acceptable, but a frame is too far away from the
        best-fit line, the frame is discarded.

    3. Flats are split into low and high:

        3.1. The median (MEDMEAN below) and the median absolute deviation from
        the median (MADMED) of all means from step 1 is calculated.

        3.2. Frames with mean above MEDMEAN - (rej_params.min_level * MADMED)
        and above some absolute threshold are considered high.

        3.3. Frames below a different threshold are considered low.

        3.4. Frames that are neither low nor high are discarded.

    4. Frames for which the pointing as described in the header is within some
       critical arc-distance are discarded. So are frames missing pointing
       information in their headers.

    5. If after all rejection steps above, the number of flats is not at least
       some specified threshold, no master is generated.

    6. A preliminary master flat is created from all high flats using an
       iteratively rejected median::

           ficombine --mode 'rejmed,sigma=4,iterations=1'\
           calib_flat1.fits\
           calib_flat2.fits\
           ...\
           --output preliminary_master.fits

       Translation:

       Each pixel of the preliminary_master.fits image is the median of the
       corresponding pixels of the individual frames, with a single iteration of
       rejecting pixels more than 4 standard devitaions away and re-fitting.

    7. Scale each individual calibrated flat frame to the same large scale
       structure as the preliminary master flat from step 6. For
       calib_flat1.fits the commands are::

           fiarith "'preliminary_master.fits'/'calib_flat1.fits'"\
           | fitrans --shrink 4\
           | fitrans --input - \
             --smooth median,hsize=6,spline,order=3,iterations=1,sigma=5,unity\
             --output -\
           | fitrans --zoom 4\
           | fiarith "'calib_flat1.fits'*'-'*4" --output scaled_flat1.fits

       Translation:

       For each individual calibrated flat (target):

       * Calculate the ratio of the preliminary master to the target.

       * Take each 4x4 array of pixels and average all their values into a
         single pixels of the output image, thus reducing the resolution by a
         factor of 4 in each direction.

       * Perform median box-filtering with a box half-size of 6 pixels,
         somehow combined with cubic spline fitting, with a single iteration
         of discarding pixels more than 5 sigma discrepant. The resulting
         image is the fit scaled to have a mean of 1.

       * Expand the image back up by a factor of 4, using 

             "a biquadratic subpixel-level interpolation and therefore exact
             flux  conservation."

         To quote from the fitrans --long-help message.

       * The individual flat is multiplied by the expanded image and by an
         additional factor of 4 to make its large scale structure the same as
         the preliminary master flat.

    8. Calculate the maximum deviation between each scaled frame and the
       preliminary master in a stamp near the center spanning 75% of each
       dimension of the input scaled flat. Assuming a frame resolution of
       4096x4096::

            fiarith "'scaled_flat1.fits'/'preliminary_master.fits'-1"\
            | fitrans --shrink 4\
            | fitrans --offset '128,128' --size '768,768'\
            | fitrans --smooth 'median,hsize=4,iterations=1,sigma=3'\
            | fitrans --zoom 4\
            | fiinfo --data 'min,max'

       The deviation is the maximum in absolute value of the two values
       returned.

       Translation

           * Create an image with each pixel being the fractional difference
             between the scaled flat from step 7 and the preliminary master from
             step 6.

           * Shrink the image by a factor of four along each dimension.

           * Cut-out the central 75% of the relusting frame.

           * Smooth the cut-out by median box-filter with a box half-size of 4
             pixels, with a single iteration of rejecting more than 3-sigma
             outliers and re-smoothing.

           * The result is zoomed back up using bi-quadratic interpolation.

           * Get the largest absolute value of the smoothed image.

    9. If the deviation from step 8 is bigger than some critical value (0.05 for
       HATSouth) the frame is rejected as cloudy.

    10. If enough unrejected frames remain, a master flat is generated by median
        combining with rejecting outliers::

            ficombine --mode 'rejmed,iterations=2,lower=3,upper=2'\
            scaled_flat1.fits\
            scaled_flat2.fits\
            ...\
            --output master_flat.fits

        Each pixel of the final master flat is the median of the corresponding
        pixels of the surviving individual scaled flats with up to two
        iterations of rejecting more than 3-sigma outliers in the downward
        directions and 2-sigma in the upward direction.

Possible Modifications:
-----------------------

    #. For step 1:

        #. Allow an array of smoothing methods in step 1.

        #. Allow more general frame rejection.

    #. For step 2:

        #. More general cloud detection, possibly using color or other external
           information.

    #. For step 6:

        #. more general stacking (e.g. different weights for different flats.)

    #. For step 7:

        #. More general smoothing

        #. More general matching of large scale structure

    #. For step 10: see step 6.

    #. Support for an entirely different procedure.
