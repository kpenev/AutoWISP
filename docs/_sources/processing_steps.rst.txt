**************
Pipeline Steps
**************

The pipeline operations can be broken down into 10 big steps each of which is
described in more detail below. The results of each step are saved in three
types of files:

calibrated images:
    These are the individual images of the night sky, corrected for bias, dark
    current and flat fielding (see below). Those are stored using the `FITS
    <https://fits.gsfc.nasa.gov/fits_primer.html>`_ format with three
    extensions: 

      * The values of the image pixels 

      * Formal estimate of the standard deviation of each pixel value

      * A pixel quality mask, flagging things like saturated pixels, pixels
        which may have received charge from overflowing neighboring pixels, etc.

data reduction files:
    These are the files that contain information about the stars in each image
    extracted from the calibrated images. Including things like star positions,
    point spread function information, flux measurements, etc. They are
    generally in FITS format and can be used for further analysis. Those are in
    `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_ format.

lightcurves:
    These are the files that contain the time series of all available
    measurements for a given star star in each image. They are generally in
    `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_ format, and include
    things like several different version of flux measurements, coordinates of
    the star in the image, and many others.

Split Raw Frames by Type:
=========================

Before you begin processing the images you have accumulated, a bit of
preparation is needed. It is very convenient and helps avoid mistakes if you
split your images inte separate directories by type:


    bias
        Images with near-zero exposure intended to measure the behavior of the
        analog-to-digital converter(s) of your camera. This shows up in the
        final image as a value that each pixel starts at, even if there is no
        signal.

    dark
        Images with no light falling on the detector, but exposure similar
        (ideally equal) to the exposure used for science images. These are
        intended to measure the rate of accumulation of charge in the detector
        pixels in the absence of light.

    flat 
        Images of something with uniform brightness (or as close to it as one
        can manage). There are intended to measure the sensitivity to light of
        the system coming from different directions. 

    object 
        Images of the night sky from which photometry is to be extracted. Those
        can further be split into sub-groups from which independent lightcurves
        need to be generated. For example if several different exposure times
        were used, or there could be a number of filters or other chages in the
        optical system between frames which may produce better results if
        processed independently.

.. include:: calibration_steps.rst

.. _srcfind-section:

.. include:: source_extraction.rst

.. include:: astrometry.rst

.. _psffit-section:

.. include:: psf_fitting.rst

.. _apphot-section:

.. include:: apphot.rst

.. include:: source_extraction_psf_map.rst

.. include:: magfit.rst

.. include:: create_lcs.rst

.. include:: epd.rst

.. _tfa-section:

.. include:: tfa.rst
