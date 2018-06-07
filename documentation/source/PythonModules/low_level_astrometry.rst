******************************
Low Level Astrometry Interface
******************************

Given calibrated FITS frames of the night sky produced as described in
:doc:`low_level_image_calibration`, this module finds an astrometric solution
defining how (RA, Dec) coordinates on the sky are mapped to image (x, y). The
transformations are done in two steps:

    #. Using some pre-defined projection (e.g. one of the well know cartograhpic
       projections) to convert (RA, Dec) to a new set of coordinates
       (:math:`\xi`, :math:`\eta`) around the (RA, Dec) position corresponding
       to the center of the frame. **This transformatino has no free
       parameters.**
       
    #. Using a low-order polynomial to transform (:math:`\xi`, :math:`\eta`) to
       (x, y). The polynomial coefficients are determined using least squares
       fitting between extracted source positions and projected catalogue
       positions.

The procedure involves the following steps: 

Source Extraction
=================

Identify sources in the image. This is done by invoking an external source
extractor (e.g. fistar, hatphot or sextractor) as a subprocess, then parsing and
filtering the result to produce as clean and as complete as possible list of
stars contained in the frame.

fistar
    This tool provides (x, y), estimate of the flux, an elliptical Gaussian fit
    for the shape of the source and a signal to noise estimate. Filtering is
    done based on signal to noise and shape parameters to remove outliers.

hatphot:
    Similarly to fistar, this tools also provides estimates of the source shape
    which is used to filter strange looking sources.

sextractor
    This tool is currently not supported but should be.

Currently, source extraction can use fiphot or hatphot. Both of these are
supported along with a clear scheme for adding support for other source
extraction tools. In order to accomplish this, all source extraction inherints
from an abstract base class, which enforces e a common interface. The scheme to
followed is similar to the low level image calibration module, where all
configuration of the source extractor can be specified through __init__,
modified later through calling configuratino methods and overwritten for for a
single frame only through additional arguments when invoking source extraction.
Extracting the sources from a single frame is done through the __call__ method.

When source extraction is performed, extracted sources can be stored in a file
and/or returned as a structured numpy array. The array has at least the
following fields:

  * ``id`` - A unique integer identifier of the source
  * ``x`` - The x coordinate of the souce in units of pixels
  * ``y`` - The x coordinate of the souce in units of pixels
  * ``signal_to_noise`` - The combined signal to noise of all the pixels
    assigned to the source.
  * ``flux`` - Some estimate of the flux of the souce (as calculated by the
    source extractor used).

In addition, depending on the source extractor used, more columns are available.

The files generated have the column name as a first line, marked as a comment by
a leading ``#``. To ensure readability by humans, all columns values are aligned
to a common left boundary, which is also the left boundary for the column name
in the header. All output columns are numeric.

Catalogue query
===============

For the moment, only the UCAC4 catalogue is supported, and work is in progress
to implement support for GAIA. Just like source extraction configuration can be
specified in three stages, and output can be as a structured numpy array and/or
a file. In this case however, not all columns are numeric.

The catalogue file format and the return nmupy array contain the following
columns for UCAC4:

  * ``2MASS_ID``: unique 2MASS identifier of the object
  * ``RA`` The right ascention of the object, optionally corrected for proper
    motion
  * ``Dec``: The declination of the object, optionally corrected for proper
    motion
  * ``eRA``: Estimated uncertainty in RA
  * ``eDec``: Estemiated uncertainty in Dec
  * ``PM_RA``: The RA proper motion in units of milli arcsec / yr
  * ``PM_Dec``: The Dec proper motion in units of milli arcsec / yr
  * ``ePM_RA``: Estimated uncertainty in PM_RA
  * ``ePM_Dec``: Estimated uncertainty in PM_Dec
  * ``epochRA``: central epoch for RA entry, before any proper motion correction
  * ``epochDec``: central epoch for Dec entry, before any proper motion
    correction
  * ``J``, ``H``, ``K``: J-, H-, K- band magnitude
  * ``eJ``, ``eH``, ``eK``: Estimated uncertainty in J, H, K respectively
  * ``PH_QUALITY``: Photometric quality flag (see
    `2MASS documentation
    <https://www.ipac.caltech.edu/2mass/releases/allsky/doc/sec2_2a.html>`_\ )
  * ``B``, ``V``, ``R``, ``I``, ``u``, ``g``, ``r``, ``i``, ``z``: Estimated
    magnitude in the corresponding band.

Sources can be filtered based on the various quality flags available in the
corresponding catalogue used. For UCAC4, this means the following:

  * object type
  * double star flag
  * 2MASS photometric quality flag
  
Filtering can also be done on any of the numeric output columns listed above,
enforcing that only values in a specific interval are allowed.

Approximate solution
====================

The `astromtery.net <http://astrometry.net/>`_ code is used to derive a very
good first guess for the transformation. Experience shows that this is an
extremely robust tool, probably sufficient for all our needs.

Refining the solution
=====================

This is done through the following steps:

    1. Once an approximate solution is known, it is used to match catalogue
       sources to extracted sources by building a `cKDtree
       <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html>`_
       from the projected catalogue sources and querying it for the closest
       match to all extracted sources, with matches only considered
       successful if they are within some pre-determined radius.

    2. Matched catalogue (RA, Dec) coordinates are then projected to
       (:math:`\xi`, :math:`\eta`) around the best guess for the (RA, Dec)
       corresponding to the frame center. These then are used to derive a
       refined transformation between (:math:`\xi`, :math:`\eta`) and (x, y)
       using least squares fitting.

    3. This transformation is then inverted to find a new best guess for the
       (RA, Dec) of the center of the frame.
       
Steps 2 and 3 are iterated until the central (RA, Dec) remains within some
tolerance.

As with source extraction and catalogue querrying configuration for this step
can be specified in three stages: construction of the solver, after construction
and as one-time overwrites during individual solver invocations.

Successfully extracted solutions are saved as a collection of datasets in an
HDF5 file associated with the frame:

  * The sky-to-frame transformation found
  * The match between the extracted and catalogue sources
  * The extracted sources for which no catalogue match was found
  * The catalogue sources which transform to a position included in the frame
    per the transformation found, but for which no extracted source was found.
    
The inclusion of any of these data sets can be turned on or off separately, with
exact layout within the file configured separately by a base class for HDF5 I/O,
which uses either an XML file or the database.
