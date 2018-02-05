# Low Level Astrometry Tools

## Source extraction

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
  * id - A unique integer identifier of the source
  * x - The x coordinate of the souce in units of pixels
  * y - The x coordinate of the souce in units of pixels
  * signal_to_noise - The combined signal to noise of all the pixels assigned
    to the source.
  * flux - Some estimate of the flux of the souce (as calculated by the source
    extractor used).
In addition, depending on the source extractor used, more columns are available.

The files generated have the column name as a first line, marked as a comment
by a leading `'#'`. To ensure readability by humans, all columns values are
aligned to a common left boundary, which is also the left boundary for the
column name in the header. All output columns are numeric.

## Catalogue query

For the moment, only the UCAC4 catalogue is supported, with the idea that
eventually, when GAIA data is publicly released it will be useable through a
very similar interface. Just like source extraction configuration can be
specified in three stages, and output can be as a structured numpy array and/or
a file. In this case however, not all columns are numeric.

The catalogue file format and the return nmupy array contain the following
columns for UCAC4:

  * `2MASS_ID`: unique 2MASS identifier of the object
  * `RA` The right ascention of the object, optionally corrected for proper motion
  * `Dec`: The declination of the object, optionally corrected for proper motion
  * `eRA`: Estimated uncertainty in RA
  * `eDec`: Estemiated uncertainty in Dec
  * `PM_RA`: The RA proper motion in units of milli arcsec / yr
  * `PM_Dec`: The Dec proper motion in units of milli arcsec / yr
  * `ePM_RA`: Estimated uncertainty in PM_RA
  * `ePM_Dec`: Estimated uncertainty in PM_Dec
  * `epochRA`: central epoch for RA entry, before any proper motion correction
  * `epochDec`: central epoch for Dec entry, before any proper motion correction
  * `J`, `H`, `K`: J-, H-, K- band magnitude
  * `eJ`, `eH`, `eK`: Estimated uncertainty in J, H, K respectively
  * `PH_QUALITY`: Photometric quality flag (see [2MASS documentation](https://www.ipac.caltech.edu/2mass/releases/allsky/doc/sec2_2a.html))
  * B, V, R, I, u, g, r, i, z: Estimated magnitude in the corresponding band.

Sources can be filtered based on the various quality flags available in the
corresponding catalogue used. For UCAC4, this means the following:

  * object type
  * double star flag
  * 2MASS photometric quality flag
  
Filtering can also be done on any of the numeric output columns listed above,
enforcing that only values in a specific interval are allowed.
  
## Matching and solving

Currently only anmatch is supported. As with source extraction and catalogue
querrying configuration can be specified in three stages: construction of the
solver, after construction and as one-time overwrites during individual solver
invocations. 

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

12345678901234567890123456789012345678901234567890123456789012345678901234567890
         1         2         3         4         5         6         7         8
