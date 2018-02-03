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

## Matching and solving
