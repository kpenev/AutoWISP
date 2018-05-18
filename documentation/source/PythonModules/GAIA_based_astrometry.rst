*********************
GAIA Based Astrometry
*********************

Querying the GAIA Catalogue
===========================

The first task is to allow creating lists of sources  in a rectangular or
circular area of the sky down to some limiting magnitude, optionally imposing
cuts based on other factors, e.g.  color quality flags etc. The minimum
information required is:

  * GAIA Source Identifier

  * Sky position (RA, Dec), optionally corrected for proper motion

  * Some indicator of the brightness of the source in a relevant filter

  * Any relevant quality flags for the reliability of the position or
    brightness.

It is worth considering if more efficient queries can be accomplished by
re-packaging the GAIA data release files into another format, e.g. one splitting
the data based on HEALPix of a level coarser than 12, sorting by flux etc. Some
options include:

+--------+----------------------------------+----------------------------------+
| Format |             Advantages           |      Disadvantages/Concerns      |
+========+==================================+============+=====================+
|        |  * Can add indices by            |  * Could result in big data      |
|        |    sub-HEALPix, brightness,      |    volume.                       |
| mysql  |    color etc.                    |                                  |
|        |                                  |                                  |
|        |  * Can partition table by super- |                                  |
|        |    HEALPix.                      |                                  |
+--------+----------------------------------+----------------------------------+
|        |  * Can add indices by            |  * Could result in big data      |
|        |    sub-HEALPix, brightness,      |    volume.                       |
| sqlite |    color etc.                    |                                  |
|        |                                  |                                  |
|        |  * Each batch of data can be in  |                                  |
|        |    a separate file               |                                  |
+--------+----------------------------------+----------------------------------+
|        |  * Very standard                 |  * More work to perform queries. |
| FITS   |                                  |                                  |
|        |  * Each batch of data can be in  |  * Possibly slower queries       |
|        |    a separate file               |    (no indices).                 |
+--------+----------------------------------+----------------------------------+
|        |  * Good compression options.     |  * More work to perform queries. |
| HDF5   |                                  |                                  |
|        |  * Each batch of data can be in  |  * Possibly slower queries.      |
|        |    a separate file               |    (no indices).                 |
+--------+----------------------------------+----------------------------------+

Queries will be accomplished in three or four steps depnending on how the data
is organized:

  1. Using the healpy module get a list of HEALPix indices covering the query
     area. 

  2. **If not using a database**, get a list of the GAIA data release files
     containing the identified indices.

  3. Extract all sources from the selected pixels and apply the specified
     magnitude, color etc. cuts.

  4. Apply proper motion corrections to the desired epoch.

Source Extraction
=================

There are four possible source extractor tools:

  * `fistar` (from HATpipe)

  * `hatphot` (from HATpipe)

  * `sextractor`    

  * `simplexy` (from astrometry.net)

In practice `simplexy` seems to misbehave if sources are over-sampled, requiring
binning, which necessarily degrades the precision of the extracted source
positions, os it is probably not competitive with the other options. All other
options will eventually be supported by the pipeline. The result of source
extraction will always consist of (x, y) image positons, an estiamte of the
source brightness and some indicators for the properties of the source (e.g.
extendend vs point source, signal to noise, etc.

Two Step Plate Solving
======================

Step 1: Astrometry.net
----------------------

`Astrometry.net <http://astrometry.net>`_ provides unparalleled robustness in
finding the location on sky of a collection of sources. To take advantage of
that we will first use a locally compiled copy of `solve-field` along with the
standard indices shipped by `astrometry.net` to find an approximate astrometric
solution for the extracted sources from the `Source Extraction`_ step, producing
a WCS file.

Step 2: Refinement based on GAIA
--------------------------------

The approximate solution derived above is refined as follows:

  1. The sky to frame transformaion from the WCS file above will be used to
     project the list of catalogue sources obtained by `Querying the GAIA
     Catalogue`_\ , using the `astropy.wcs module
     <http://docs.astropy.org/en/stable/wcs/index.html>`

  2. Build a `cKDtree
     <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html>`_
     from the projected sources, and query it for the neighbours within some
     tolerance to each extracted source.
     
  3. Resolve multiple candidate matches are using brightness informatino or
     discard matches if no clear determination can be made.

  4. Derive a polynomial transformation between tan-projected (RA, Dec)
     positions of the matched sources and the corresponding extracted image (x,
     y) positions. This will be done iteratively, re-deriving the center used
     for tan projection by inverting the last solution for the central image
     position, until the center changes no more than some tolerance. Usually
     just a single re-fit is necessary.
