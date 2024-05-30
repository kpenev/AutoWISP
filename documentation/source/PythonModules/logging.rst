*******
Logging
*******

Design considerations
=====================

Logging all steps taken by a pipeline is a critical task to allow users to
monitor progress, debug and tune the various steps. Designing a logging system
which is flexible enough to handle all the various tasks that the pipeline will
perform, while at the same time supporting easy automated processing to generate
things like progress reports, statistics for various quantities of interest etc
is a non-trivial task. A badly designed logging system could lead to numerous
headaches down the line.

The requirements the selected logging scheme employed by the pipeline must
satisfy are:

    1. Offer a unifom mechanism to provide a logger and context information to
       each pipeline processor.

    2. Handle parallel processing without mangling the log.

    3. Allow configuring the formatting of log files and/or logging to database
       in a uniform way for all pipeline processors.


Common logger mechanism
=======================

Since the logging module ensures that a logger with the same name always returns
the same instance, the common ``__init__`` method of all pipeline processors
simply need to create child loggers in a tree leading to the same base logger.
This is trivially accomplished by each module using a logger with a name given
by ``__name__``.


Adding context information
==========================

This is also handled by the python logging module. In fact there are `two
mechanisms
<https://docs.python.org/3/howto/logging-cookbook.html#context-info>`_ to choose
from: LoggerAdapters and Filters. In AutoWISP we will use Filters, since it
results in easier handling of the extra contextual information by formatters and
handlers.

What remains to decide is what contextual information to add. Some possibilities
include:

1. Identifying information of the data being processed.
-------------------------------------------------------

This could be for example the frame being calibrated/astrometried/photometered,
the lightcurve being dumped/EPD-d/TFA-d. However, some steps work on collections
of data (e.g. stacking frames to a master, multi-frame PSF/PRF fitting, dumping
lightcurves for multiple sources from multiple frames). So the exact format and
entries become processor specific.

One possibily of handling this difficulty is to define a super-set of fields and
simply have some set to ``None`` depending on the situation.

Another possibility is to define a formatt for each separate set of
possibilities. If this is chosen, there should be some identifying flag in the
message identifying the formatter used to allow for easy automatic parsing.

2. Identifying information about the processing going on.
---------------------------------------------------------

This could be for example identifying whether an image being read was requested
by the image calibration or astrometry or photometry module. However, how much
granularity do we want. Do we specify that an image was requested by the mask
creation vs the bias subtraction part of the image calibration module.
