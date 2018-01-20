# Design of the pipeline code structure {#PipelineDesign_page}

This document outlines the basic design of the pipeline aimed at achieving
maximum flexibility and extendability.

## Processors

All pipeline processing should be done through classes that inherit from a
common base class: PipelineProcessor, which should provide a uniform interface
for configuring things like \ref Logging_page and \ref CrashRecovery_page.
Operations which are shared among multiple processors, yet are so atomic that
they could not issue useful logging messages or upon crash can only be recovered
by discarding all progress should avoid this mechanism and be implemented as
stand-alone functions.

## Code layout

Each main-level step in \ref mainpage (i.e. those with a single number) sit in
separate python modules, with first level sub-steps implemented as classes each
sitting in its own `.py` file. In order to avoid excessively long import
statements the `__init__.py` files for each main-level module should import the
individual classes from their respective python files and adding them to its
`__all__` variable.
