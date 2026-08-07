*************************************
Design of the pipeline code structure
*************************************

This document outlines the basic design of the pipeline aimed at achieving
maximum flexibility and extendability.

Processors
==========

NOT TRUE!!!

All pipeline processing should be done through classes that inherit from a
common base class: :class:`autowisp.processor`, which should provide a
uniform interface for configuring things like :doc:`logging` and
:doc:`crash_recovery`.  Operations which are shared among multiple processors,
yet are so atomic that they could not issue useful logging messages or upon
crash can only be recovered by discarding all progress should avoid this
mechanism and be implemented as stand-alone functions.

Code layout
===========

Each main-level step of the pipeline (see :doc:`../test_data`) sits in a
separate python module, with first level sub-steps implemented
as classes each sitting in its own ``.py`` file. In order to avoid excessively
long import statements the ``__init__.py`` files for each main-level module
should import the individual classes from their respective python files and
adding them to its ``__all__`` variable.
