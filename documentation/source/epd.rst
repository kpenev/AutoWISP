9. External Parameter Decorrelation (EPD)
=========================================

**Commands:** ``wisp-epd``, ``wisp-generate-epd-statistics``

Important Parameters
--------------------

* :option:`variables`

* :option:`epd-datasets` (automatically set by BUI but not on the command line)

* :option:`epd-terms-expression`

Description
-----------

The ``wisp-epd`` command removes from each individual lightcurve the linear
combintion of user specified instrumental and other time variable parameters
that explain the most variance. Clearly care must be taken when selecting the
parameters to decorrelate against, lest they vary on similar timescales as the
target signal.  If this happens, this step will highly distort if not eliminate
the target signal.

The ``wisp-generate-epd-statistics`` command is used to calculate summary
statistics for each lightcurve, showing how much scatter remains in it. This
information is useb by the :ref:`TFA step <tfa-section>` to select stars that
would serve as good templates.



