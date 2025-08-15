10. Trend Filtering Algorithm (TFA)
===================================

**Commands:** ``wisp-tfa``, ``wisp-generate-tfa-statistics``

Important Parameters
--------------------

* :option:`tfa-datasets` (automatically set by BUI but not on the command line)

* :option:`tfa-faint-mag-limit`

* :option:`tfa-sqrt-num-templates`

In this step signals which are shared by mulitple stars are removed from each
star's lightcurve. The idea is that most instrumental effects will affect
multiple stars in a similar way, and thus signals common to several sources
are suspected of being instrumental, rather than real astrophysical variability.
Again this steps has the potential to distort or eliminate target signals, so it
should be used with care. If the shape of the target signal is known, there are
versions of this procedure which tend to preserve it.
