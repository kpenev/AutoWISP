7. Magnitude fitting
====================

**Command:** ``wisp-fit-magnitudes``

Important Parameters
--------------------

* :option:`single-photref-dr-fname`

* :option:`correction-parametrization`

* :option:`magfit-catalog-max-magnitude`

Description
-----------

In ground based applications, the night sky is imaged through variable amount of
atmosphere, which itself is subject to changes (i.e. clouds, humidity, etc.). In
addition various instrumental effects are generally present. The purpose of the
magnitude fitting step is to eliminate as much as possible effects that modify
the measured source brightness within an image in a manner that depends
smoothly on the properties of the source.

In short, a reference frame is selected (and later generated). Then for each
individual frame (target frame from now on) a smooth multiplicative correction
is derived that when applied to the brightness measurements in the target frame
matches the brightness measurements in the reference frame as closely as
possible.

In the pipeline this is actually done multiple times. The first time, a single
frame which appears to be of very high quality (sharp PSF, high atmospheric
transparency, dark sky etc.) is used as the reference frame. The corrected
brightness measurements of the individua frames are then stacked to produce a
much highe signal to noise "master reference frame", which is then used in a
second iteration of the magnitude fitting process and so on.


