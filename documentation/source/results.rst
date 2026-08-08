***************************
Looking at the light curves
***************************

A light curve is what all the processing was for: one star's brightness
measured over and over, with as much of the instrument and the atmosphere
removed from it as the pipeline can manage. The results page is where you
look at them.

It is called **Review Results** in the navigation. You can also arrive at
a particular star by clicking its point on a detrending diagnostics plot
(see :doc:`diagnostics`), which is the usual way in when you have spotted
one star behaving unlike the rest.

The first plot
==============

Press **Apply** and you get the plot you most often want: magnitude
against time.

Two things about it are worth knowing, because neither is obvious from
looking at it.

The magnitude has its median subtracted, so the vertical axis is
*variation* rather than brightness -- zero means "this star's usual
brightness", and the numbers either side are how far it strayed. Time is
measured in days from the first observation for the same reason. Both are
choices about presentation, not about the data, and both can be changed.

More consequential: the aperture is not one you chose. Aperture
photometry measured the star through every aperture you configured, and
the plot uses whichever of them gives the least scatter for *this* star --
found by trying each and keeping the smallest median absolute deviation
about the median. Since the best aperture depends on how bright the star
is, this is usually what you want, and it means two stars plotted from the
same data may be shown through different apertures. The rule is yours to
change, and what it minimises is an expression like any other.

Comparing what detrending did
=============================

**Open Config** reveals the plotting controls, and among them two rows of
buttons labelled **magfit**, **EPD** and **tfa**. They do different
things, which is worth getting straight:

* the pair beside **Minimize** set what is being minimised when the best
  aperture is chosen -- the scatter after magnitude fitting, after EPD, or
  after TFA;
* the pair under **Plot Quantities** switch what is drawn to that stage's
  magnitudes.

Plotting the same star at each of the three stages in turn is the most
useful thing this page does. The detrending diagnostics tell you what EPD
and TFA did to the ensemble; this tells you what they did to *your* star,
which is not the same question and occasionally has an unwelcome answer.

Both stages remove signals shared between stars, and a real signal that
happens to resemble one -- anything slow, or anything the whole field
does together -- can be removed along with the instrumental trends. TFA
is the more aggressive of the two. If a variation you believe in is
present after magnitude fitting, weaker after EPD and gone after TFA, the
detrending is the first suspect rather than the last.

Plotting other things
=====================

The quantities are not a fixed menu. Each is an expression over the
light curve's own datasets, so anything the light curve contains can go on
either axis: magnitude against airmass, against background, against
position on the detector, or against another magnitude.

This is how you chase down a trend you can see but cannot explain.  Residual
scatter that turns out to correlate with airmass is extinction; with position, a
flat field that is not doing its job; etc.. There is also a selection
expression, which restricts the plot to the points satisfying it -- useful for
cutting out a bad night, or looking only at what was observed above some
altitude.

Several selections can be drawn on the same axes, and the figure can be
divided into panels, so a before-and-after comparison or a magnitude panel
above a background panel are both a matter of configuration rather than of
exporting the data and plotting it yourself.

Overlaying a transit model
==========================

A transit model can be drawn over the data, given the usual parameters:
period, epoch, planet-to-star radius ratio, scaled semi-major axis,
inclination, limb darkening, and optionally eccentricity and argument of
periastron.

It is drawn, not fit. The point is to see whether a candidate signal sits
where a transit of the parameters you have in mind would sit, which is a
quick way to dismiss a coincidence or decide something deserves proper
attention.

The figure itself
=================

**Figure Config.** exposes the underlying plotting settings -- fonts,
sizes, and the rest -- for when a figure is going somewhere other than
your own screen. **Download Figure** writes the current figure as a PDF.

**Clear LC Buffer** discards the light curve data held for the session.
The data is kept so that changing a setting and pressing **Apply** again
is quick rather than re-reading files each time; clear it when you want to
be certain you are looking at what is on disk now, after re-running a
detrending step for instance.
