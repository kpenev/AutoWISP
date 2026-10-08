**********************************
Choosing the photometric reference
**********************************

Magnitude fitting works by correcting every frame to agree with a
reference. You choose the frame it starts from, and it is the one point
in processing where the pipeline stops and asks you to decide.

It asks because the decision cannot be made from numbers alone, which is
the whole reason this page exists.

Getting there
=============

Processing pauses when it reaches magnitude fitting and needs a reference
it does not have. On the progress page, the image type beside the
magnitude fitting row becomes a link -- hovering says "select reference
image" -- and that is the way in.

A starting point, not the final standard
========================================

Worth knowing before you agonise over the choice: the frame you pick is
where magnitude fitting begins, not what it ultimately measures against.

Once every frame has been corrected to agree with your choice, the fit
averages each star's corrected magnitudes across all the frames in the
group. That average becomes a new reference -- one assembled from the
whole group rather than from any single frame -- and everything is fitted
again against it. The cycle repeats until two successive references stop
differing appreciably, or until :option:`max-magfit-iterations` (five by
default) is reached.

So a slightly imperfect choice is not fatal. The ensemble pulls the
standard towards itself, and a frame that is merely unremarkable will
converge to much the same place as the best one.

What the choice still decides is where that convergence starts from -- a
bad enough starting frame can leave the iteration somewhere worse than it
needed to be -- and, through the pointing limit described below, which
frames are allowed to take part at all. That second one does not wash out
with iteration: frames excluded at the start are excluded for good.

One reference per group
=======================

You do not choose one frame for the whole project. A reference is only
comparable to frames that resemble it, so the frames are divided into
groups and each group gets its own. By default a group is one field, one
colour channel and one exposure time.

The first page lists the groups still waiting, what defines each of them,
and how many candidate frames it has to choose from. A group with very
few candidates deserves more care than one with hundreds, since there may
be nothing good to fall back on.

What defines a group is configuration, not a fixed rule
-------------------------------------------------------

The three things above are the default, not a law. The grouping is the
set of expressions the project matches photometric references on, and it
is yours to set -- along with the rest of the master configuration, when
the project is created. :doc:`bringing_your_own_data` covers where that
is done and what else it decides.

It is worth thinking about before you start, because the sensible answer
depends on the project. With more than one camera, for instance, the
choice is whether to include the camera in the grouping:

* **Include it** and each camera gets its own reference. Every camera is
  then calibrated against a frame of its own, which is the safer choice
  when the cameras differ enough that a frame from one is a poor standard
  for another -- different optics, different filters, different
  sensitivity.
* **Leave it out** and all the cameras are fitted to a single reference.
  That puts every camera on one magnitude scale, so their measurements of
  the same star can be combined directly rather than each floating on its
  own zero point.

Neither is right in general. Separate references keep each camera
internally consistent but leave the cameras independent of one another;
a shared reference ties them together, at the cost of asking one camera's
frame to serve as the standard for the rest. The same question arises
wherever frames fall into natural batches -- observing seasons, sites,
filters -- and the answer is the one that matches what you intend to
compare with what.

Small groups are usually not worth a reference
----------------------------------------------

Because each group is calibrated against its own reference, each ends up
on its own magnitude zero point. Combining groups into one light curve
therefore means putting them onto a common scale first, and the usual way
to do that is to subtract each group's own median before stitching them
together.

That subtraction is only as good as the median it removes. Over many
frames the median is a fair estimate of the star's usual brightness and
taking it out aligns the groups properly. Over a handful of frames it is
mostly noise -- and if the star varies, it is partly signal, so removing
it takes a piece of the variation with it. Either way the group arrives
on the common scale with an offset of its own, which is worse than not
having it at all.

There is a second reason, from the iteration described above. The
reference the fit builds for itself is an average over the frames in the
group, so a group with few frames has little to average: the ensemble
never becomes much better determined than the one frame you started from,
and the iteration has nothing to converge towards. A large group ends up
measured against itself; a small one stays measured against whichever
frame you happened to pick, quirks and all.

So a group with only a few frames is generally better left without a
reference: its frames then stay out of magnitude fitting, rather than
joining the light curve carrying an offset nobody can measure. How few is
too few depends on how much the star varies and how precise the
photometry is, but a group whose median rests on a handful of points is
not contributing anything you would want to trust.

Start at the top, then look
===========================

Within a group the candidates are ranked, best first, by the merit
function -- by default the sharpest stars on the darkest sky, described
in :doc:`diagnostics` along with how to change the rule.

That ranking is the pipeline's opinion, and it is formed entirely from
the diagnostics. The diagnostics are summaries: a median, a residual, a
count. **They cannot see an aeroplane drawing a line across the field, a
satellite streak, the edge of a cloud that happened to miss the centre of
the frame, dew starting on the corrector, or a reflection from something
bright just outside the field.** A frame can be the best in the group on
every number recorded and still be unusable for the one job you are
picking it for.

So the procedure is: start at the top of the ranking, look at the frame,
and step down until you find one that is clean. Usually that is the first
one. When it is not, the reason is normally visible at a glance.

What you are shown
==================

The frame itself, with the display stretch adjustable -- the default
scaling is chosen to show faint detail, which is what makes a satellite
trail or a thin cloud edge stand out rather than disappear into the
background. Next and previous step through the candidates in ranked
order, and you can also jump straight to a position in the ranking.

Beside it, a histogram for each diagnostic across all the candidates in
the group, with a line marking where the frame you are looking at falls.
This is more informative than the rank alone: it tells you whether the
best candidate is comfortably better than the rest, or one of fifty
near-identical frames where the ordering is arbitrary and you may as well
take whichever looks cleanest. The merit values get the same treatment.

Watch where it points
=====================

The last group of plots is about pointing, and it matters more than it
first appears. Frames whose centres lie farther from the reference than
:option:`max-photref-separation` -- measured in units of the frame's own
diagonal field of view -- are **excluded** from magnitude fitting
altogether.

So a reference taken from one end of a night's drift quietly strands the
frames from the other end. They are not corrected badly; they are not
corrected at all, and the stars in them simply do not appear in the light
curves.

The plots show where every candidate points, which of them are within
reach of the one you are looking at, and how the separations are
distributed against the limit. A reference near the middle of the cluster
keeps the most frames. Where the pointing has drifted a long way over a
run, the trade-off between the cleanest frame and the most central one is
a real one, and worth making deliberately.

Recording the choice
====================

Choosing a frame registers it as the reference for its group and binds
the group's frames to it, after which the group disappears from the list.
When none are left, processing can carry on into magnitude fitting.

**Take the time to be sure before you choose.** The choice is one-way:
there is at present no way to undefine a reference or swap it for another
short of editing the project database by hand. A frame you regret picking
stays the standard everything in its group is measured against.

If the ranking rule itself does not suit your data, the merit function
can be changed on the group list and the candidates re-ranked -- but that
only helps for groups you have not decided yet.
