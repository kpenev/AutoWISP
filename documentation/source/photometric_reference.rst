**********************************
Choosing the photometric reference
**********************************

Magnitude fitting works by comparing every frame against one chosen
frame, correcting each to agree with it. That chosen frame is the
photometric reference, and everything downstream inherits whatever is
wrong with it. It is the one point in processing where the pipeline stops
and asks you to decide.

It asks because the decision cannot be made from numbers alone, which is
the whole reason this page exists.

Getting there
=============

Processing pauses when it reaches magnitude fitting and needs a reference
it does not have. On the progress page, the image type beside the
magnitude fitting row becomes a link -- hovering says "select reference
image" -- and that is the way in.

One reference per group
=======================

You do not choose one frame for the whole project. A reference is only
comparable to frames taken of the same field, in the same colour channel,
with the same exposure time, so the frames are divided into groups on
exactly those three things and each group gets its own.

The first page lists the groups still waiting, what defines each of them,
and how many candidate frames it has to choose from. A group with very
few candidates deserves more care than one with hundreds, since there may
be nothing good to fall back on.

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

If you want to revisit a choice, or the ranking rule turns out not to
suit your data, the merit function can be changed on the group list and
the candidates re-ranked.
