# Regenerate the test data's products by running each step on its own, from
# the bundle's directory (its RAW and test.cfg).
#
# This does not reproduce the bundle entirely. The pipeline engine adds the
# observing session's identifiers (TARGETID, TELSCPID, CAMERAID, OBSSSNID,
# OBSERVER) to the headers it calibrates, and steps run on their own have no
# session to take them from. The bundle carries them, copied from the outputs
# of a TestFullPipeline run into the calibrated images, the masters, the DR
# files' copies of the frame header and the lightcurve catalogue, so that the
# full pipeline test compares them. Copy them in again after running this, or
# TestFullPipeline fails.
rm -rf logs CAL MASTERS DR LC
wisp-calibrate . -c test.cfg RAW/zero/*.fits.fz \
&& wisp-stack-to-master . -c test.cfg CAL/zero/ \
&& wisp-calibrate . -c test.cfg RAW/dark/*.fits.fz --master-bias 'R:{PROJHOME}/MASTERS/zero_R.fits' \
&& wisp-stack-to-master . -c test.cfg CAL/dark/ \
&& wisp-calibrate . -c test.cfg RAW/flat/*.fits.fz --master-bias 'R:{PROJHOME}/MASTERS/zero_R.fits' --master-dark 'R:{PROJHOME}/MASTERS/dark_R.fits' \
&& wisp-stack-to-master-flat . -c test.cfg CAL/flat/ \
&& wisp-calibrate . -c test.cfg RAW/object/*.fits.fz --master-bias 'R:{PROJHOME}/MASTERS/zero_R.fits' --master-dark 'R:{PROJHOME}/MASTERS/dark_R.fits' --master-flat 'R:{PROJHOME}/MASTERS/flat_R.fits' \
&& wisp-find-stars . -c test.cfg CAL/object \
&& wisp-solve-astrometry . -c test.cfg DR \
&& wisp-fit-star-shape . -c test.cfg CAL/object \
&& wisp-measure-aperture-photometry . -c test.cfg CAL/object \
&& wisp-fit-source-extracted-psf-map . -c test.cfg DR \
&& wisp-fit-magnitudes . -c test.cfg DR \
&& wisp-create-lightcurves . -c test.cfg DR \
&& wisp-epd . -c test.cfg LC \
&& wisp-generate-epd-statistics . -c test.cfg LC \
&& wisp-tfa . -c test.cfg LC \
&& wisp-generate-tfa-statistics . -c test.cfg LC
