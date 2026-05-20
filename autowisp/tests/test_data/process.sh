rm -rf logs CAL MASTERS DR LC
wisp-calibrate -c test.cfg RAW/zero/*.fits.fz \
&& wisp-stack-to-master -c test.cfg CAL/zero/ \
&& wisp-calibrate -c test.cfg RAW/dark/*.fits.fz --master-bias 'R:MASTERS/zero_R.fits.fz' \
&& wisp-stack-to-master -c test.cfg CAL/dark/ \
&& wisp-calibrate -c test.cfg RAW/flat/*.fits.fz --master-bias 'R:MASTERS/zero_R.fits.fz' --master-dark 'R:MASTERS/dark_R.fits.fz' \
&& wisp-stack-to-master-flat -c test.cfg CAL/flat/*.fits.fz \
&& wisp-calibrate -c test.cfg RAW/object/*.fits.fz --master-bias 'R:MASTERS/zero_R.fits.fz' --master-dark 'R:MASTERS/dark_R.fits.fz' --master-flat 'R:MASTERS/flat_R.fits.fz' \
&& wisp-find-stars -c test.cfg CAL/object \
&& wisp-solve-astrometry -c test.cfg DR \
&& wisp-fit-star-shape -c test.cfg CAL/object \
&& wisp-measure-aperture-photometry -c test.cfg CAL/object \
&& wisp-fit-source-extracted-psf-map -c test.cfg DR \
&& wisp-fit-magnitudes -c test.cfg DR \
&& wisp-create-lightcurves -c test.cfg DR \
&& wisp-epd -c test.cfg LC \
&& wisp-generate-epd-statistics -c test.cfg LC \
&& wisp-tfa -c test.cfg LC \
&& wisp-generate-tfa-statistics -c test.cfg LC
