rm -rf logs CAL MASTERS DR LC
python3 ../..//database/initialize_database.py --drop-hdf5-structure-tables \
&& python3 ../../processing_steps/calibrate.py -c test.cfg RAW/zero/*.fits.fz \
&& python3 ../../processing_steps/stack_to_master.py -c test.cfg CAL/zero/ \
&& python3 ../../processing_steps/calibrate.py -c test.cfg RAW/dark/*.fits.fz --master-bias 'R:MASTERS/zero_R.fits.fz' \
&& python3 ../../processing_steps/stack_to_master.py -c test.cfg CAL/dark/ \
&& python3 ../../processing_steps/calibrate.py -c test.cfg RAW/flat/*.fits.fz --master-bias 'R:MASTERS/zero_R.fits.fz' --master-dark 'R:MASTERS/dark_R.fits.fz' \
&& python3 ../../processing_steps/stack_to_master_flat.py -c test.cfg CAL/flat/*.fits.fz \
&& python3 ../../processing_steps/calibrate.py -c test.cfg RAW/object/*.fits.fz --master-bias 'R:MASTERS/zero_R.fits.fz' --master-dark 'R:MASTERS/dark_R.fits.fz' --master-flat 'R:MASTERS/flat_R.fits.fz' \
&& python3 ../../processing_steps/find_stars.py -c test.cfg CAL/object \
&& python3 ../../processing_steps/solve_astrometry.py -c test.cfg DR \
&& python3 ../../processing_steps/fit_star_shape.py -c test.cfg CAL/object \
&& python3 ../../processing_steps/measure_aperture_photometry.py -c test.cfg CAL/object \
&& python3 ../../processing_steps/fit_source_extracted_psf_map.py -c test.cfg DR \
&& python3 ../../processing_steps/fit_magnitudes.py -c test.cfg DR \
&& python3 ../../processing_steps/create_lightcurves.py -c test.cfg DR \
&& python3 ../../processing_steps/epd.py -c test.cfg LC \
&& python3 ../../processing_steps/generate_epd_statistics.py -c test.cfg LC \
&& python3 ../../processing_steps/tfa.py -c test.cfg LC \
&& python3 ../../processing_steps/generate_tfa_statistics.py -c test.cfg LC
