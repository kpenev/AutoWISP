rm -rf logs CAL MASTERS
python3 ../../../autowisp/processing_steps/calibrate.py -c test.cfg RAW/zero/*.fits.fz
python3 ../../../autowisp/processing_steps/stack_to_master.py -c test.cfg CAL/zero/
python3 ../../../autowisp/processing_steps/calibrate.py -c test.cfg RAW/dark/*.fits.fz --master-bias 'R:MASTERS/zero_R.fits.fz'
python3 ../../../autowisp/processing_steps/stack_to_master.py -c test.cfg CAL/dark/
python3 ../../../autowisp/processing_steps/calibrate.py -c test.cfg RAW/flat/*.fits.fz --master-bias 'R:MASTERS/zero_R.fits.fz' --master-dark 'R:MASTERS/dark_R.fits.fz'
python3 ../../../autowisp/processing_steps/stack_to_master_flat.py -c test.cfg CAL/flat/*.fits.fz
