Image Calibration Implementation {#ImageCalibrationImplementation_page}
================================

The \ref LowLevelImageCalibration_page provides tools to:
    * manually calibrate frames.
    * stack calibrated master, bias or dark frames into masters.

Designing and implementing the higher lever interface is still pending. The high
level interface must:
    * automate the process
    * take its configuration and log progress and results to the database
    * provide crash recovery
