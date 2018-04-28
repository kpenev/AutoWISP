********************************
Image Calibration Implementation
********************************

The :doc:`low_level_image_calibration` provides tools to:

    * manually calibrate frames.

    * stack calibrated master, bias or dark frames into masters.

The :doc:`low_level_master_stack` provides tools to stack a collection of
calibrated bias/dark/flat frames into a master.

Designing and implementing the higher lever interface is still pending. The high
level interface must:

    * automate the process

    * take its configuration and log progress and results to the database

    * provide crash recovery
