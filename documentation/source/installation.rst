Installation
============

Use the `pip package manager <https://pip.pypa.io/en/stable/>`_ to install
**AutoWISP**::

    pip install autowisp

Run the Tests (optional)
------------------------

Autowisp comes with a set of tests that apply all processing steps to a small
set images and check that the results are as expected. To run the tests::

    python3 -m autowisp.tests failed_test -vvvv

The `failed_test` argument above tells AutoWISP test suite to save a test which
fails to a directory with that name. If no tests fail, no such directory should
exist.

It may take a while to run all the tests, so please be patient. In particular,
unless you are using a local installation of astrometry.net, finding a function
that allows converting sky coordinates to coordinates within each image uses a
web interface that only allows submitting one image at a time per user and can
take a minute or more to return a solution. 

