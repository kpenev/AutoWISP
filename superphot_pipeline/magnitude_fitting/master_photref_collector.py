"""Define class for collecting statistics required to make master photref."""

import os
from subprocess import Popen, PIPE

import scipy

class MasterPhotrefCollector:
    """Class collecting magfit output to generate statics for master photref."""

    def __init__(self,
                 statistics_fname,
                 num_photometries,
                 temp_directory,
                 *,
                 output_lock=None,
                 outlier_threshold=5.0,
                 max_rejection_iterations=10,
                 rejection_center='median',
                 rejection_units='meddev',
                 max_memory='2g',
                 source_name_format='HAT-%03d-%07d'):
        """
        Prepare for collecting magfit results for master photref creation.

        Args:
            statistics_fname(str):    The filename where to save the statistics
                relevant for creating a master photometric reference.

            num_photometries(int):    The number of photometric measurements
                available for each star (e.g. number of apertures + 1 if psf
                fitting + ...).

            outlier_threshold(float):    A threshold value for outlier
                rejection. The units of this are determined by the
                ``rejection_units`` argument.

            max_rejection_iterations(int):    The maximum number of iterations
                between rejecting outliers and re-deriving the statistics to
                allow.

            temp_directory(str):    A location in the file system to use for
                storing temporary files during statistics colletion.

            rejection_center(str):    Outliers are define around some central
                value, either ``'mean'``, or ``'median'``.

            rejection_units(str):    The units of the outlier rejection
                threshold. One of ``'stddev'``, ``'meddev'``, or ``'absolute'``.

            max_memory(str):    The maximum amount of RAM the statistics process
                is allowed to use (if exceeded intermediate results are dumped
                to files in ``temp_dir``).

        Returns:
            None
        """

        grcollect_cmd = ['grcollect', '-', '-V', '--stat']
        stat_columns = range(1, 2 * num_photometries + 1)
        if outlier_threshold:
            grcollect_cmd.append(','.join(['count',
                                           'rcount',
                                           'rmedian',
                                           'rmediandev',
                                           'rmedianmeddev']))
            for col in stat_columns:
                grcollect_cmd.extend([
                    '--rejection',
                    (
                        'column=%d,iterations=%d,%s,%s=%f'
                        %
                        (
                            col,
                            max_rejection_iterations,
                            rejection_center,
                            rejection_units,
                            outlier_threshold
                        )
                    )
                ])
        else:
            grcollect_cmd.append('count,' + ','.join(['count',
                                                      'median',
                                                      'mediandev',
                                                      'medianmeddev']))
        grcollect_cmd.extend([
            '--col-base', '1',
            '--col-stat', ','.join([str(c) for c in stat_columns]),
            '--max-memory', max_memory,
            '--tmpdir', temp_directory,
            '--output', statistics_fname
        ])
        if not os.path.exists(temp_directory):
            os.makedirs(temp_directory)
        print("Starting grcollect command: '" + "' '".join(grcollect_cmd) + "'")
        self._grcollect = Popen(grcollect_cmd,
                                stdin=PIPE,
                                stdout=PIPE,
                                stderr=PIPE)
        self._output_lock = output_lock
        self._num_photometries = num_photometries
        self._source_name_format = source_name_format

    def add_input(self, phot, fitted):
        """Ingest a fitted frame's photometry into the statistics."""

        formal_errors = phot['mag_err'][:, -1]
        phot_flags = phot['phot_flag'][:, -1]

        src_count = formal_errors.shape[0]
        assert self._num_photometries == formal_errors.shape[1]
        assert formal_errors.shape == phot_flags.shape
        assert fitted.shape == formal_errors.shape

        line_format = (self._source_name_format
                       +
                       (' %9.5f') * (2 * self._num_photometries))
        if self._output_lock is not None:
            self._output_lock.acquire()
        for source_ind in range(src_count):
            print_args = (
                tuple(phot['ID'][source_ind][1:])
                +
                tuple(fitted[source_ind])
                +
                tuple(formal_errors[source_ind])
            )

            all_finite = True
            for value in print_args:
                if scipy.isnan(value):
                    all_finite = False
                    break

            if all_finite:
                self._grcollect.stdin.write(
                    (line_format % print_args + '\n').encode('ascii')
                )
        if self._output_lock is not None:
            self._output_lock.release()

    def calculate_statistics(self):
        """Finish the work of the object, creating the statics file."""

        grcollect_output, grcollect_error = self._grcollect.communicate()
        if self._grcollect.returncode:
            raise ChildProcessError(
                'grcollect command failed! '
                +
                'Return code: %d.' % self._grcollect.returncode
                +
                '\nOutput:\n' + grcollect_output.decode()
                +
                '\nError:\n' + grcollect_error.decode()
            )

    def __enter__(self):
        return self

    def __exit__(self, *args):

        self.calculate_statistics()
        return False
