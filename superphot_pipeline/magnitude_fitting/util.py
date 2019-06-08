"""Functions for creating and working with photometric renfereneces."""

from tempfile import TemporaryDirectory
from multiprocessing import Pool, Lock
import logging

import numpy
from numpy.lib import recfunctions
from astropy.io import fits

from superphot_pipeline import DataReductionFile

from superphot_pipeline.magnitude_fitting import\
    LinearMagnitudeFit,\
    MasterPhotrefCollector

_logger = logging.getLogger(__name__)

def get_single_photref(data_reduction_file, **path_substitutions):
    """Create a photometric reference out of the raw photometry in a DR file."""

    source_data = data_reduction_file.get_source_data(
        magfit_iterations=[0],
        shape_map_variables=False,
        string_source_ids=False,
        **path_substitutions
    )
    return {tuple(source['ID']): dict(x=source['x'],
                                      y=source['y'],
                                      mag=source['mag'],
                                      mag_err=source['mag_err'])
            for source in source_data}

def get_master_photref(photref_fname):
    """Read a FITS photometric reference created by MasterPhotrefCollector."""

    result = dict()
    with fits.open(photref_fname, 'readonly') as photref_fits:
        num_photometries = len(photref_fits) - 1
        for phot_ind, phot_reference in enumerate(photref_fits[1:]):
            for source_index, source_id in enumerate(
                    zip(phot_reference.data['IDprefix'].astype('int'),
                        phot_reference.data['IDfield'],
                        phot_reference.data['IDsource'])
            ):
                if source_id not in result:
                    result[source_id] = dict(
                        mag=numpy.full((1, num_photometries),
                                       numpy.nan,
                                       numpy.float64),
                        mag_err=numpy.full((1, num_photometries),
                                           numpy.nan,
                                           numpy.float64)
                    )
                result[source_id]['mag'][0, phot_ind] = phot_reference.data[
                    'magnitude'
                ][
                    source_index
                ]
                result[source_id]['mag_err'][0, phot_ind] = phot_reference.data[
                    'mediandev'
                ][
                    source_index
                ]
    return result

def read_master_catalogue(fname, source_id_parser):
    """Return the catalogue info in the given file formatted for magfitting."""

    data = numpy.genfromtxt(fname, dtype=None, names=True, deletechars='')
    data.dtype.names = [colname.split('[', 1)[0]
                        for colname in data.dtype.names]
    catalogue_sources = data['ID']
    data = recfunctions.drop_fields(data, 'ID', usemask=False)
    return {source_id_parser(source_id): source_data
            for source_id, source_data in zip(catalogue_sources, data)}

#Could not come up with a sensible way to simplify
#pylint: disable=too-many-locals
def iterative_refit(fit_dr_filenames,
                    *,
                    single_photref_dr_fname,
                    master_catalogue_fname,
                    configuration,
                    master_photref_fname_pattern,
                    magfit_stat_fname_pattern,
                    max_iterations=5,
                    **path_substitutions):
    """
    Iteratively performa magnitude fitting/generating master until convergence.

    Args:
        fit_dr_filenames(str iterable):    A list of the data reduction files to
            fit.

        single_photref_dr_fname(str):    The name of the data reduction file of
            the single photometric reference to use to start the magnitude
            fitting iterations.

        master_catalogue_fname(str):    The name of the catalogue file to use as
            extra information in magnitude fitting terms and for excluding
            sources from the fit.

        configuration:    Passed directly as the config argument to
            LinearMagnitudeFit.__init__() but it must also contain the following
            attributes:

                * num_parallel_processes(int): the the maximum number of
                  magnitude fitting parallel processes to use.

                * max_photref_change(float): the maximum square average change
                  of photometric reference magnitudes to consider the iterations
                  converged.

        master_photref_fname_pattern(str):    A %-substitution pattern involving
            a %(magfit_iteration)s substitution along with any variables passed
            through the path_substitutions arguments, that expands to the name of
            the file to save the master photometric reference for a particular
            iteration.

        magfit_stat_fname_pattern(str):    Similar to
            ``master_photref_fname_pattern``, but defines the name to use for
            saving the statistics of a magnitude fitting iteration.

        num_parallel_processes(int):     How many processes to use for
            simultaneus fitting.

        max_iterations(int):    The maximum number of iterations of deriving a
            master and re-fitting to allow.

        path_substitutions:     Any variables to substitute in
            ``master_photref_fname_pattern`` or to pass to data reduction files
            to identify components to use in the fit.

    Returns:
        None
    """

    def update_photref(magfit_stat_collector,
                       old_reference,
                       source_id_parser,
                       num_photometries):
        """
        Return the next iteration photometric reference or None if converged.

        Args:
            magfit_stat_collector(MasterPhotrefCollector):    The object used by
                the magnitude fitting processes to generate the magnitude
                fitting statistics.

            old_reference(dict):    The photometric reference used for the last
                magnitude fitting iteration.

            source_id_parser(callable):    Should return the integers
                identifying a source, given its string ID.

            num_photometries(int):    How many different photometric
                measurements are being fit.
        """

        master_reference_fname = (master_photref_fname_pattern
                                  %
                                  path_substitutions)
        magfit_stat_collector.generate_master(
            master_reference_fname=master_reference_fname,
            catalogue=catalogue,
            fit_terms_expression='O2{r,xi,eta}',
            parse_source_id=source_id_parser
        )
        new_reference = get_master_photref(master_reference_fname)

        common_sources = set(new_reference) & set(old_reference)

        average_square_change = numpy.zeros(num_photometries,
                                            dtype=numpy.float64)
        num_finite = numpy.zeros(num_photometries, dtype=numpy.float64)
        for source in common_sources:
            square_diff = (old_reference[source]['mag'][0]
                           -
                           new_reference[source]['mag'][0])**2
            finite_entries = numpy.isfinite(square_diff)
            print('Num photometries: ' + repr(num_photometries))
            print('square_diff (shape=%s): ' % repr(square_diff.shape)
                  +
                  repr(square_diff))
            print('finite_entries (shape=%s): ' % repr(finite_entries.shape)
                  +
                  repr(finite_entries))
            print('average_square_change (shape=%s): '
                  %
                  repr(average_square_change.shape)
                  +
                  repr(average_square_change))

            average_square_change[finite_entries] += square_diff[finite_entries]
            num_finite += finite_entries

        average_square_change /= num_finite
        _logger.debug(
            'Fit iteration resulted in average square change in magnitudes of: '
            '%s',
            repr(average_square_change)
        )

        if average_square_change.max() <= configuration.max_photref_change:
            return None

        return new_reference

    photref_dr = DataReductionFile(single_photref_dr_fname, 'r')
    photref = get_single_photref(photref_dr, **path_substitutions)
    photref_dr.close()

    catalogue = read_master_catalogue(master_catalogue_fname,
                                      photref_dr.parse_hat_source_id)
    path_substitutions['magfit_iteration'] = 0

    num_photometries = next(iter(photref.values()))['mag'].size

    with TemporaryDirectory() as grcollect_tmp_dir:
        while (
                photref
                and
                path_substitutions['magfit_iteration'] <= max_iterations
        ):
            assert next(iter(photref.values()))['mag'].size == num_photometries

            magfit_stat_collector = MasterPhotrefCollector(
                magfit_stat_fname_pattern % path_substitutions,
                num_photometries,
                grcollect_tmp_dir,
                output_lock=(
                    Lock() if configuration.num_parallel_processes > 1
                    else None
                )
            )
            magfit = LinearMagnitudeFit(config=configuration,
                                        reference=photref,
                                        master_catalogue=catalogue,
                                        magfit_collector=magfit_stat_collector)
            if configuration.num_parallel_processes > 1:
                with Pool(configuration.num_parallel_processes) as magfit_pool:
                    magfit_pool.map(
                        lambda dr_fname: magfit(dr_fname, **path_substitutions),
                        fit_dr_filenames
                    )
            else:
                for dr_fname in fit_dr_filenames:
                    magfit(dr_fname, **path_substitutions)

            photref = update_photref(magfit_stat_collector,
                                     photref,
                                     photref_dr.parse_hat_source_id,
                                     num_photometries)
            path_substitutions['magfit_iteration'] += 1
#pylint: enable=too-many-locals
