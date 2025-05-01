#!/usr/bin/env python3

"""Define class for collecting statistics and making a master photref."""

import os
from subprocess import Popen, PIPE
import logging

import numpy
from astropy.io import fits

from autowisp.fit_expression import Interface as FitTermsInterface
from autowisp.fit_expression import iterative_fit
from autowisp.iterative_rejection_util import iterative_rejection_average


class MasterPhotrefCollector:
    """
    Class collecting magfit output to generate a master photometric reference.

    Attributes:
        _statistics_fname:    The file name to save the collected statistics
            under.

        _grcollect:    The running ``grcollect`` process responsible for
            generating the statistics.

        _num_photometries:    How many photometry measurements being fit.

        _source_name_format:    A %-substitution string for turning a source ID
            tuple to as string in the statistics file.

        _stat_quantities:    The quantities that ``grcollect`` is told to
            calculate for each input magnitude and error estimate.
    """

    _logger = logging.getLogger(__name__)

    def _report_grcollect_crash(self, grcollect_outerr=None):
        """Raise an error providing information on why grcollect crashed."""

        if grcollect_outerr is None:
            grcollect_outerr = self._grcollect.communicate()
        raise ChildProcessError(
            "grcollect command failed! "
            + f"Return code: {self._grcollect.returncode:d}."
            + "\nOutput:\n"
            + grcollect_outerr[0].decode()
            + "\nError:\n"
            + grcollect_outerr[1].decode()
        )

    def _calculate_statistics(self):
        """Creating the statics file from all input collected so far."""

        if self._grcollect.poll() is not None:
            self._report_grcollect_crash()
        grcollect_outerr = self._grcollect.communicate()
        if self._grcollect.returncode:
            self._report_grcollect_crash(grcollect_outerr)

    def _get_med_count(self):
        """Return median number of observations per source in the stat. file."""

        if "rcount" in self._stat_quantities:
            count_col = self._stat_quantities.index("rcount")
        else:
            count_col = self._stat_quantities.index("count")
        with open(self._statistics_fname, "r", encoding="ascii") as stat_file:
            med = numpy.median([float(l.split()[count_col]) for l in stat_file])
        return med

    def _read_statistics(self, catalog, parse_source_id):
        """
        Read the magnitude & scatter for each source for each photometry.

        Args:
            catalog(dict):    See ``master_catalog`` argument to
                MagnitudeFit.__init__().

            parse_source_id(callable):    See same name argument to
                generate_master().

        Returns:
            numpy structured array:
                The fields are as follows:

                    source_id: The ID of the source.

                    xi, eta: The projected angular coordinates of the source
                        from the catalog.

                    full_count: The full number of measurements available for
                        the sources

                    rejected_count: The number of measurements of the source
                        after outlier rejection during statistics collection.

                    mediandev: An estimate of the scatter for the source,
                        as the deviation around the median, calculated during
                        statistics collection.

                    medianmeddev: An estimate of the scatter for the source as
                        the median deviation around the median, calculated
                        during statistics collection.
        """

        def get_stat_data():
            """Read the statistics file."""

            column_names = ["source_id"]
            for phot_quantity in ["mag", "mag_err"]:
                for phot_ind in range(self._num_photometries):
                    column_names.extend(
                        [
                            f"{stat_quantity}_{phot_quantity}_{phot_ind:d}"
                            for stat_quantity in self._stat_quantities
                        ]
                    )
            return numpy.genfromtxt(
                self._statistics_fname, dtype=None, names=column_names
            )

        def create_result(num_sources, catalog_columns):
            """Create an empty result to fill with data."""

            special_dtypes = {"phqual": "S3", "magsrcflag": "S9"}
            example_source_id = next(iter(catalog.keys()))
            if isinstance(example_source_id, tuple):
                source_id_size = (len(example_source_id),)
            else:
                source_id_size = 1
            dtype = [
                ("source_id", numpy.uint64),
                ("full_count", numpy.intc, (self._num_photometries,)),
                ("rejected_count", numpy.intc, (self._num_photometries,)),
                ("median", numpy.float64, (self._num_photometries,)),
                ("mediandev", numpy.float64, (self._num_photometries,)),
                ("medianmeddev", numpy.float64, (self._num_photometries,)),
            ] + [
                (colname, special_dtypes.get(colname, numpy.float64))
                for colname in catalog_columns
            ]
            self._logger.debug("Stat result dtype: %s", repr(dtype))
            return numpy.empty(num_sources, dtype=dtype)

        def add_stat_data(stat_data, result):
            """Add the information from get_stat_data() to result."""

            print("Stat data columns: " + repr(stat_data.dtype.names))
            if stat_data["source_id"][0].dtype.kind in "OSU":
                for source_index, source_id in enumerate(
                    stat_data["source_id"]
                ):
                    result["source_id"][source_index] = parse_source_id(
                        source_id
                    )
            else:
                result["source_id"] = stat_data["source_id"]

            for phot_index in range(self._num_photometries):
                result["full_count"][:, phot_index] = stat_data[
                    f"count_mag_{phot_index:d}"
                ]
                result["rejected_count"][:, phot_index] = stat_data[
                    f"rcount_mag_{phot_index:d}"
                ]
                for statistic in ["median", "mediandev", "medianmeddev"]:
                    result[statistic][:, phot_index] = stat_data[
                        "r" + statistic + f"_mag_{phot_index:d}"
                    ]

        def add_catalog_info(catalog_columns, result):
            """Add the catalog data for each source to the result."""

            for source_index, source_id in enumerate(result["source_id"]):
                catalog_source = catalog[
                    source_id if source_id.shape == () else tuple(source_id)
                ]
                for colname in catalog_columns:
                    result[colname][source_index] = catalog_source[colname]

        catalog_columns = next(iter(catalog.values())).dtype.names
        stat_data = get_stat_data()
        self._logger.debug("Stat data:\n%s", repr(stat_data))
        result = create_result(stat_data.size, catalog_columns)
        add_stat_data(stat_data, result)
        self._logger.debug("Result without catalog:\n%s", repr(result))
        add_catalog_info(catalog_columns, result)

        return result

    # Can't simplify further
    # pylint: disable=too-many-locals
    @staticmethod
    def _fit_scatter(
        statistics,
        fit_terms_expression,
        *,
        min_counts,
        outlier_average,
        outlier_threshold,
        max_rej_iter,
        scatter_quantity="medianmeddev",
    ):
        """
        Fit for the dependence of scatter on source properties.

        Args:
            statistics:    The return value of _read_statistics().

            fit_terms_expression(str):    A fitting terms expression to use to
                generate the terms to include in the fit of the scatter.

            min_counts(int):    The smallest number of observations to require
                for a source to participate in the fit.

            outlier_average:    See ``fit_outlier_average`` argument to
                generate_master().

            outlier_threshold:    See ``fit_outlier_threshold`` argument to
                generate_master().

            max_rej_iter:    See ``fit_max_rej_iter`` argument to
                generate_master().

            scatter_quantity(str):    The name of the field in ``statistics``
                which contains the estimated scatter to fit.

        Returns:
            numpy array:
                The residuals of the scatter from ``statistics`` from the
                best-fit values found.
        """

        predictors = FitTermsInterface(fit_terms_expression)(statistics)
        num_photometries = statistics["full_count"][0].size
        residuals = numpy.empty((statistics.size, num_photometries))
        for phot_ind in range(num_photometries):
            enough_counts = (
                statistics["rejected_count"][:, phot_ind] >= min_counts
            )
            phot_predictors = predictors[:, enough_counts]
            target_values = numpy.log10(
                statistics[scatter_quantity][enough_counts, phot_ind]
            )
            coefficients = iterative_fit(
                phot_predictors,
                target_values,
                error_avg=outlier_average,
                rej_level=outlier_threshold,
                max_rej_iter=max_rej_iter,
                fit_identifier=(
                    "Generating master photometric reference, "
                    "photometry #" + repr(phot_ind)
                ),
            )[0]
            if coefficients is None:
                return None
            residuals[:, phot_ind] = numpy.log10(
                statistics[scatter_quantity][:, phot_ind]
            ) - numpy.dot(coefficients, predictors)
        return residuals

    # pylint: enable=too-many-locals

    def _create_reference(
        self,
        statistics,
        residual_scatter,
        *,
        min_counts,
        outlier_average,
        outlier_threshold,
        reference_fname,
        primary_header,
    ):
        """
        Create the master photometric reference.

        Args:
            statistics:    The return value of _read_statistics().

            residual_scatter:    The return value of _fit_scatter().

            min_counts(int):    The smallest number of observations to require
                for a source to be included in the refreence.

            outlier_average:    See ``fit_outlier_average`` argument to
                generate_master().

            outlier_threshold:    See ``fit_outlier_threshold`` argument to
                generate_master().

            reference_fname(str):    The name to use for the generated master
                photometric reference file.

            primary_header(fits.Header):    The header to use for the primary
                (non-table) HDU of the resulting master FITS file.

        Returns:
            None
        """

        def get_phot_reference_data(phot_ind):
            """
            Return the reference magnitude fit data as numpy structured array.
            """

            self._logger.info(
                "Generating master photometric reference for phot #%d", phot_ind
            )
            max_scatter = (
                getattr(numpy, outlier_average)(
                    numpy.abs(residual_scatter[:, phot_ind])
                )
                * outlier_threshold
            )
            self._logger.debug("Max sctter allowed: %s", repr(max_scatter))
            self._logger.debug("Min # observations allowed: %d", min_counts)
            include_source = numpy.logical_and(
                statistics["rejected_count"][:, phot_ind] >= min_counts,
                residual_scatter[:, phot_ind] <= max_scatter,
            )

            num_phot_sources = include_source.sum()
            self._logger.debug(
                "Suitable master photometric reference sources %d/%d",
                num_phot_sources,
                include_source.size,
            )

            column_map = [
                ("full_count", "full_count", phot_ind),
                ("rejected_count", "rejected_count", phot_ind),
                ("magnitude", "median", phot_ind),
                ("mediandev", "mediandev", phot_ind),
                ("medianmeddev", "medianmeddev", phot_ind),
            ]
            if statistics["source_id"][0].shape == ():
                result_dtype = [("source_id", "u8")]
            else:
                result_dtype = [
                    ("IDprefix", "i1"),
                    ("IDfield", numpy.intc),
                    ("IDsource", numpy.intc),
                ]
                column_map.extend(
                    [
                        ("IDprefix", "source_id", 0),
                        ("IDfield", "source_id", 1),
                        ("IDsource", "source_id", 2),
                    ]
                )

            result_dtype.extend(
                [
                    ("full_count", numpy.float64),
                    ("rejected_count", numpy.float64),
                    ("magnitude", numpy.float64),
                    ("mediandev", numpy.float64),
                    ("medianmeddev", numpy.float64),
                    ("scatter_excess", numpy.float64),
                ]
            )

            reference_data = numpy.empty(num_phot_sources, dtype=result_dtype)

            if statistics["source_id"][0].shape == ():
                reference_data["source_id"] = statistics["source_id"][
                    include_source,
                ]

            for reference_column, stat_column, stat_index in column_map:
                reference_data[reference_column] = statistics[stat_column][
                    include_source, stat_index
                ]
            reference_data["scatter_excess"] = residual_scatter[
                include_source, phot_ind
            ]
            return reference_data

        num_photometries = statistics["full_count"][0].size
        primary_hdu = fits.PrimaryHDU(header=primary_header)
        master_hdus = [
            fits.BinTableHDU(get_phot_reference_data(phot_ind))
            for phot_ind in range(num_photometries)
        ]
        fits.HDUList([primary_hdu] + master_hdus).writeto(reference_fname)

    # Could not refactor to simply.
    # pylint: disable=too-many-locals
    def __init__(
        self,
        statistics_fname,
        num_photometries,
        temp_directory,
        *,
        outlier_threshold=5.0,
        max_rejection_iterations=10,
        rejection_center="median",
        rejection_units="meddev",
        max_memory="2g",
        source_name_format="{0:d}",
    ):
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

        grcollect_cmd = ["grcollect", "-", "-V", "--stat"]
        stat_columns = range(2, 2 * num_photometries + 2)
        self._num_photometries = num_photometries
        self._stat_quantities = [
            "count",
            "count",
            "median",
            "mediandev",
            "medianmeddev",
        ]
        if outlier_threshold:
            for i in range(1, len(self._stat_quantities)):
                self._stat_quantities[i] = "r" + self._stat_quantities[i]
            grcollect_cmd.append(",".join(self._stat_quantities))
            for col in stat_columns:
                grcollect_cmd.extend(
                    [
                        "--rejection",
                        (
                            f"column={col:d},iterations="
                            f"{max_rejection_iterations:d},{rejection_center!s},"
                            f"{rejection_units!s}={outlier_threshold:f}"
                        ),
                    ]
                )
        else:
            grcollect_cmd.append(",".join(self._stat_quantities))

        self._statistics_fname = statistics_fname

        grcollect_cmd.extend(
            [
                "--col-base",
                "1",
                "--col-stat",
                ",".join([str(c) for c in stat_columns]),
                "--max-memory",
                max_memory,
                "--tmpdir",
                temp_directory,
                "--output",
                statistics_fname,
            ]
        )
        if not os.path.exists(temp_directory):
            os.makedirs(temp_directory)
        print("Starting grcollect command: '" + "' '".join(grcollect_cmd) + "'")
        # Needs to persist after function exits
        # pylint: disable=consider-using-with
        self._grcollect = Popen(
            grcollect_cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE
        )
        # pylint: enable=consider-using-with

        self._num_photometries = num_photometries
        self._source_name_format = source_name_format
        self._line_format = "%s" + (" %9.5f") * (2 * self._num_photometries)

    # pylint: enable=too-many-locals

    def add_input(self, fit_results):
        """Ingest a fitted frame's photometry into the statistics."""

        logger = logging.getLogger(__name__)
        for phot, fitted in fit_results:
            if phot is None:
                continue

            logger.debug(
                "Collecting %d magfit sources for master.",
                len(phot["source_id"]),
            )

            logger.debug(
                "Return code of grcollect: %s", repr(self._grcollect.poll())
            )
            assert self._grcollect.poll() is None

            formal_errors = phot["mag_err"][:, -1]
            phot_flags = phot["phot_flag"][:, -1]

            src_count = formal_errors.shape[0]
            assert self._num_photometries == formal_errors.shape[1]
            assert formal_errors.shape == phot_flags.shape
            assert fitted.shape == formal_errors.shape

            for source_ind in range(src_count):
                print_args = (
                    (
                        self._source_name_format.format(
                            phot["source_id"][source_ind]
                        ),
                    )
                    + tuple(fitted[source_ind])
                    + tuple(formal_errors[source_ind])
                )

                all_finite = True
                for value in print_args[1:]:
                    if numpy.isnan(value):
                        all_finite = False
                        break

                if all_finite:
                    self._grcollect.stdin.write(
                        (self._line_format % print_args + "\n").encode("ascii")
                    )
            logger.debug("Finished sending fit data to grcollect")

    # TODO: Add support for scatter cut based on quantile of fit residuals.
    def generate_master(
        self,
        *,
        master_reference_fname,
        catalog,
        fit_terms_expression,
        parse_source_id,
        min_nobs_median_fraction=0.5,
        fit_outlier_average="median",
        fit_outlier_threshold=3.0,
        fit_max_rej_iter=20,
        extra_header=None,
    ):
        """
        Finish the work of the object and generate a master.

        Args:
            master_reference_fname(str):   The file name to use for the newly
                created master photometric reference.

            catalog:     See ``master_catalog`` argument to
                MagnitudeFit.__init__().

            fit_terms_expression(str):    An expression expanding to the terms
                to include in the scatter fit. May use any catalog column.

            parse_source_id(callable):    Should convert a string source ID into
                the source ID format expected by the catalog.

            min_nobs_median_fraction(float):    The minimum fraction of the
                median number of observations a source must have to be inclruded
                in the master.

            fit_outlier_average(str):    The averaging method to use for scaling
                averaging residuals from scatter fit. The result is used as the
                unit for ``fit_outlier_threshold``.

            fit_outlier_threshold(float):    A factor to multiply the
                ``fit_outlier_average`` averaged residual from scatter fit in
                order to get the threshold to consider a scatter point an
                outlier, and hence discard from cantributing to the reference.

            fit_max_rej_iter(int):    The maximum number of iterations to allow
                for fitting/rejecting outliers. If this number is reached, the
                last result is accepted.

            extra_header(None or dict-like):    Header keywords to add to the
                generated FITS header in addition to the ones describing the
                master fit.

        Returns:
            None
        """

        self._calculate_statistics()
        statistics = self._read_statistics(catalog, parse_source_id)
        min_counts = min_nobs_median_fraction * self._get_med_count()
        residual_scatter = self._fit_scatter(
            statistics,
            fit_terms_expression,
            min_counts=min_counts,
            outlier_average=fit_outlier_average,
            outlier_threshold=fit_outlier_threshold,
            max_rej_iter=fit_max_rej_iter,
        )
        if residual_scatter is None:
            raise RuntimeError(
                "Failed to generate master photometric reference: %s",
                repr(master_reference_fname),
            )
        primary_header = fits.Header()
        if extra_header is not None:
            primary_header.update(extra_header)
        primary_header["FITTERMS"] = fit_terms_expression
        primary_header["MINOBSFR"] = min_nobs_median_fraction
        primary_header["OUTL_AVG"] = fit_outlier_average
        primary_header["OUTL_THR"] = fit_outlier_threshold
        primary_header["MAXREJIT"] = fit_max_rej_iter
        self._create_reference(
            statistics=statistics,
            residual_scatter=residual_scatter,
            min_counts=min_counts,
            outlier_average=fit_outlier_average,
            outlier_threshold=fit_outlier_threshold,
            reference_fname=master_reference_fname,
            primary_header=primary_header,
        )


if __name__ == "__main__":
    from time import time

    import zarr
    from rechunker import rechunk

    nstars = 5000
    nframes = 500
    nphot = 50
    max_mem = 2 * 1024**2
    dtype = numpy.dtype(float)
    star_chunk = int(max_mem / (nframes * nphot * dtype.itemsize))
    frame_chunk = int(max_mem / (nstars * nphot * dtype.itemsize))
    print(f"Using chunk size: frame: {frame_chunk}, star: {star_chunk}")
    arr = zarr.empty(
        (nstars, nframes, nphot),
        chunks=(None, frame_chunk, None),
        dtype=dtype,
        store=zarr.TempStore(),
    )
    print(f"Stored at: {arr.store.dir_path()}")
    print("Setting random values")
    start = time()
    for frame_i in range(nframes):
        arr[:, frame_i, :] = numpy.random.rand(nstars, nphot)
        if frame_i % 100 == 0:
            print(f"Progress {frame_i + 1}/{nframes}")

    print(f"Setting took {time() - start} sec")

    print("Rechunking")
    start = time()
    rechunked_store = zarr.TempStore()
    rechunk_plan = rechunk(
        arr,
        (star_chunk, nframes, nphot),
        "2MB",
        rechunked_store,
        temp_store=zarr.TempStore(),
    )
    print(f"Rechunk plan: {rechunk_plan}")
    arr = rechunk_plan.execute()
    print(f"Rechunking took {time() - start} sec")

    # for b in range(arr.cdata_shape[0]):
    #    arr.blocks[b] = numpy.random.rand(
    #        min(nstars - b * chunk_size, chunk_size),
    #        nframes,
    #        nphot
    #    )

    print("Calculating median")
    start = time()
    results = {
        stat: numpy.empty(
            (nstars, nphot), dtype=int if stat == "count" else float
        )
        for stat in ["med", "stddev", "count"]
    }

    for b in range(arr.cdata_shape[0]):

        (
            results["med"][
                b * star_chunk : min(nstars, (b + 1) * star_chunk), :
            ],
            results["stddev"][
                b * star_chunk : min(nstars, (b + 1) * star_chunk), :
            ],
            results["count"][
                b * star_chunk : min(nstars, (b + 1) * star_chunk), :
            ],
        ) = iterative_rejection_average(arr.blocks[b], 3.0, axis=1)
        print(f"Progress {b + 1}/{arr.cdata_shape[0]}")
    print(f"Median took {time() - start} sec.")
    for stat, values in results.items():
        print(f"{stat}: {values!r}\n")
