#!/usr/bin/env python3

"""Define class for collecting statistics and making a master photref."""

import os
import logging
from itertools import chain

import numpy
from astropy.io import fits
import zarr
from rechunker import rechunk

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

        _dimensions:    How many photometry measurements are being fit based on
            how many images, and what is the total number of coluns included in
            the statistics (usually twice the number of photometries).

        _source_name_format:    A %-substitution string for turning a source ID
            tuple to as string in the statistics file.

        _stat_quantities:    The quantities that ``grcollect`` is told to
            calculate for each input magnitude and error estimate.
    """

    _logger = logging.getLogger(__name__)
    _default_config = {
        "outlier_threshold": 5.0,
        "max_rejection_iterations": 10,
        "rejection_center": "median",
        "rejection_units": "medmeddev",
        "max_memory": "2g",
        "source_name_format": "{0:d}",
    }
    _stat_quantities = [
        "full_count",
        "rejected_count",
        "median",
        "mediandev",
        "medianmeddev",
    ]

    def _get_stat_dtype(self, catalog_columns):
        """Return the dtype to use for statistics."""

        special_dtypes = {"phqual": "S3", "magsrcflag": "S9"}
        return (
            [
                ("source_id", numpy.uint64),
            ]
            + [
                (
                    stat_quantity,
                    (
                        numpy.intc
                        if stat_quantity.endswith("count")
                        else numpy.float64
                    ),
                    (self._dimensions["columns"],),
                )
                for stat_quantity in self._stat_quantities
            ]
            + [
                (colname, special_dtypes.get(colname, numpy.float64))
                for colname in catalog_columns
            ]
        )

    def _calculate_statistics(self, statistics):
        """Calculate and fill in the statics from all input collected so far."""

        self._logger.debug("Planning the rechunking.")
        rechunked_store = zarr.TempStore()
        chunk_stars = self._config["max_memory"] // (
            self._dimensions["images"]
            * self._dimensions["columns"]
            * self._magfit_data.dtype.itemsize
        )
        chunk_stars = min(chunk_stars, self._sources.size)
        rechunk_plan = rechunk(
            self._magfit_data,
            (
                chunk_stars,
                self._dimensions["images"],
                self._dimensions["columns"],
            ),
            self._config["max_memory"],
            rechunked_store,
            temp_store=zarr.TempStore(),
        )
        self._logger.debug("Rechunking")
        rechunked_data = rechunk_plan.execute()
        self._logger.debug(
            "Calculating statistics for %d chunks of %d stars.",
            rechunked_data.cdata_shape[0],
            chunk_stars,
        )

        statistics["source_id"] = self._sources
        for block in range(rechunked_data.cdata_shape[0]):
            res_slice = slice(block * chunk_stars, (block + 1) * chunk_stars)
            statistics["full_count"][res_slice, :] = numpy.isfinite(
                rechunked_data.blocks[block][
                    :, :, : self._dimensions["columns"]
                ]
            ).sum(axis=1)
            (
                statistics["median"][res_slice, :],
                average_dev,
                statistics["rejected_count"][res_slice, :],
            ) = iterative_rejection_average(
                rechunked_data.blocks[block],
                self._config["outlier_threshold"],
                axis=1,
                average_func=getattr(
                    numpy, f"nan{self._config['rejection_center']}"
                ),
                deviation_average=(
                    (numpy.nanmean, numpy.nanmedian)
                    if self._config["rejection_units"] == "meddev"
                    else (numpy.nanmedian, numpy.nanmean)
                ),
            )
            if self._config["rejection_units"] == "meddev":
                (
                    statistics["mediandev"][res_slice, :],
                    statistics["medianmeddev"][res_slice, :],
                ) = average_dev
            else:
                assert self._config["rejection_units"] == "medmeddev"
                (
                    statistics["medianmeddev"][res_slice, :],
                    statistics["mediandev"][res_slice, :],
                ) = average_dev

            for column in ["mediandev", "medianmeddev"]:
                statistics[column] *= numpy.sqrt(
                    statistics["rejected_count"][res_slice, :] - 1
                )

    def _save_statistics(self, statistics):
        """Save the given statistics as a master statistics file."""

        save_column_fmt = "{stat_quantity}_{phot_quantity}_{phot_i:d}"
        save_dtype = [("source_id", numpy.dtype("u8"))] + [
            (
                save_column_fmt.format(
                    stat_quantity=stat_quantity,
                    phot_quantity=phot_quantity,
                    phot_i=phot_i,
                ),
                (int if stat_quantity.endswith("count") else float),
            )
            for phot_quantity in ["mag", "mag_err"]
            for phot_i in range(self._dimensions["photometries"])
            for stat_quantity in self._stat_quantities
        ]
        self._logger.debug("Dtype for saving: %s", repr(save_dtype))
        save_stat = numpy.empty(statistics.shape, dtype=save_dtype)
        save_fmt = (
            "%25d "
            + "%10d %10d %25.16e %25.16e %25.16e" * self._dimensions["columns"]
        )
        save_stat["source_id"] = statistics["source_id"]
        quantity_column = 0
        for phot_quantity in ["mag", "mag_err"]:
            for phot_i in range(self._dimensions["photometries"]):
                for stat_quantity in self._stat_quantities:
                    save_stat[
                        save_column_fmt.format(
                            stat_quantity=stat_quantity,
                            phot_quantity=phot_quantity,
                            phot_i=phot_i,
                        )
                    ] = statistics[stat_quantity][:, quantity_column]
                quantity_column += 1

        destination_dir = os.path.dirname(self._statistics_fname)
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        numpy.savetxt(self._statistics_fname, save_stat, fmt=save_fmt)
        with open(self._statistics_fname, "r", encoding="utf-8") as stat_file:
            self._logger.debug("Statistics file:\n%s", stat_file.read())

    def _add_catalog_info(self, catalog, statistics, catalog_columns):
        """Add the catalog data for each source to the statistics."""

        for source_index, source_id in enumerate(statistics["source_id"]):
            catalog_source = catalog[
                source_id if source_id.shape == () else tuple(source_id)
            ]
            for colname in catalog_columns:
                statistics[colname][source_index] = catalog_source[colname]

    def _get_enough_count_flags(self, statistics, min_nobs_median_fraction):
        """Return median number of observations per source in the stat. file."""

        count_col = (
            "rejected_count"
            if "rejected_count" in self._stat_quantities
            else "full_count"
        )
        min_counts = (
            numpy.median(statistics[count_col], axis=0)
            * min_nobs_median_fraction
        )
        return statistics[count_col] > min_counts[None, :]

    # Can't simplify further
    # pylint: disable=too-many-locals
    def _fit_scatter(
        self,
        statistics,
        fit_terms_expression,
        *,
        enough_counts_flags,
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
        residuals = numpy.empty(
            (statistics.size, self._dimensions["photometries"])
        )
        for phot_ind in range(self._dimensions["photometries"]):
            usable = numpy.isfinite(statistics[scatter_quantity][:, phot_ind])
            for column in predictors:
                usable = numpy.logical_and(usable, numpy.isfinite(column))
            usable = numpy.logical_and(enough_counts_flags[:, phot_ind], usable)
            phot_predictors = predictors[:, usable]
            target_values = numpy.log10(
                statistics[scatter_quantity][usable, phot_ind]
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
            self._logger.debug(
                "Calculating residual for scatter (%s):\n%s\n"
                "Fit scatter (%s):\n%s\n"
                "Predictors (%s):\n%s\n"
                "Coefficients (%s):\n%s",
                statistics[scatter_quantity][:, phot_ind].shape,
                repr(statistics[scatter_quantity][:, phot_ind]),
                numpy.dot(coefficients, predictors).shape,
                repr(numpy.dot(coefficients, predictors)),
                predictors.shape,
                repr(predictors),
                coefficients.shape,
                repr(coefficients),
            )
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
        enough_counts_flags,
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

            enough_counts_flags(bool array):    Flag for each photometry for
                each star indicating whether sufficient number of points
                remained after rejection to use in the master.

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
                getattr(numpy, "nan" + outlier_average)(
                    numpy.abs(residual_scatter[:, phot_ind])
                )
                * outlier_threshold
            )
            self._logger.debug("Max sctter allowed: %s", repr(max_scatter))
            self._logger.debug(
                "Sufficient observations flags: %s", enough_counts_flags
            )
            include_source = numpy.logical_and(
                enough_counts_flags[:, phot_ind],
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
                    ("full_count", numpy.uint64),
                    ("rejected_count", numpy.uint64),
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

        primary_hdu = fits.PrimaryHDU(header=primary_header)
        master_hdus = [
            fits.BinTableHDU(get_phot_reference_data(phot_ind))
            for phot_ind in range(self._dimensions["photometries"])
        ]
        fits.HDUList([primary_hdu] + master_hdus).writeto(reference_fname)

    def _init_data(self, num_sources):
        """Create the file-based array for holding magfit data."""

        dtype = numpy.dtype(numpy.float64)
        image_chunk = min(self._config["max_memory"], 100 * 1024**2) // (
            num_sources * self._dimensions["columns"] * dtype.itemsize
        )
        self._logger.debug(
            "Initializing zarr array with %d images per chunk.", image_chunk
        )
        self._magfit_data = zarr.create(
            shape=(
                num_sources,
                self._dimensions["images"],
                self._dimensions["columns"],
            ),
            chunks=(None, image_chunk, None),
            dtype=dtype,
            store=zarr.TempStore(),
            fill_value=numpy.nan,
        )
        self._logger.debug(
            "Zarr storage directory: %s",
            repr(self._magfit_data.store.dir_path()),
        )

    def __init__(
        self,
        statistics_fname,
        num_photometries,
        num_frames,
        temp_directory,
        **config,
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

        self._config = config
        for param, value in self._default_config.items():
            if param not in self._config:
                self._config[param] = value

        mem_unit = self._config["max_memory"][-1].lower()
        if mem_unit == "g":
            scale = 1024**3
        elif mem_unit == "m":
            scale = 1024**2
        elif mem_unit == "k":
            scale = 1024
        else:
            assert mem_unit == "b"
            scale = 1
        self._config["max_memory"] = (
            int(self._config["max_memory"][:-1]) * scale
        )

        self._magfit_data = None
        self._sources = None
        self._dimensions = {
            "photometries": num_photometries,
            "columns": 2 * num_photometries,
            "images": num_frames,
        }
        self._added_frames = 0
        self._statistics_fname = statistics_fname

    def add_input(self, fit_results):
        """Ingest a fitted frame's photometry into the statistics."""

        for phot, fitted in fit_results:

            formal_errors = phot["mag_err"][:, -1]
            phot_flags = phot["phot_flag"][:, -1]

            finite = True
            for column in chain(fitted.T, formal_errors.T):
                finite = numpy.logical_and(finite, numpy.isfinite(column))

            if phot is None or not finite.any():
                continue

            phot = phot[finite]
            formal_errors = formal_errors[finite]

            source_sorter = numpy.argsort(phot["source_id"])
            phot = phot[source_sorter]
            fitted = fitted[source_sorter]

            if self._magfit_data is None:
                assert self._sources is None
                self._sources = phot["source_id"]
                self._init_data(self._sources.size)
                source_indices = numpy.arange(self._sources.size)
            else:
                source_sorter = numpy.argsort(self._sources)
                source_indices = numpy.searchsorted(
                    self._sources, phot["source_id"], sorter=source_sorter
                )
                source_indices[source_indices == self._sources.size] = (
                    self._sources.size - 1
                )
                source_indices = source_sorter[source_indices]
                new_sources = numpy.nonzero(
                    self._sources[source_indices] != phot["source_id"]
                )[0]
                if new_sources.size:
                    source_indices[new_sources] = numpy.arange(
                        self._sources.size,
                        self._sources.size + new_sources.size,
                    )
                    self._sources = numpy.concatenate(
                        (self._sources, phot["source_id"][new_sources])
                    )
                    self._magfit_data.resize(
                        self._sources.size,
                        self._dimensions["images"],
                        self._dimensions["columns"],
                    )

            self._logger.debug(
                "Adding %d magfit sources for master.", fitted.shape[0]
            )

            self._magfit_data[
                source_indices, self._added_frames, : fitted.shape[1]
            ] = fitted
            self._magfit_data[
                source_indices, self._added_frames, fitted.shape[1] :
            ] = formal_errors
            self._added_frames += 1

            self._logger.debug("Finished adding fit data.")

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

        catalog_columns = next(iter(catalog.values())).dtype.names
        statistics = numpy.empty(
            self._sources.size, dtype=self._get_stat_dtype(catalog_columns)
        )

        self._calculate_statistics(statistics)
        self._add_catalog_info(catalog, statistics, catalog_columns)
        self._save_statistics(statistics)

        enough_counts_flags = self._get_enough_count_flags(
            statistics, min_nobs_median_fraction
        )
        residual_scatter = self._fit_scatter(
            statistics,
            fit_terms_expression,
            enough_counts_flags=enough_counts_flags,
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
            enough_counts_flags=enough_counts_flags,
            outlier_average=fit_outlier_average,
            outlier_threshold=fit_outlier_threshold,
            reference_fname=master_reference_fname,
            primary_header=primary_header,
        )
