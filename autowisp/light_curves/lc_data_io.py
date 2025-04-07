# pylint: disable=too-many-lines
"""Define a class for collecting LC data from DR files."""

import re
from multiprocessing import Value
from traceback import format_exception
import sys
import os
from os.path import dirname, exists
import itertools
import logging
from ctypes import memset, sizeof, c_char_p

import h5py
import numpy
from astropy import units
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

from autowisp.magnitude_fitting.util import format_master_catalog
from autowisp import DataReductionFile
from autowisp.miscellaneous import get_hat_source_id_str
from autowisp.data_reduction.utils import get_source_extracted_psf_map

from .light_curve_file import LightCurveFile, _config_dset_key_rex
from .lc_data_slice import create_lc_data_slice_type
from .hashable_array import HashableArray


# TODO: Add catalogue information as top-level attributes.
# TODO: Add xi and eta as config datasets.
class LCDataIO:
    """
    A callable class which gathers a slice of LC data from frames/DR files.

    Suitable for multiprocessing.

    Two instances of this class should never be created simultaneously!

    Attributes:
        %(something)s_config:    Attributes containing the configurations for
            the various components for which configuration was defined in the
            processed frames and values which are in turn dictionaries indexed
            by the configurations found for this component as frozenset suitable
            for directly passing to LightCurve.add_configuration and values
            giving the number assigned to those configurations in the
            LCDataSlice structure that was filled.

        lc_data_slice (LCDataSlice):    An object which will be filled with
            data as it is being read from data reduction files.


        dataset_dimensions (dict):    Identifiers for the dimensions of
            datasets. The keys are the pipeline keys of lightcurve datasets and
            the values are tuples containing some of: `'frame'`,
            `'source'`, `'srcproj_column_name'`, `'aperture_index'`,
            `'magfit_iteration'`, `'srcextract_psf_param'` identifying what the
            entries in the dataset depend on. It also contains a special dataset
            `'source_in_frame'` which indicates which sources have observations
            in which frames.

        header_datasets (dict):    The set of datasets which contain FITS header
            keywords. The index is the pipeline key identifying the dataset, and
            the value is the corresponding header keyword.

        max_dimension_size (dict):    A dictionary with keys the various
            dimensions on which datasets can depend (see dataset_dimensions) and
            entries the maximum size for each dimension.

        source_destinations(dict):    Key: (field, source). Value: the offset of
            the given source in the LCDataSlice arrays that are per-source.

        config_components(dict):    Keys: all configuration components that can
            be defined. Values: 2-tuple:

                * a tuple of the keywords required to resolve the path to the
                  configuration index dataset for this component for which all
                  values found must be dumped.

                * a set of the datasets belonging to this component

        [_ra_dec]:    Optional. Only exists if one of BJD, Z or HA quantities
            are evaluated per source instead of using the values for the frame
            center.

        _config:    The configuration of how to perform the LC dumping.

        _path_substitutions:    See path_substitutions argument to create().

        _organized_config(dict):    The keys are configuration components and
            the values are dictionaries with keys the coordinates along each
            dimension of the configuration index dataset and values lists of the
            configurations for the component with indices in the list
            corresponding to the entries added to the config ID columns in
            ReadLCData.lc_data_slice. Gets initialized by prepare_for_writing().
    """

    _logger = logging.getLogger(__name__)
    dataset_dimensions = {}
    header_datasets = {}
    config_components = {}
    max_dimension_size = {}
    _catalogue = {}
    _ra_dec = []
    _path_substitutions = {}
    _multivalued_entry_datasets = ["sky_coord"]
    cfg_index_id = "cfg_index"
    _organized_config = {}

    # TODO: perhaps worth simplifying later.
    # pylint: disable=too-many-statements
    @classmethod
    def _classify_datasets(cls, lc_example, ignore_splits):
        """
        Set dataset_dimensions, header_datasets, & config_components attributes.

        Args:
            lc_example(LightCurveFile):    A fully configured instance of
                LightCurveFile to serve as an example of the structure of
                lightcurve files to expect.

            ignore_splits:    See path_substitutions argument of create().

        """

        # TODO: perhaps worth simplifying later.
        # pylint: disable=too-many-branches
        def organize_datasets():
            """Set dataset_dimensions and header_datasets attributes."""

            config_datasets = {}
            config_components = {}

            substitution_rex = re.compile(r".*?%[(](?P<substitution>.*?)[)]")
            ignore_rex = re.compile(r"\.epd\.|\.tfa\.")
            key_rex = {
                "config": _config_dset_key_rex,
                "perframe": re.compile(
                    "|".join(
                        [
                            r"skytoframe\.(sky_center|residual|unitarity)$",
                            r"^shapefit\.global_chi2$",
                            r"magfit\.(num_input_src|num_fit_src|fit_residual)$",
                            r"^fitsheader\.(?!cfg\.)",
                            r"\.cfg_index$",
                            r"^catalogue\.cfg\.(epoch|fov|orientation)$",
                            r"^srcextract.psf_map.(residual|num_fit_src)$",
                        ]
                    )
                ),
                "persource": re.compile(
                    "|".join(
                        [
                            r"^srcextract\.psf_map\.eval",
                            r"^srcproj\.columns$",
                            r"^bg\.(value|error|npix)$",
                            r"^shapefit\.(chi2|num_pixels|signal_to_noise)$",
                            r"\.(magnitude|magnitude_error|quality_flag)$",
                            r"skypos\..*",
                        ]
                    )
                ),
            }

            for lc_quantity in lc_example.elements["dataset"]:
                cls._logger.debug(
                    "Organizing LC quantity: %s", repr(lc_quantity)
                )
                if ignore_rex.search(lc_quantity):
                    cls._logger.debug("Skipping: %s", repr(lc_quantity))
                    continue

                split_quantity = lc_quantity.split(".")
                if split_quantity[0] == "fitsheader":
                    cls.header_datasets[lc_quantity] = split_quantity[-1]

                path_template = lc_example.get_element_path(lc_quantity)
                parent = path_template.rsplit("/", 1)[0]
                dimensions = substitution_rex.findall(path_template)
                for split in ignore_splits:
                    try:
                        dimensions.remove(split)
                    except ValueError:
                        pass

                dimensions = tuple(sorted(set(dimensions)))

                found_match = False
                for key_type, type_rex in key_rex.items():
                    if type_rex.search(lc_quantity):
                        assert not found_match
                        found_match = True
                        if key_type == "config":
                            if parent not in config_datasets:
                                config_datasets[parent] = set()
                            config_datasets[parent].add(lc_quantity)
                        else:
                            dimensions += ("frame",)
                        if key_type == "persource":
                            dimensions += ("source",)

                if lc_quantity in [
                    "skytoframe.cfg.frame_center",
                    "skytoframe.sky_center",
                ]:
                    dimensions += ("sky_coord",)

                cls._logger.debug(
                    "Dimensions for %s: %s", repr(lc_quantity), repr(dimensions)
                )
                cls.dataset_dimensions[lc_quantity] = dimensions

                if lc_quantity.endswith("." + cls.cfg_index_id):
                    cls._logger.debug(
                        "%s is configuration quantity.", repr(lc_quantity)
                    )

                    config_group = parent + "/Configuration"
                    assert config_group not in config_components
                    config_components[config_group] = lc_quantity[
                        : -(len(cls.cfg_index_id) + 1)
                    ]
            cls.dataset_dimensions["source_in_frame"] = ("frame", "source")
            return config_datasets, config_components

        # pylint: enable=too-many-branches

        def organize_config(config_datasets, config_components):
            """Fill the :attr:`config_components` attribute."""

            for parent, component in config_components.items():
                component_dsets = config_datasets[parent]
                dimensions = cls.dataset_dimensions[next(iter(component_dsets))]
                if (
                    dimensions
                    and dimensions[-1] in cls._multivalued_entry_datasets
                ):
                    dimensions = dimensions[:-1]
                for dset in component_dsets:
                    dset_dimensions = cls.dataset_dimensions[dset]
                    if dset_dimensions and dset_dimensions[-1] in (
                        cls._multivalued_entry_datasets
                    ):
                        dset_dimensions = dset_dimensions[:-1]
                    assert dset_dimensions == dimensions
                assert cls.dataset_dimensions[
                    component + "." + cls.cfg_index_id
                ] == dimensions + ("frame",)
                cls.config_components[component] = (dimensions, component_dsets)

        organize_config(*organize_datasets())

    # pylint: enable=too-many-statements

    @classmethod
    def create(
        cls,
        *,
        catalog_sources,
        config,
        source_id_parser,
        dr_fname_parser,
        source_list=None,
        optional_header=None,
        observatory=None,
        **path_substitutions,
    ):
        """
        Configure the class for use in multiprocessing LC collection.

        Args:
            config:    The configuratoin of how to generate the lightcurves,
                usually as parsed from the command line. In the latter case, the
                configuration should contain the following attributes:

                    - max_apertures: The maximum number of photometriec
                      apertures in any input frame.

                    - num_magfit_iterations: The maximum number of magnitude
                      fitting iterations in any input frame.

                    - lcdump_catalogue: The filename of a catalogue file
                      containing at least RA and Dec.

                    - single_photref_dr_fname: The filename of the single
                      photometric reference used for magfit of input images.

                    - srcextract_psf_params: List of the parameters describing
                      PSF shapes of the extracted sources.

                    - srcproj_column_names: List of the projected source columns
                      to transfer to the collected lightcurves.

                    - memblocksize: The maximum amount of memory (in bytes) to
                      allocate for temporaririly storing source information
                      before dumping to LC.

            source_id_parser:    A callable that can convert string source IDs
                to the corresponding tuple of integers source IDs.

            dr_fname_parser:    A callable that parser the filename of DR files
                returning extra keywrds to add to the header.

            source_list:    A list that includes all sources for which
                lightcurves will be generated. Sources not in this list are
                ignored. If None, the sources in the catalogue are used instead.

            optional_header: Indicate which header keywords could
                be missing from the FITS header. Missing keywors will
                be replaced by appropriate values indicating an unknown
                value. Should be in one of 3 formats:

                    - `None`: all expected header keywords must be present
                      in every frame.

                    - `'all'`: No header keywords are required.

                    - iterable of strings: to be querried by the python
                      `in` keyword.

            observatory(None or dict):    If specified, should be a dictionary
                with keys `'SITELAT'`, `'SITELONG'`, and `'SITEALT'` which
                overwrite or add the header information for the observatory's
                location on earth.

            path_substitutions:    Path substitutions to be kept fixed during
                the entire lightcurve dumping process. Used to resolve paths
                both within the input data reduction files and with the
                generated lightcurves.

        Returns:
            None
        """

        if hasattr(cls, "lc_data_slice"):
            del cls.lc_data_slice

        cls._path_substitutions = path_substitutions

        cls._catalogue = format_master_catalog(
            catalog_sources, source_id_parser
        )
        cls.dr_fname_parser = staticmethod(dr_fname_parser)

        if source_list is None:
            source_list = list(cls._catalogue.keys())

        cls._logger.debug(
            "Creating LC Data IO with source list: %s", repr(source_list)
        )

        with LightCurveFile() as no_light_curve:
            num_sources = len(source_list)

            cls.source_destinations = {
                source: index for index, source in enumerate(source_list)
            }
            cls.max_dimension_size = {
                "source": num_sources,
                "aperture_index": config["max_apertures"],
                "srcextract_psf_param": len(config["srcextract_psf_params"]),
                "srcproj_column_name": len(config["srcproj_column_names"]),
                "sky_coord": 2,
            }
            if config["num_magfit_iterations"]:
                cls.max_dimension_size["magfit_iteration"] = config[
                    "num_magfit_iterations"
                ]

            cls._classify_datasets(no_light_curve, path_substitutions.keys())

            (LCDataSlice, cls.max_dimension_size["frame"]) = (
                create_lc_data_slice_type(
                    get_dtype=no_light_curve.get_dtype,
                    dataset_dimensions=cls.dataset_dimensions,
                    max_dimension_size=cls.max_dimension_size,
                    max_mem=config["max_memory"],
                )
            )

        cls.lc_data_slice = Value(LCDataSlice, lock=False)
        cls._config = config
        cls._optional_header = optional_header
        cls._observatory = observatory

        cls._ra_dec = numpy.empty(shape=(2, num_sources), dtype=numpy.float64)
        for src_index, src in enumerate(source_list):
            # False positve
            # pylint: disable=invalid-sequence-index
            cls._ra_dec[:, src_index] = (
                cls._catalogue[src]["RA"],
                cls._catalogue[src]["Dec"],
            )
            # pylint: enable=invalid-sequence-index

        cls._logger.debug(
            "Max dimension size: %s", repr(cls.max_dimension_size)
        )

        return cls()

    # Handling the many branches is exactly the point.
    # pylint: disable=too-many-branches
    @classmethod
    def _config_to_lc_format(cls, lc_quantity, lc_dtype, value):
        """Return value as it would be read from the LC."""

        if lc_dtype is None:
            cls._logger.debug(
                "Not changing dtype of %s = %s", repr(lc_quantity), repr(value)
            )
            lc_dtype = value.dtype
        try:
            try:
                if not numpy.isfinite(value):
                    value = "NaN"
            except ValueError:
                assert numpy.isfinite(value).all()
            except TypeError:
                pass
            if value is None:
                if numpy.dtype(lc_dtype).kind == "b":
                    result = False
                elif numpy.dtype(lc_dtype).kind == "i":
                    result = numpy.iinfo(lc_dtype).min
                elif numpy.dtype(lc_dtype).kind == "u":
                    result = numpy.iinfo(lc_dtype).max
                elif numpy.dtype(lc_dtype).kind == "f":
                    result = b"NaN"
                elif (
                    numpy.dtype(lc_dtype).kind == "S"
                    or h5py.check_dtype(vlen=numpy.dtype(lc_dtype)) is str
                ):
                    result = b""
                else:
                    assert False
            elif numpy.dtype(lc_dtype).kind == "S":
                if isinstance(value, str):
                    return value.encode("ascii")
                if value.size > 1:
                    return HashableArray(value)
                return value
            elif isinstance(value, str) and (value == "NaN"):
                return value.encode("ascii")
            else:
                lc_dtype = numpy.dtype(lc_dtype)
                vlen_type = h5py.check_dtype(vlen=lc_dtype)
                if vlen_type is str:
                    result = str(value).encode("ascii")
                elif vlen_type is None:
                    result = lc_dtype.type(value)
                    if result.size > 1:
                        result = HashableArray(result)
                else:
                    result = HashableArray(numpy.array(value, dtype=vlen_type))
            return result
        except Exception as ex:
            # pylint: disable=broad-exception-raised
            raise Exception(
                "".join(format_exception(*sys.exc_info()))
                + f"\nWhile converting to LC type: {lc_quantity!s}={value!r}"
            ) from ex
            # pylint: enable=broad-exception-raised

    # pylint: enable=too-many-branches

    @classmethod
    def _get_dimensions_iterator(
        cls,
        dimensions,
        contract_multivalue=True,
        skip_dimensions=("frame", "source"),
    ):
        """
        Return iterator over all possible values for a set of dimensions.

        Args:
            dimensions(str or iterable):    Either the name of dataset for which
                to iterate over all dimensions other than `'frame'` and
                `'source'` or a tuple contaning all dimensions to iterate over.

            contract_multivalue(bool):    If true, multi-valued dimensions are
                treated as if they have a size of 1.

        Returns:
            iterator:
                Covering all possible combinatinos of values for each dimensions
                identified by `dimensions`.
        """

        if isinstance(dimensions, str):
            dimensions = cls.dataset_dimensions[dimensions]

        return itertools.product(
            *(
                range(
                    1
                    if (
                        contract_multivalue
                        and dim in cls._multivalued_entry_datasets
                    )
                    else cls.max_dimension_size[dim]
                )
                for dim in filter(
                    lambda dim: dim not in skip_dimensions, dimensions
                )
            )
        )

    @classmethod
    def _get_substitutions(cls, dimensions, dimension_values):
        """
        Return dict of path substitutions to fully resolve path to quantity.

        Args:
            dimensions:    See _get_dimensions_iterator().

            dimension_values(tuple(int)):    The values for the various dataset
                dimensions which to turn to substitutions.

        Returns:
            dict:
                Ready to be passed to functions needing it to resolve path.
        """

        if isinstance(dimensions, str):
            dimensions = cls.dataset_dimensions[dimensions]

        result = dict(
            zip(
                filter(lambda dim: dim not in ["frame", "source"], dimensions),
                dimension_values,
            ),
            **cls._path_substitutions,
        )

        for multicolumn_quantity in [
            "srcextract_psf_param",
            "srcproj_column_name",
        ]:
            if multicolumn_quantity in result:
                result[multicolumn_quantity] = cls._config[
                    multicolumn_quantity + "s"
                ][result[multicolumn_quantity]]

        return result

    @classmethod
    def _get_field_index(
        cls, quantity, dim_values, frame_index, source_index=None
    ):
        """Return the index within the field for the specified entry."""

        dimensions = cls.dataset_dimensions[quantity]
        if dimensions and dimensions[-1] == "source":
            assert dimensions[-2] == "frame"
            assert source_index is not None
            dimensions = dimensions[:-2] + ("source", "frame")
            dim_values += (source_index, frame_index)
        else:
            assert (
                not dimensions
                or dimensions[-1] == "frame"
                or (
                    dimensions[-1] in cls._multivalued_entry_datasets
                    and dimensions[-2] == "frame"
                )
            )
            dim_values += (frame_index,)

        index = 0
        for coord, dimension in zip(dim_values, dimensions):
            index = index * cls.max_dimension_size[dimension] + coord

        return index

    @classmethod
    def _get_num_entries(cls, dimensions):
        """Return how many entries are in a single value for the given dims."""

        for dimension_name in reversed(dimensions):
            if dimension_name not in ["frame", "source"]:
                if dimension_name in cls._multivalued_entry_datasets:
                    return cls.max_dimension_size[dimension_name]
                return 1
        return 1

    @classmethod
    def _set_field_entry(
        cls, quantity, value, *, frame_index, dim_values, source_index=None
    ):
        """
        Set the correct index within the field for the specified entry.

        Args:
            quantity(str):    The pipeline key of the dataset being filled.

            value:    The value to set to the field. Must have the correct
                type.

            frame_index:    See _add_to_data_slice().

            dim_values(tuple(int,...)):    The value for each dimension of
                the dataset.

            source_index(int or None):    For source dependent datasets
                only, the index of the source in the data_slice.

        Returns:
            None
        """

        dimensions = cls.dataset_dimensions[quantity]
        index = cls._get_field_index(
            quantity, dim_values, frame_index, source_index
        )

        num_entries = cls._get_num_entries(dimensions)
        if num_entries > 1:
            assert num_entries == len(value)
            for param_index, param_value in enumerate(value):
                cls._get_slice_field(quantity)[
                    index * num_entries + param_index
                ] = param_value
        else:
            cls._get_slice_field(quantity)[index] = value

    @classmethod
    def _get_lc_data(
        cls, quantity, dimension_values, source_index, defined_indices=None
    ):
        """
        Return the data that should be added to an LC dataset.

        Args:
            quantity(str):    The pipeline key of the LC quantity that
                will be extended.

            dimension_values(tuple(int)):    The values of the dimensions on
                which the dataset depends.

            source_index(int):    The index of the source for which the LC is
                being updated within the LCDataReader.lc_data_slice.

            defined_indices:    See _add_configurations().

        Returns:
            numpy.array:
                The data to pass to LightCurveFile.extend_dataset.
        """

        dimensions = cls.dataset_dimensions[quantity]

        if "frame" not in dimensions:
            raise IOError(
                f"Adding {quantity!s} dataset, which does not depend on frame"
            )

        num_entries = cls._get_num_entries(dimensions)

        num_frames = cls.max_dimension_size["frame"]

        first_index = (
            cls._get_field_index(
                quantity=quantity,
                dim_values=dimension_values,
                frame_index=0,
                source_index=source_index,
            )
            * num_entries
        )

        slice_data = cls._get_slice_field(quantity)
        # pylint: disable=protected-access
        if slice_data._type_ == c_char_p:
            source_data = numpy.array(
                slice_data[
                    first_index : first_index + num_frames * num_entries
                ],
                dtype="S70",
            )
        else:
            source_data = numpy.frombuffer(
                slice_data, numpy.dtype(slice_data._type_).base
            )[first_index : first_index + num_frames * num_entries]
            source_data.shape = (
                (num_frames, num_entries) if num_entries > 1 else (num_frames,)
            )
        # pylint: enable=protected-access

        if defined_indices is None:
            return source_data

        if num_entries > 1:
            return source_data[defined_indices, :]

        return source_data[defined_indices]

    def _get_configurations(self, data_reduction, frame_header, get_lc_dtype):
        """
        Extract all configurations from the given data reduction file.

        Args:
            data_reduction(DataReductionFile):    An opened (at least for
                reading) data reduction file for the frame being processed.

            frame_header(dict-like):    A pyfits header or the equivalent
                dictionary for the frame being processed.

            get_lc_dtype(callable):    Should return the data type of a dataset
                within light curves.

        Returns:
            dict:
                The keys are the various components for which configuration is
                defined in the given data reduction file/frame. The values are
                lists of 2-tuples:

                    * a tuple of the coordinates along each dataset dimension
                      for which the configuration applies.

                    * a frozen set of 2-tuples:

                        - the pipeline key of the configuration quantity

                        - The value of the configuration parameter.
        """

        def get_component_config(component_dsets, substitutions):
            """
            Return the configuration for a component/substitution combo.

            Args:
                component_dsets(iterable of strings):    The datasets which make
                    up the configuration component.

                substitutions(dict):    Substitutions required to fully resolve
                    the light curve paths of the datasets.

            Returns:
                frozenset or None:
                    The frozen set part of the result of the parent function if
                    at least one of the datasets had an entry for the given
                    substitutions. Otherwise None.
            """

            found_config = False
            config_list = []
            for dset_key in component_dsets:
                dset_dtype = get_lc_dtype(dset_key)
                try:
                    #                    if dset_key.endswith('.magfit.cfg.single_photref'):
                    #                        value = self._config['single_photref_dr_fname']
                    #                    el
                    if dset_key in self.header_datasets:
                        value = frame_header[
                            self.header_datasets[dset_key].upper()
                        ]
                    else:
                        value = data_reduction.get_attribute(
                            dset_key,
                            **substitutions,
                        )
                    found_config = True
                except (
                    (OSError, KeyError) if self._optional_header else OSError
                ):
                    if (
                        dset_key in self.header_datasets
                        and self._optional_header != "all"
                        and (
                            self.header_datasets[dset_key].upper()
                            not in self._optional_header
                        )
                    ):
                        raise

                    self._logger.warning(
                        "Failed to read %s from DR file for %s.",
                        dset_key,
                        repr(substitutions),
                    )
                    value = None

                config_list.append(
                    (
                        dset_key,
                        self._config_to_lc_format(dset_key, dset_dtype, value),
                    )
                )

            return frozenset(config_list) if found_config else None

        result = {}
        for component, (
            dimensions,
            component_dsets,
        ) in self.config_components.items():
            result[component] = []
            for dim_values in self._get_dimensions_iterator(dimensions):
                substitutions = self._get_substitutions(dimensions, dim_values)

                configuration = get_component_config(
                    component_dsets, substitutions
                )
                if configuration is not None:
                    result[component].append((dim_values, configuration))
        return result

    @classmethod
    def _get_slice_field(cls, pipeline_key):
        """Return the field in the data slice containing the given quantity."""

        return getattr(cls.lc_data_slice, pipeline_key.replace(".", "_"))

    # Split internally into sub-functions for readability
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements
    def _add_to_data_slice(
        self, *, data_reduction, frame_header, frame_index, lc_example
    ):
        """
        Add the information from a single frame to the LCDataSlice.

        Args:
            data_reduction(DataReductionFile):    An opened (at least for
                reading) data reduction file for the frame being processed.

            frame_header(dict-like):    A pyfits header or the equivalent
                dictionary for the frame being processed.

            frame_index(int):    The index at which to place this frame's data
                  in the LCDataSlice object being filled.

            lc_example(LightCurveFile):    See _classify_datasets().

        Returns:
            []:
                The of sources IDs (as tuples of integers) between (exclusive)
                self.last_source_below and self.first_source_above found in at
                least some of the frames which were not included in the
                LCDataSlice.
        """

        def fill_header_keywords():
            """Fill the non-config datasets equal to header keywords."""

            for lc_quantity, hdr_quantity in self.header_datasets.items():
                if (
                    "frame" in self.dataset_dimensions[lc_quantity]
                    and hdr_quantity != self.cfg_index_id
                ):
                    assert len(self.dataset_dimensions[lc_quantity]) == 1
                    header_key = hdr_quantity.upper()
                    value = frame_header.get(header_key)
                    assert (
                        value is not None
                        or self._optional_header == "all"
                        or (
                            self._optional_header is not None
                            and header_key in self._optional_header
                        )
                    )
                    if value is None:
                        lc_dtype = numpy.dtype(
                            type(
                                self._get_slice_field(lc_quantity)[frame_index]
                            )
                        )
                        if lc_dtype.kind == "f":
                            value = numpy.nan
                        elif lc_dtype.kind == "S":
                            value = b""
                        elif lc_dtype.kind in ["i", "u"]:
                            value = 0
                        elif lc_dtype.kind in ["b"]:
                            value = False

                    if isinstance(value, str):
                        value = value.encode("ascii")

                    self._get_slice_field(lc_quantity)[frame_index] = value

        def get_dset_default(quantity):
            """Return a value to fill undefined entries datasets with."""

            creation_args = lc_example.get_dataset_creation_args(quantity)
            default_value = creation_args.get("fillvalue")
            if default_value is None:
                if quantity == "catalogue.cfg.epoch":
                    return 2451545.0
                if (
                    h5py.check_dtype(vlen=numpy.dtype(creation_args["dtype"]))
                    is str
                ):
                    return ""
                if numpy.dtype(creation_args["dtype"]).kind == "f":
                    return numpy.nan
            return default_value

        def fill_from_attribute(quantity):
            """Fill the value of a quantity equal to a DR attribute."""

            dimensions = self.dataset_dimensions[quantity]
            for dim_values in self._get_dimensions_iterator(
                dimensions,
                skip_dimensions=("frame", "source", "srcextract_psf_param"),
            ):
                try:
                    attribute_value = data_reduction.get_attribute(
                        quantity,
                        default_value=get_dset_default(quantity),
                        **self._get_substitutions(quantity, dim_values),
                    )
                except OSError:
                    self._logger.warning(
                        "Attribute %s not found in %s for %s",
                        quantity,
                        repr(data_reduction.filename),
                        repr(self._get_substitutions(quantity, dim_values)),
                    )
                    continue
                if "srcextract_psf_param" in dimensions:
                    assert (
                        len(attribute_value)
                        == self.max_dimension_size["srcextract_psf_param"]
                    )
                    assert not dim_values
                    for param_index, param_value in enumerate(attribute_value):
                        self._set_field_entry(
                            quantity,
                            param_value,
                            frame_index=frame_index,
                            dim_values=(param_index,),
                        )
                else:
                    self._set_field_entry(
                        quantity,
                        attribute_value,
                        frame_index=frame_index,
                        dim_values=dim_values,
                    )

        def fill_source_field(
            quantity, values, dim_values, data_slice_source_indices
        ):
            """Fill a single per-source field from a sequence of values."""

            for source_value, data_slice_src_ind in zip(
                values, data_slice_source_indices
            ):
                if data_slice_src_ind is None:
                    continue
                self._set_field_entry(
                    quantity,
                    source_value,
                    frame_index=frame_index,
                    dim_values=dim_values,
                    source_index=data_slice_src_ind,
                )

        def fill_from_dataset(quantity, data_slice_source_indices):
            """
            Fill the value of a quantity equal to a DR dataset.

            Args:
                quantity(str):    The pipeline key of the dataset to fill.

                data_slice_source_indices:    See fill_direct_from_dr().

            Returns:
                None
            """

            num_sources = len(data_slice_source_indices)
            for dim_values in self._get_dimensions_iterator(quantity):
                substitutions = self._get_substitutions(quantity, dim_values)
                try:
                    fill_source_field(
                        quantity=quantity,
                        values=data_reduction.get_dataset(
                            quantity,
                            expected_shape=(num_sources,),
                            **substitutions,
                        ),
                        dim_values=dim_values,
                        data_slice_source_indices=data_slice_source_indices,
                    )

                except OSError:
                    self._logger.warning(
                        "Dataset identified by %s does not exist in %s for %s",
                        quantity,
                        repr(data_reduction.filename),
                        repr(substitutions),
                    )

        def fill_direct_from_dr(data_slice_source_indices):
            """
            Fill all quantities coming directly from DR files.

            Args:
                data_slice_source_indices(iterable of int):    The indices
                    within data_slice of the sources in the DR currently being
                    processed

            Returns:
                None
            """

            non_header_or_config = set(self.dataset_dimensions.keys()) - set(
                self.header_datasets.keys()
            )
            for config_quantity in self.config_components.values():
                non_header_or_config -= config_quantity[1]

            for quantity in non_header_or_config:
                if "source" not in self.dataset_dimensions[quantity]:
                    assert quantity not in data_reduction.elements["dataset"]
                    assert quantity not in data_reduction.elements["link"]
                    if quantity in data_reduction.elements["attribute"]:
                        fill_from_attribute(quantity)
                else:
                    assert quantity not in data_reduction.elements["attribute"]
                    if (
                        quantity in data_reduction.elements["dataset"]
                        or quantity in data_reduction.elements["link"]
                    ):
                        fill_from_dataset(quantity, data_slice_source_indices)

        def fill_sky_position_datasets(data_slice_source_indices):
            """Fill all datasets with pipeline key prefix `'skypos'`."""

            num_sources = len(data_slice_source_indices)

            # False positive, pylint does not see units attributes
            # pylint: disable=no-member
            location = EarthLocation(
                lat=frame_header["SITELAT"] * units.deg,
                lon=frame_header["SITELONG"] * units.deg,
                height=frame_header["SITEALT"] * units.m,
            )

            obs_time = Time(
                frame_header["JD-OBS"], format="jd", location=location
            )

            source_coords = SkyCoord(
                ra=self._ra_dec[0] * units.deg,
                dec=self._ra_dec[1] * units.deg,
                frame="icrs",
            )

            data = {}

            alt_az = source_coords.transform_to(
                AltAz(obstime=obs_time, location=location)
            )
            data["a180"] = 180.0 + alt_az.az.to(units.deg).value
            data["a180"][data["a180"] > 180.0] -= 360
            data["zenith_distance"] = 90.0 - alt_az.alt.to(units.deg).value
            # pylint: enable=no-member

            data["BJD"] = (
                obs_time.tdb + obs_time.light_travel_time(source_coords)
            ).jd
            # False positive
            # pylint: disable=no-member
            data["hour_angle"] = (
                obs_time.sidereal_time("apparent").to(units.hourangle).value
                - self._ra_dec[0] / 15.0
            )
            # pylint: enable=no-member
            data["per_source"] = numpy.ones((num_sources,), dtype=numpy.bool_)

            for quantity, values in data.items():
                fill_source_field(
                    quantity="skypos." + quantity,
                    values=values,
                    dim_values=(),
                    data_slice_source_indices=data_slice_source_indices,
                )

        def fill_srcextract_psf_map(source_data, data_slice_source_indices):
            """Fill all datasets containing the source exatrction PSF map."""

            psf_map = get_source_extracted_psf_map(
                data_reduction, **self._path_substitutions
            )
            psf_param_values = psf_map(source_data)

            assert set(psf_param_values.dtype.names) == set(
                self._config["srcextract_psf_params"]
            )

            for param_index, param_name in enumerate(
                self._config["srcextract_psf_params"]
            ):
                fill_source_field(
                    quantity="srcextract.psf_map.eval",
                    values=psf_param_values[param_name],
                    dim_values=(param_index,),
                    data_slice_source_indices=data_slice_source_indices,
                )

        fill_header_keywords()
        source_data = data_reduction.get_source_data(
            string_source_ids=False,
            all_numeric_source_ids=True,
            shape_fit=False,
            apphot=False,
            shape_map_variables=False,
            background=False,
            **self._path_substitutions,
        )

        data_slice_source_indices = [
            self.source_destinations.get(src_id) for src_id in source_data.index
        ]

        skipped_sources = []
        for skip_src, slice_ind in zip(
            source_data.index, data_slice_source_indices
        ):
            if slice_ind is None:
                skipped_sources.append(skip_src)

        if skipped_sources:
            source_data.drop(skipped_sources, inplace=True)

        fill_direct_from_dr(data_slice_source_indices)
        fill_sky_position_datasets(data_slice_source_indices)
        fill_srcextract_psf_map(source_data, data_slice_source_indices)
        fill_source_field(
            quantity="source_in_frame",
            values=numpy.ones(len(data_slice_source_indices), dtype=bool),
            dim_values=(),
            data_slice_source_indices=data_slice_source_indices,
        )

        return skipped_sources

    # pylint: enable=too-many-locals
    # pylint: enable=too-many-statements

    @classmethod
    def prepare_for_reading(cls):
        """Must be called every time a new batch of frames is being read."""

        to_reset = cls._get_slice_field("source_in_frame")
        memset(to_reset, 0, sizeof(to_reset))

    @classmethod
    def prepare_for_writing(cls, configurations_collection):
        """
        Prepare for writing after a slice of data has been read.

        Args:
            configurations_collection:    The list of configurations returned by
                LCDataReader for the data currently in the slice. It is assumed
                that the configurations follow the same order as the data slice
                entries.

        Returns:
            None
        """

        cls._organized_config = {
            component: {} for component in cls.config_components
        }
        for frame_index, configurations in enumerate(configurations_collection):
            for component, component_config in configurations.items():
                for dim_values, config in component_config:
                    if dim_values not in cls._organized_config[component]:
                        cls._organized_config[component][dim_values] = {}
                    if config in cls._organized_config[component][dim_values]:
                        config_id = cls._organized_config[component][
                            dim_values
                        ][config]
                    else:
                        config_id = len(
                            cls._organized_config[component][dim_values]
                        )
                        cls._organized_config[component][dim_values][
                            config
                        ] = config_id
                    cls._set_field_entry(
                        component + "." + cls.cfg_index_id,
                        config_id,
                        frame_index=frame_index,
                        dim_values=dim_values,
                    )

    @classmethod
    def print_organized_configurations(cls):
        """Print the result of organize_configurations() nicely formatted."""

        print("Organized configurations:")
        for component, component_config in cls._organized_config.items():
            print("\t" + component + ":")
            for dim_values, config_list in component_config.items():
                dim_id = dict(
                    zip(
                        filter(
                            lambda dim: dim not in ["frame", "source"],
                            cls.config_components[component][0],
                        ),
                        dim_values,
                    )
                )
                print("\t\t" + repr(dim_id) + ":")
                for config, config_id in config_list.items():
                    print("\t\t\t" + repr(config_id) + " -> " + repr(config))

    @classmethod
    def _write_configurations(
        cls,
        light_curve,
        source_index,
        defined_indices,
        resolve_lc_size="confirmed",
    ):
        """
        Add all configurations to the LC and fix their config_ids in slice.

        Args:
            light_curve(LightCurveFile):    The light curve to update.

            source_index(int):    The index of the source for which the
                lightcurve is being updated within the slice.

            defined_indices(numpy.array(dtype=bool)):    Array of flags
                indicating for each entry in the data slice should be included
                in this light curve.

        Returns:
            None
        """

        if not cls._organized_config:
            raise IOError(
                "Call prepare_for_writing() method after each slice "
                "input is complete."
            )

        for component, component_config in cls._organized_config.items():
            index_pipeline_key = component + ".cfg_index"
            for dim_values, config_list in component_config.items():

                config_ids = cls._get_lc_data(
                    quantity=index_pipeline_key,
                    dimension_values=dim_values,
                    source_index=source_index,
                    defined_indices=defined_indices,
                )

                config_to_add = []
                for config, slice_config_id in config_list.items():
                    if slice_config_id in config_ids:
                        config_to_add.append(config)

                light_curve.add_configurations(
                    component,
                    config_to_add,
                    config_ids,
                    **cls._get_substitutions(index_pipeline_key, dim_values),
                )

    @classmethod
    def _write_slice_data(
        cls, light_curve, source_index, defined_indices, resolve_lc_size
    ):
        """Add all non-configuration datasets to the light curve."""

        for quantity, dimensions in cls.dataset_dimensions.items():
            if (
                quantity.endswith("." + cls.cfg_index_id)
                or "frame" not in dimensions
                or quantity == "source_in_frame"
            ):
                continue

            for dim_values in cls._get_dimensions_iterator(quantity):
                light_curve.extend_dataset(
                    quantity,
                    cls._get_lc_data(
                        quantity=quantity,
                        dimension_values=dim_values,
                        source_index=source_index,
                        defined_indices=defined_indices,
                    ),
                    resolve_size=resolve_lc_size,
                    **cls._get_substitutions(quantity, dim_values),
                )

    def read(self, frame):
        """
        Add single frame's information to configurations and the LCDataSlice.

        Args:
            - frame: A 2-tulpe containing the following:

                - The filename of the DR file to read.

                - The index at which to place this frame's data in the
                  LCDataSlice object being filled.

        Returns:
            dict:
                See return value of _get_configurations().

            []:
                See return value of _add_to_data_slice().
        """

        def fix_header(frame_header):
            """Patch problems with FITS headers and adds filename info."""

            if "COMPOSIT" not in frame_header:
                frame_header["COMPOSIT"] = 1
            if "FILVER" not in frame_header:
                frame_header["FILVER"] = 0
            if "MNTSTATE" not in frame_header:
                frame_header["MNTSTATE"] = "unkwnown"
            elif frame_header["MNTSTATE"] is True:
                frame_header["MNTSTATE"] = "T"

            frame_header.update(self.dr_fname_parser(dr_fname))

            if self._observatory is not None:
                frame_header.update(self._observatory)

        try:
            with LightCurveFile() as lc_example:
                frame_index, dr_fname = frame
                with DataReductionFile(dr_fname, "r") as data_reduction:
                    frame_header = dict(
                        data_reduction.get_frame_header().items()
                    )
                    fix_header(frame_header)
                    configurations = self._get_configurations(
                        data_reduction, frame_header, lc_example.get_dtype
                    )

                    skipped_sources = self._add_to_data_slice(
                        data_reduction=data_reduction,
                        frame_header=frame_header,
                        frame_index=frame_index,
                        lc_example=lc_example,
                    )
                return configurations, skipped_sources
        except Exception as ex:
            raise IOError("While reading: " + repr(frame)) from ex

    def write(self, source_and_light_curve):
        """Write the data of the given source to the given light curve."""

        try:
            source_id, light_curve_fname = source_and_light_curve
            source_index = self.source_destinations.get(source_id)
            defined_indices = self._get_lc_data(
                quantity="source_in_frame",
                dimension_values=(),
                source_index=source_index,
            )
            if not defined_indices.any():
                return

            lc_directory = dirname(light_curve_fname)
            if not exists(lc_directory):
                self._logger.info("Created LC directory: %s", lc_directory)
                os.makedirs(lc_directory)

            try:
                source_ids = {"HAT": get_hat_source_id_str(source_id)}
            except (IndexError, TypeError):
                source_ids = {"Gaia DR3": repr(source_id)}

            with LightCurveFile(
                light_curve_fname, "a", source_ids=source_ids
            ) as light_curve:
                self._write_configurations(
                    light_curve,
                    source_index,
                    defined_indices,
                    resolve_lc_size="confirmed",
                )
                self._write_slice_data(
                    light_curve,
                    source_index,
                    defined_indices,
                    resolve_lc_size="confirmed",
                )
        except Exception as ex:
            raise IOError("While writing source: " + repr(source_id)) from ex
