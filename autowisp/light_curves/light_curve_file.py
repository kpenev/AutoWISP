"""Define a class for working with light curve files."""

import re

import numpy
import h5py

from autowisp.database.hdf5_file_structure import HDF5FileDatabaseStructure
from autowisp.evaluator import Evaluator
from .hashable_array import HashableArray

_config_dset_key_rex = re.compile(
    "|".join(
        [
            r"_cfg_version$",
            r".software_versions$",
            r"\.cfg\.(?!(epoch|fov|orientation))",
            r"^srcextract\.psf_map\.cfg\.",
        ]
    )
)


# Come from H5py.
# pylint: disable=too-many-ancestors
class LightCurveFile(HDF5FileDatabaseStructure):
    """
    Interface for working with the pipeline generated light curve files.

    Attributes:
        _config_indices(dict):    A dictionary of the already read-in
            configuration indices (re-used if requested again).
    """

    @classmethod
    def _product(cls):
        return "light_curve"

    @classmethod
    def _get_root_tag_name(cls):
        """The name of the root tag in the layout configuration."""

        return "LightCurve"

    def _get_hashable_dataset(self, dataset_key, **substitutions):
        """Return the selected dataset with hashable entries."""

        try:
            values = self.get_dataset(dataset_key, **substitutions)
        except IOError:
            return []

        if isinstance(values[0], numpy.ndarray):
            if h5py.check_dtype(vlen=values[0].dtype) is bytes:
                return [HashableArray(numpy.array(list(v))) for v in values]
            return [HashableArray(v) for v in values]
        if values.dtype.kind == "f":
            return [v if numpy.isfinite(v) else "NaN" for v in values]
        return values

    def _get_configurations(self, component, quantities, **substitutions):
        """
        Return a the configurations for a given component.

        Args:
            component:    What to return the configuration of. Should correspond
                to a configuration index variable (withouth the `.cfg_index`
                suffix).

            quantities:    A list of the pipeline keys identifying all
                quantities belonging to this configuration component. Undefined
                behavior results if `component` and `quantities` are not
                consistent.

            substitutions:    Substitutions required to fully resolve the paths
                to the datasets contaning the configurations.

        Returns: A dictionary indexed by the hash of a configuration with
                 entries 2-tuples of:
                    - the ID assigned to a configuration.
                    - and frozenset of (name, value) pairs containing the
                    configuration.

                 Also stores the extracted list of configurations as
                 self.__configurations[component][set(substitutions.items())]
        """

        def report_indistinct_configurations(config_list):
            """Report all repeating configurations in an exception."""

            message = (
                f"Identical {component!s} configurations found in "
                f"{self.filename}:\n"
            )
            hash_list = [hash(c) for c in config_list]
            for config in set(config_list):
                if config_list.count(config) != 1:
                    this_hash = hash(config)
                    message += (
                        "Indices ("
                        + ", ".join(
                            [
                                str(i)
                                for i, h in enumerate(hash_list)
                                if this_hash == h
                            ]
                        )
                        + ") contain: \n"
                    )
                    for key, value in zip(quantities, config):
                        message += f"\t {key} = {value!r}\n"
            raise IOError(message)

        substitution_set = frozenset(substitutions.items())
        if (
            component in self._configurations
            and substitution_set in self._configurations[component]
        ):
            return self._configurations[component][substitution_set]

        stored_configurations = list(
            zip(
                *[
                    self._get_hashable_dataset(pipeline_key, **substitutions)
                    for pipeline_key in quantities
                ]
            )
        )
        if len(set(stored_configurations)) != len(stored_configurations):
            report_indistinct_configurations(stored_configurations)
        stored_config_sets = [
            frozenset(zip(quantities, config))
            for config in stored_configurations
        ]
        result = {
            hash(config): (index, config)
            for index, config in enumerate(stored_config_sets)
        }
        if component not in self._configurations:
            self._configurations[component] = {}
        self._configurations[component][substitution_set] = result
        return result

    def __init__(self, *args, source_ids=None, **kwargs):
        """
        Open a lightcurve file.

        Args:
            source_ids(None or dict):    The known identifiers of this source in
                catalogues. Must be set if the lightcurve file is being created.
                If it already exists, identifiers already defined in the
                lightcurve are checked against supplied values, and new
                identifiers are added if the file is being opened for writing.

            args:    Passed directly to super().__init__()

            kwargs:    Passed directly to super().__init__()

        Returns:
            None
        """

        super().__init__(*args, **kwargs)
        self._configurations = {}
        self._config_indices = {}

        if "Identifiers" not in self and self.driver != "core":
            if not source_ids:
                raise ValueError(
                    "Must specify at least one identifier when creating new "
                    "lightcurve file!"
                )
            self.create_dataset(
                "Identifiers",
                (0, 2),
                maxshape=(None, 2),
                chunks=(10, 2),
                dtype=h5py.string_dtype(),
            )

        if source_ids is not None:
            add_source_ids = dict(source_ids)
            # False positive
            # pylint: disable=no-member
            stored_identifiers = dict(self["Identifiers"].asstr())
            # pylint: enable=no-member
            for catalogue, identifier in source_ids.items():
                if catalogue in stored_identifiers:
                    assert identifier == stored_identifiers[catalogue]
                    del add_source_ids[catalogue]
            # False positive
            # pylint: disable=no-member
            destination = self["Identifiers"].shape[0]
            self["Identifiers"].resize((destination + len(add_source_ids), 2))
            # pylint: enable=no-member
            for new_id in add_source_ids.items():
                self["Identifiers"][destination] = new_id
                destination += 1

    def get_config_indices(self, dataset_key, **substitutions):
        """Return the config index dset for indexing the given config dset."""

        substitution_key = frozenset(substitutions.items())
        config_component = dataset_key
        while True:
            try:
                result = self._config_indices.get(config_component)
                if result is not None:
                    result = result.get(substitution_key)
                if result is None:
                    result = self.get_dataset(
                        config_component + ".cfg_index", **substitutions
                    )
                    if config_component not in self._config_indices:
                        self._config_indices[config_component] = {}
                    self._config_indices[config_component][
                        substitution_key
                    ] = result

                return result
            except KeyError:
                config_component = config_component.rsplit(".", 1)[0]

    def read_data(self, dataset_key, **substitutions):
        """Similar to get_dataset, except config datasets are expanded."""

        data = self.get_dataset(dataset_key, **substitutions)

        if _config_dset_key_rex.search(dataset_key):
            config_indices = self.get_config_indices(
                dataset_key, **substitutions
            )
            data = data[config_indices]
        return data

    def read_data_array(self, variables):
        """
        Return a numpy structured array of the given variables.

        Args:
            variables([dict]):     The variables to read. Each key
                is a variable name in the resulting array and the corresponding
                value is a  2-tuple giving the dataset key to use for that
                variable, along with any substitutions required to fully resolve
                the dataset path.

        Retuns:
            numpy.array:
                Array with field names the variables specified on input
                containing the specified data. Configuration datasets are
                expanded to lightcurve points using the corresponding
                configuration index.
        """

        def result_column_dtype(dset_key):
            """The type to use for the given column in the result."""

            result = self.get_dtype(dset_key)
            if result == numpy.string_:
                return numpy.dtype("O")
            return result

        def create_empty_result(result_size):
            """Create an uninitialized dasates to hold the result."""

            return numpy.empty(
                result_size,
                dtype=[
                    (vname, result_column_dtype(dset_key))
                    for vname, (dset_key, subs) in variables.items()
                ],
            )

        result = None
        for var_name, (dataset_key, substitutions) in variables.items():
            data = self.read_data(dataset_key, **substitutions)

            if result is None:
                result = create_empty_result(data.size)
                first_dset = dataset_key, substitutions
            elif data.shape != result.shape:
                raise RuntimeError(
                    f"For {self.filename!r}, {dataset_key!r}: {substitutions!r}"
                    f"dataset shape {data.shape!r} does not match the shape of "
                    f"{result.shape!r} of {first_dset[0]!r}: {first_dset[1]!r}"
                )
            result[var_name] = data

        return result

    def add_configurations(
        self,
        component,
        configurations,
        config_indices,
        *,
        config_index_selection=None,
        **substitutions,
    ):
        """
        Add a list of configurations to the LC, merging with existing ones.

        Also updates the configuration index dataset.

        Args:
            component(str):    The component for which these configurations
                apply (i.e. it should have an associated configuration
                index dataset).

            configurations(iterable):    The configurations to add. Each
                configuration should be an iterable of 2-tuples formatted like
                (`pipeline_key`, `value`).

            config_indices(array of int):    For each frame, the corresponding
                entry is the index within configurations of the configuration
                that applies for that frame.

            resolve_size(str):    How to deal with confirm LC length differing
                from actual? See extend_dataset() for details.

            config_index_selection:    Either None, slice or boolean array for
                the configuration index dataset to set the new indices. If None,
                the new indices are appended at the end of the configuration
                index dataset, otherwise, it must selected exactly the same
                number of elements as are found in config_indices.

            substitutions:    Any substitutions required to fully resolve the
                paths to the configuration and configuration index datasets.

        Returns:
            None
        """

        def get_new_data():
            """Return a dict of pipeline_key, data of the updates needed."""

            index_dset = numpy.empty(config_indices.shape, dtype=numpy.uint)

            config_keys = None
            for config_index, new_config in enumerate(configurations):
                config_hash = hash(new_config)
                if config_keys is None:
                    config_keys = [entry[0] for entry in new_config]
                    stored_configurations = self._get_configurations(
                        component, config_keys, **substitutions
                    )
                    config_data_to_add = {key: [] for key in config_keys}
                else:
                    assert len(new_config) == len(config_keys)
                    for entry in new_config:
                        # Will be set to sequence before this
                        # pylint: disable=unsupported-membership-test
                        assert entry[0] in config_keys
                        # pylint: enable=unsupported-membership-test

                if config_hash in stored_configurations:
                    index_dset[config_indices == config_index] = (
                        stored_configurations[config_hash][0]
                    )
                else:
                    index_dset[config_indices == config_index] = len(
                        stored_configurations
                    )
                    stored_configurations[config_hash] = (
                        index_dset[config_index],
                        new_config,
                    )
                    for key, value in new_config:
                        config_data_to_add[key].append(
                            value.unwrap()
                            if isinstance(value, HashableArray)
                            else value
                        )
            for key in config_data_to_add:
                config_data_to_add[key] = numpy.array(
                    config_data_to_add[key],
                    dtype=h5py.check_dtype(
                        vlen=numpy.dtype(self.get_dtype(key))
                    ),
                )
            config_data_to_add[component + ".cfg_index"] = index_dset
            return config_data_to_add

        for pipeline_key, new_data in get_new_data().items():
            if config_index_selection is not None and pipeline_key == (
                component + ".cfg_index"
            ):
                self.add_dataset(
                    dataset_key=pipeline_key,
                    data=None,
                    if_exists="ignore",
                    unlimited=True,
                    shape=new_data.shape,
                    dtype=new_data.dtype,
                    **substitutions,
                )
                self[
                    self._file_structure[pipeline_key].abspath % substitutions
                ][config_index_selection] = new_data
            else:
                self.extend_dataset(
                    pipeline_key,
                    new_data,
                    resolve_size="actual",
                    is_config=True,
                    **substitutions,
                )

    def get_lc_length(self, **substitutions):
        """Return the number of poinst present in this lightcurve."""

        dataset_path = (
            self._file_structure["skypos.BJD"].abspath % substitutions
        )
        if dataset_path not in self:
            return 0
        return len(self[dataset_path])

    def extend_dataset(
        self,
        dataset_key,
        new_data,
        resolve_size=None,
        is_config=False,
        **substitutions,
    ):
        """
        Add more points to the dataset identified by dataset_key.

        If the given dataset does not exist it is created as unlimited in its
        first dimension, and matching the shape in `new_data` for the other
        dimensions.

        Args:

            dataset_key:    The key identifying the dataset to update.

            new_data:    The additional values that should be written, a numpy
                array with an appropriate data type and shape.

            resolve_size:    Should be either 'actual' or 'confirmed'.
                Indicating which dataset length to accept when
                adding new data. If left as `None`, an error is rasied if the
                confirmed length does not match the actual length of the
                dataset.

            substitututions:    Any arguments that should be substituted in the
                dataset path.

        Returns:
            None
        """

        def get_pad_value(replace_nonfinite):
            """If replace_nonfinite is undefined base padding on dtype."""

            if replace_nonfinite is not None:
                return replace_nonfinite

            dtype_str = self._file_structure[dataset_key].dtype
            if dtype_str == "numpy.bool_":
                return False
            if dtype_str == "numpy.float64":
                return numpy.nan
            if dtype_str == "numpy.int32":
                return numpy.iinfo(numpy.int32).min
            if dtype_str == "numpy.string_":
                return ""
            if dtype_str == "numpy.uint":
                return numpy.iinfo(numpy.uint).max
            assert False

        def add_new_data(dataset, confirmed_length):
            """Add new_data to the given dataset after confirmed_length."""

            dtype = self.get_dataset_creation_args(
                dataset_key, **substitutions
            ).get("dtype")
            if dtype is None:
                dtype = new_data.dtype
            else:
                dtype = numpy.dtype(dtype)
            self._logger.debug(
                "Adding new data after confirmed length %s:\n%s",
                repr(confirmed_length),
                repr(new_data),
            )
            data_copy = self._replace_nonfinite(
                new_data, dtype, dataset_config.replace_nonfinite
            )

            new_dataset_size = confirmed_length + len(data_copy)
            if new_dataset_size < len(dataset):
                try:
                    all_data = numpy.concatenate(
                        (dataset[:confirmed_length], data_copy)
                    )
                except Exception as exc:
                    raise IOError(
                        "Failed to read lightcurve dataset "
                        f"'{self.filename}/{dataset.name}' "
                        f"(actual length of {len(dataset):d}, "
                        f"expected {confirmed_length:d})!"
                    ) from exc
                self._logger.debug(
                    "Dataset %s length (%s) exceeds confirmed (%d) + new %s. "
                    "Recreating from scratch.",
                    dataset.name,
                    repr(dataset.shape),
                    confirmed_length,
                    repr(data_copy.shape),
                )
                self.add_dataset(
                    dataset_key=dataset_key,
                    data=all_data,
                    unlimited=True,
                    **substitutions,
                )
            else:
                pad = confirmed_length - len(dataset)
                dataset.resize(new_dataset_size, 0)
                if pad > 0:
                    self._logger.debug(
                        "Padding %s (shape %s) from %s to %s with %s",
                        repr(dataset.name),
                        repr(dataset.shape),
                        repr(confirmed_length - pad),
                        repr(confirmed_length),
                        repr(dataset_config.replace_nonfinite),
                    )
                    dataset[confirmed_length - pad : confirmed_length] = (
                        get_pad_value(dataset_config.replace_nonfinite)
                    )
                dataset[confirmed_length:] = data_copy

        dataset_config = self._file_structure[dataset_key]
        dataset_path = dataset_config.abspath % substitutions

        confirmed_length = self.get_attribute(
            "confirmed_lc_length", default_value=0
        )
        actual_length = self.get_lc_length(**substitutions)
        if confirmed_length > actual_length and resolve_size != "actual":
            raise IOError(
                f"The {self.filename} lightcurve has a length of "
                f"{actual_length}, smaller than the confirmed "
                f"length of {confirmed_length}."
            )
        if confirmed_length != actual_length:
            if not resolve_size:
                raise IOError(
                    f"The lightcurve {self.filename!r} has an actual "
                    f"length of {actual_length:d}, expected "
                    f"{confirmed_length:d}!"
                )
            if resolve_size == "actual":
                confirmed_length = actual_length
            elif resolve_size != "confirmed":
                raise IOError(
                    "Unexpected lightcurve length resolution: "
                    + repr(resolve_size)
                )
        confirmed_length = int(confirmed_length)

        if dataset_path in self:
            assert actual_length > 0
            dataset = self[dataset_path]
            add_new_data(
                dataset, len(dataset) if is_config else confirmed_length
            )
        else:
            if confirmed_length == 0 or is_config:
                data_to_add = new_data
            else:
                assert confirmed_length > 0
                data_to_add = numpy.concatenate(
                    (
                        numpy.full(
                            confirmed_length,
                            get_pad_value(dataset_config.replace_nonfinite),
                        ),
                        new_data,
                    )
                )
            self.add_dataset(
                dataset_key, data_to_add, unlimited=True, **substitutions
            )

    def confirm_lc_length(self, **substitutions):
        """Set the confirmed length of the lightcurve to match actual length."""

        self.add_attribute(
            "confirmed_lc_length", self.get_lc_length(**substitutions)
        )

    def get_num_magfit_iterations(
        self, photometry_mode, lc_points, **path_substitutions
    ):
        """
        Return how many magnitude fitting iterations are in the file.

        Args:
            path_substitutions:    See get_source_count().

        Returns:
            int:
                The number of magnitude fitting iterations performed on the
                set of photometry measurements identified by the
                path_substitutions argument.
        """

        path_substitutions["magfit_iteration"] = 0
        dataset_key = photometry_mode + ".magfit.magnitude"
        path_zero = (
            self._file_structure[dataset_key].abspath % path_substitutions
        )
        while True:
            path_substitutions["magfit_iteration"] += 1
            if (
                self._file_structure[dataset_key].abspath % path_substitutions
                == path_zero
            ):
                return 0
            try:
                if not numpy.isfinite(
                    self.get_dataset(dataset_key, **path_substitutions)[
                        lc_points
                    ]
                ).any():
                    break
            except IOError:
                break

        return path_substitutions["magfit_iteration"]

    def add_corrected_dataset(
        self,
        original_key,
        corrected_key,
        corrected_values,
        corrected_selection,
        **substitutions,
    ):
        """
        Add corrected values for a dataset (e.g. after EPD or TFA).

        Args:
            original_key(str):    The pipeline key identifying the original
                dataset that was corrected.

            corrected_key(str):    The pipeline key identifying the dataset
                where the corrected values should be stored.

            corrected_values:    The resulting values after the correction has
                been applied.

            corrected_selection:    Some sort of slice on the dataset that
                identifies the points which were corrected.

            substitutions:    Any arguments that need to be substituted in the
                paths of the original and corrected datasets to get a unique
                entry.

        Returns:
            None
        """

        self._logger.debug(
            "Adding to %s corrected version of %d points of %s: %s",
            repr(self.filename),
            corrected_selection.sum(),
            repr(self._file_structure[original_key].abspath % substitutions),
            repr(self._file_structure[corrected_key].abspath % substitutions),
        )
        original_dset = self[
            self._file_structure[original_key].abspath % substitutions
        ]
        self.add_dataset(
            dataset_key=corrected_key,
            data=None,
            if_exists="ignore",
            unlimited=True,
            shape=original_dset.shape,
            dtype=original_dset.dtype,
            **substitutions,
        )

        destination_config = self._file_structure[corrected_key]
        self[destination_config.abspath % substitutions][
            corrected_selection
        ] = self._replace_nonfinite(
            corrected_values,
            self.get_dataset_creation_args(corrected_key, **substitutions).get(
                "dtype"
            ),
            destination_config.replace_nonfinite,
        )

    def evaluate_expression(self, variables, expression):
        """Return the values of the given expression at each LC point."""

        return Evaluator(self.read_data_array(dict(variables)))(expression)


# pylint: enable=too-many-ancestors
