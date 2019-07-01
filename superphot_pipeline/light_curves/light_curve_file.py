"""Define a class for working with light curve files."""

import numpy
import h5py

from superphot_pipeline.database.hdf5_file_structure import\
    HDF5FileDatabaseStructure
from .hashable_array import HashableArray

#Come from H5py.
#pylint: disable=too-many-ancestors
class LightCurveFile(HDF5FileDatabaseStructure):
    """Interface for working with the pipeline generated light curve files."""

    @classmethod
    def _product(cls):
        return 'light_curve'

    @classmethod
    def _get_root_tag_name(cls):
        """The name of the root tag in the layout configuration."""

        return 'LightCurve'

    def _get_hashable_dataset(self, dataset_key, **substitutions):
        """Return the selected dataset with hashable entries."""

        try:
            values = self.get_dataset(dataset_key,
                                      **substitutions)
        except IOError:
            return []

        if isinstance(values[0], numpy.ndarray):
            return [HashableArray(v) for v in values]
        if values.dtype.kind == 'f':
            return [v if numpy.isfinite(v) else 'NaN' for v in values]
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

            message = ('Identical %s configurations found in %s:\n'
                       %
                       (component, self.filename))
            hash_list = [hash(c) for c in config_list]
            for config in set(config_list):
                if config_list.count(config) != 1:
                    this_hash = hash(config)
                    message += 'Indices (' + ', '.join(
                        [str(i) for i, h in enumerate(hash_list)
                         if this_hash == h]
                    ) + ') contain: \n'
                    for key, value in zip(quantities, config):
                        message += '\t %s = %s\n' % (key, repr(value))
            raise IOError(message)

        substitution_set = set(substitutions.items())
        if (
                component in self._configurations
                and
                substitution_set in self._configurations[component]
        ):
            return self._configurations[component][substitution_set]

        stored_configurations = zip(*[self._get_hashable_dataset(pipeline_key)
                                      for pipeline_key in quantities])
        if len(set(stored_configurations)) != len(stored_configurations):
            report_indistinct_configurations(stored_configurations)
        stored_config_sets = [frozenset(zip(quantities, config))
                              for config in stored_configurations]
        result = {hash(config): (index, config)
                  for index, config in enumerate(stored_config_sets)}
        if component not in self._configurations:
            self._configurations[component] = dict()
        self._configurations[component][substitution_set] = result
        return result

    def __init__(self, *args, **kwargs):
        """Passes all args directly to parent but sets self._configurations."""

        super().__init__(*args, **kwargs)
        self._configurations = dict()

    def add_configurations(self, component, configurations, **substitutions):
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


            substitutions:

        Returns:
            None
        """

        def get_new_data():
            """Return a dict of pipeline_key, data of the updates needed."""

            index_dset = numpy.empty(len(configurations), dtype=numpy.int)

            config_keys = None
            for config_index, new_config in enumerate(configurations):
                config_hash = hash(new_config)
                if config_keys is None:
                    config_keys = [entry[0] in entry in new_config]
                    stored_configurations = self._get_configurations(
                        component,
                        config_keys,
                        **substitutions
                    )
                    config_data_to_add = {key: [] for key in config_keys}
                else:
                    assert len(new_config) == len(config_keys)
                    for entry in new_config:
                        #Will be set to sequence before this
                        #pylint: disable=unsupported-membership-test
                        assert entry[0] in config_keys
                        #pylint: enable=unsupported-membership-test

                if config_hash in stored_configurations:
                    index_dset[config_index] = stored_configurations[
                        config_hash
                    ][
                        0
                    ]
                else:
                    index_dset[config_index] = len(stored_configurations)
                    stored_configurations[config_hash] = (
                        index_dset[config_index],
                        new_config
                    )
                    for key, value in new_config:
                        config_data_to_add[key].append(value)
            for key in config_data_to_add:
                config_data_to_add[key] = numpy.array(
                    config_data_to_add[key],
                    dtype=h5py.check_dtype(vlen=self.get_dtype(key))
                )
            config_data_to_add[component + '.cfg_index'] = index_dset
            return config_data_to_add

        for pipeline_key, new_data in get_new_data():
            self.extend_dataset(pipeline_key, new_data, **substitutions)

    def extend_dataset(self,
                       dataset_key,
                       new_data,
                       resolve_size=None,
                       **substitutions):
        """
        Add more points to the dataset identified by dataset_key.

        If the given dataset does not exist it is created as unlimited in its
        first dimension, and matching the shape in `new_data` for the other
        dimensions.

        Args:

            dataset_key:    The key identifying the dataset to update.

            new_data:    The additional values that should be written, a numpy
                array with an appropriate data type and shape.

            resolve_size:    Should be either 'actual' or 'confirmed'. In the
                first case, indicating which dataset length to accept when
                adding new data. If left as `None`, an error is rasied if the
                confirmed length does not match the actual length of the
                dataset.

            substitututions:    Any arguments that should be substituted in the
                dataset path.

        Returns:
            None
        """

        def add_new_data(dataset,
                         confirmed_length,
                         new_data,
                         replace_nonfinite):
            """Add new_data to the given dataset after confirmed_length."""

            data_copy = self._replace_nonfinite(new_data, replace_nonfinite)

            new_dataset_size = confirmed_length + len(new_data)
            if new_dataset_size < len(dataset):
                try:
                    all_data = np.concatenate((dataset[:confirmed_length],
                                               data_copy))
                except:
                    raise IOError(
                        "Failed to read lightcurve dataset '%s/%s' "
                        "(actual length of %d, expected %d)!"
                        %
                        (
                            self.filename,
                            dataset.name,
                            len(dataset),
                            confirmed_length
                        )
                    )
                self.add_dataset(
                    dataset_key=dataset_key,
                    data=all_data,
                    unlimited=True,
                    **substitutions
                )
            else:
                dataset.resize(new_dataset_size, 0)
                if h5py.check_dtype(vlen=dataset.dtype) is not None:
                    dataset[confirmed_length:] = list(data_copy)
                else:
                    dataset[confirmed_length:] = data_copy


        dataset_config = self._file_structure[dataset_key]
        dataset_path = dataset_config.abspath % substitutions

        if dataset_path in self:
            dataset = self[dataset_path]
            confirmed_length = self.get_attribute(
                'confirmed_lc_length',
                default_value=0
            )
            actual_length = len(dataset)
            if (
                    confirmed_length > actual_length
                    and
                    resolve_size != 'actual'
            ):
                raise IOError(
                    'The %s dataset of %s has a length of %d, smaller '
                    'than the confirmed length of the lightcurve (%d).'
                    %
                    (
                        dataset_path,
                        self.filename,
                        actual_length,
                        confirmed_length
                    )
                )
            if confirmed_length != actual_length:
                if not resolve_size:
                    raise IOError(
                        "The lightcurve dataset '%s/%s' has an actual "
                        "length of %d, expected %d!"
                        %
                        (
                            self.filename,
                            dataset_path,
                            len(dataset),
                            confirmed_length
                        )
                    )
                elif resolve_size == 'actual':
                    confirmed_length = actual_length
                elif resolve_size != 'confirmed':
                    raise IOError('Unexpected lightcurve length resolution: '
                                  +
                                  repr(resolve_size))
            add_new_data(dataset,
                         confirmed_length,
                         new_data,
                         dataset_config.replace_nonfinite)
        else:
            self.add_dataset(dataset_key,
                             new_data,
                             unlimited=True,
                             **substitutions)
#pylint: enable=too-many-ancestors
