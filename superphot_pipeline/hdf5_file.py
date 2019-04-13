#Only a single class is defined so hardly makes sense to split.
#pylint: disable=too-many-lines
"""Define a class for working with HDF5 files."""

from abc import ABC, abstractmethod
from io import BytesIO
import os
import os.path
from sys import exc_info
#from ast import literal_eval
from traceback import format_exception

from lxml import etree
import h5py
import numpy
from astropy.io import fits

from superphot_pipeline.pipeline_exceptions import HDF5LayoutError

git_id = '$Id$'

#This is a h5py issue not an issue with this module
#pylint: disable=too-many-ancestors
class HDF5File(ABC, h5py.File):
    """
    Base class for HDF5 pipeline products.

    The actual structure of the file has to be defined by a class inheriting
    from this one, by overwriting the relevant properties and
    :meth:`_get_root_tag_name`.

    Implements backwards compatibility for different versions of the structure
    of files.

    Attributes:
        _file_structure:    See the first entry returned by _get_file_structure.

        _file_structure_version:    See the second entry returned by
            _get_file_structure.
    """

    @classmethod
    @abstractmethod
    def _get_root_tag_name(cls):
        """The name of the root tag in the layout configuration."""

    @property
    def _layout_version_attribute(self):
        """
        Return path, name of attribute in the file holding the layout version.
        """

        return '/', 'LayoutVersion'

    @property
    @abstractmethod
    def _elements(self):
        """
        Identifying strings for the recognized elements of the HDF5 file.

        Shoul be a dictionary-like object with values being a set of strings
        containing the identifiers of the HDF5 elements and keys:

            * dataset: Identifiers for the data sets that could be included in
                the file.

            * attribute: Identifiers for the attributes that could be included
                in the file.

            * link: Identifiers for the links that could be included in
                the file.
        """

    @abstractmethod
    def _get_file_structure(self, version=None):
        """
        Return the layout structure with the given version of the file.

        Args:
            version:    The version number of the layout structure to set. If
                None, it should provide the default structure for new files
                (presumably the latest version).

        Returns:
            (dict, str):

                The dictionary specifies how to include elements in the HDF5
                file. The keys for the dictionary should be one in one of the
                lists in self._elements and the value is an object with
                attributes decsribing how to include the element. See classes in
                :mod:database.data_model for the provided attributes and their
                meining.

                The string is the actual file structure version returned. The
                same as version if version is not None.
        """

    def _flag_required_attribute_parents(self):
        """
        Flag attributes whose parents must exist when adding the attribute.

        The file structure must be fully configured before calling this method!

        If the parent is a group, it is safe to create it and then add the
        attribute, however, this in is not the case for attributes to datasets.

        Add an attribute named 'parent_must_exist' to all attribute
        configurations in self._file_structure set to False if and only if the
        attribute parent is a group.
        """

        dataset_paths = [self._file_structure[dataset_key].abspath
                         for dataset_key in self._elements['dataset']]

        for attribute_key in self._elements['attribute']:
            attribute = self._file_structure[attribute_key]
            attribute.parent_must_exist = attribute.parent in dataset_paths

    @staticmethod
    def _write_text_to_dataset(text,
                               h5group,
                               dset_path,
                               creation_args=None,
                               **attributes):
        r"""
        Adds ASCII text/file as a dateset to an HDF5 file.

        Args:
            text:    The text or file to add. If it is an open file, the
                contents is dumped, if it is a python2 string or a python3
                bytes, the value is stored.

            h5group:    An HDF5 group (could be the root group, i.e. an
                h5py.File opened for writing).

            dset_path:    The path for the new data set, either absolute or
                relative to h5group.

            creation_args:    Keyword arguments to pass to
                :meth:`create_dataset`\ . If
                ``None``, defaults to ``dict(compression='gzip',
                compression_opts=9)``.

            compression_opts:    see same name argument
                in h5py.File.create_dataset.

            attributes:    Added as attributes with the same name to the
                the data set.

        Returns:
            None
        """

        if creation_args is None:
            creation_args = dict(compression='gzip',
                                 compression_opts=9)

        if isinstance(text, bytes):
            data = numpy.frombuffer(text, dtype='i1')
        elif isinstance(text, numpy.ndarray) and text.dtype == 'i1':
            data = text
        else:
            data = numpy.fromfile(text, dtype='i1')

        dataset = h5group.create_dataset(dset_path, data=data, **creation_args)
        for key, value in attributes.items():
            dataset.attrs[key] = value

    @staticmethod
    def _read_text_from_dataset(h5dset, as_file=False):
        r"""
        Reads a text from an HDF5 dataset.

        The inverse of :meth:`_write_text_to_dataset`\ .

        Args:
            h5dset (h5py.DataSet):    The dataset containing the text to read.

            as_file (bool):    Should the return value be file-like?

        Returns:
            bytes or BytesIO:
                If as_file is False: numpy byte array (dtype='i1') containing
                the text. If as_file is True: a BytesIO wrapped around the
                stored text.
        """

        text = numpy.empty((h5dset.len(),), dtype='i1')
        if h5dset.len() != 0:
            h5dset.read_direct(text)
        if as_file:
            return BytesIO(text.data)

        return text

    @staticmethod
    def write_fitsheader_to_dataset(fitsheader, *args, **kwargs):
        r"""
        Adds a FITS header to an HDF5 file as a dataset.

        Args:
            fitsheader:    The header to save (fits.Header instance).

            args:    Passed directly to :meth:`_write_text_to_dataset`\ .

            kwargs:    Passed directly to :meth:`_write_text_to_dataset`\ .

        Returns:
            None
        """

        if isinstance(fitsheader, str):
            #pylint false positive
            #pylint: disable=no-member
            with fits.open(fitsheader, 'readonly') as fitsfile:
                header = fitsfile[0].header
                if header['NAXIS'] == 0:
                    header = fitsfile[1].header
                fitsheader_string = b''.join(map(bytes, header.cards))
            #pylint: enable=no-member
        else:
            fitsheader_string = b''.join(fitsheader.cards)
        fitsheader_array = numpy.frombuffer(fitsheader_string, dtype='i1')
        HDF5File._write_text_to_dataset(fitsheader_array, *args, **kwargs)

    @staticmethod
    def read_fitsheader_from_dataset(h5dset):
        """
        Reads a FITS header from an HDF5 dataset.

        The inverse of :meth:`write_fitsheader_to_dataset`.

        Args:
            h5dset:    The dataset containing the header to read.

        Returns:
            header:    Instance of fits.Header.
        """

        fitsheader_array = numpy.empty((h5dset.len(),), dtype='i1')
        h5dset.read_direct(fitsheader_array)
        return fits.Header.fromfile(BytesIO(fitsheader_array.data),
                                    endcard=False,
                                    padding=False)

    def _check_for_dataset(self, dataset_key, must_exist=True, **substitutions):
        """
        Check if the given key identifies a dataset and it actually exists.

        Args:
            dataset_key:    The key identifying the dataset to check for.

            must_exist:    If True, and the dataset does not exist, raise
                IOError.

            substitutions:    Any arguments that should be substituted in the
                path. Only required if must_exist == True.

        Returns:
            None

        Raises:
            KeyError:
                If the specified key is not in the currently set file structure
                or does not identify a dataset.

            IOError:
                If the dataset does not exist but the must_exist argument is
                True.
        """

        if dataset_key not in self._file_structure:

            raise KeyError(
                "The key '%s' does not exist in the list of configured data "
                "reduction file entries."
                %
                dataset_key
            )
        if dataset_key not in self._elements['dataset']:
            raise KeyError(
                "The key '%s' does not identify a dataset in '%s'"
                %
                (dataset_key, self.filename)
            )

        if must_exist:
            dataset_path = (self._file_structure[dataset_key].abspath
                            %
                            substitutions)
            if dataset_path not in self:
                raise IOError("Requried dataset ('%s') '%s' does not exist in '%s'"
                              %
                              (dataset_key, dataset_path, self.filename))

    @classmethod
    def get_element_type(cls, element_id):
        """
        Return the type of HDF5 entry that corresponds to the given ID.

        Args:
            element_id:    The identifying string for an element present in the
                HDF5 file.

        Returns:
            hdf5_type:    The type of HDF5 structure to create for this element.
                One of: 'group', 'dataset', 'attribute', 'link'.
        """

        for (element_type, recognized) in cls._elements.items():
            if element_id.rstrip('.') in recognized:
                return element_type

        raise KeyError('Unrecognized element: ' + repr(element_id))

    def layout_to_xml(self):
        """Create an etree.Element decsribing the currently defined layout."""

        root = etree.Element('group',
                             dict(name=self._get_root_tag_name(),
                                  version=self._file_structure_version))

        def require_parent(path, must_be_group):
            """
            Return group element at the given path creating groups as needed.

            Args:
                path ([str]):    The path for the group element required. Each
                    entry in the list is the name of a sub-group of the previous
                    entry.

            Returns:
                etree.Element:
                    The element holding the group at the specified path. If it
                    does not exist, it is created along with any parent groups
                    required along the way.

            Raises:
                TypeError:
                    If an element anywhere along the given path already exists,
                    but is not a group.
            """

            parent = root
            current_path = ''
            for group_name in path:
                found = False
                current_path += '/' + group_name
                for element in parent.iterfind('./*'):
                    if element.attrib['name'] == group_name:
                        if (
                                element.tag != 'group'
                                and
                                (must_be_group or element.tag != 'dataset')
                        ):
                            raise TypeError(
                                'Element '
                                +
                                repr(current_path)
                                +
                                ' exists, but is of type '
                                +
                                element.tag
                                +
                                ', expected group'
                                +
                                ('' if must_be_group else ' or dataset')
                                +
                                '!'
                            )
                        else:
                            parent = element
                            found = True
                            break
                if not found:
                    parent = etree.SubElement(parent,
                                              'group',
                                              name=group_name)
            return parent

        def add_dataset(parent, dataset):
            """
            Add the given dataset as a SubElement to the given parent.

            Args:
                parent (etree.Element):    The group element in the result
                    tree to add the dataset under.

                dataset:    The dataset to add (object with attributes
                    specifying how the dataset should be added to the file).
            """

            etree.SubElement(
                parent,
                'dataset',
                name=dataset.abspath.rsplit('/', 1)[1],
                key=dataset.pipeline_key,
                dtype=dataset.dtype,
                compression=((dataset.compression or '')
                             +
                             ':'
                             +
                             (dataset.compression_options or '')),
                scaleoffset=str(dataset.scaleoffset),
                shuffle=str(dataset.shuffle),
                fill=repr(dataset.replace_nonfinite),
                description=dataset.description
            )

        def add_attribute(parent, attribute):
            """Add the given attribute as a SubElement to the given parent."""

            etree.SubElement(
                parent,
                'attribute',
                name=attribute.name,
                key=attribute.pipeline_key,
                dtype=dataset.dtype,
                description=attribute.description
            )

        def add_link(parent, link):
            """Add the given link as a SubElement to the given parent."""

            etree.SubElement(
                parent,
                'link',
                name=link.abspath.rsplit('/', 1)[1],
                key=link.pipeline_key,
                target=link.target,
                description=link.description
            )

        for dataset_key in self._elements['dataset']:
            dataset = self._file_structure[dataset_key]
            path = dataset.abspath.lstrip('/').split('/')[:-1]
            add_dataset(require_parent(path, True), dataset)

        for attribute_key in self._elements['attribute']:
            attribute = self._file_structure[attribute_key]
            path = attribute.parent.lstrip('/').split('/')
            add_attribute(require_parent(path, False), attribute)

        for link_key in self._elements['link']:
            link = self._file_structure[link_key]
            path = link.abspath.lstrip('/').split('/')[:-1]
            add_link(require_parent(path, True), link)

        return root

    def get_dtype(self, element_key):
        """Return numpy data type for the element with by the given key."""

        result = self._file_structure[element_key].dtype

        if result == 'manual':
            return None

        #Used only on input defined by us.
        #pylint: disable=eval-used
        result = eval(result)
        #pylint: enable=eval-used

        if isinstance(result, str):
            result = numpy.dtype(result)

        if element_key.endswith('.hat_id_prefix'):
            return h5py.special_dtype(
                enum=(
                    result,
                    dict((prefix, value)
                         for value, prefix in enumerate(self._hat_id_prefixes))
                )
            )

        return result

    def get_dataset_creation_args(self, dataset_key):
        """
        Return all arguments to pass to create_dataset() except the content.

        Args:
            dataset_key:    The key identifying the dataset to delete.

        Returns:
            dict:
                All arguments to pass to create_dataset() or require_dataset()
                except: name, shape and data.
        """

        self._check_for_dataset(dataset_key, False)

        dataset_config = self._file_structure[dataset_key]
        result = dict()

        if dataset_config.compression is not None:
            result['compression'] = dataset_config.compression
            if (
                    dataset_config.compression == 'gzip'
                    and
                    dataset_config.compression_options is not None
            ):
                result['compression_opts'] = int(
                    dataset_config.compression_options
                )

        if dataset_config.scaleoffset is not None:
            result['scaleoffset'] = dataset_config.scaleoffset

        result['shuffle'] = dataset_config.shuffle

        if dataset_config.replace_nonfinite is not None:
            result['fillvalue'] = dataset_config.replace_nonfinite

        return result

    @staticmethod
    def hdf5_class_string(hdf5_class):
        """Return a string identifier of the given hdf5 class."""

        if issubclass(hdf5_class, h5py.Group):
            return "group"
        elif issubclass(hdf5_class, h5py.Dataset):
            return "dataset"
        elif issubclass(hdf5_class, h5py.HardLink):
            return "hard link"
        elif issubclass(hdf5_class, h5py.SoftLink):
            return "soft link"
        elif issubclass(hdf5_class, h5py.ExternalLink):
            return "external link"
        else:
            raise ValueError(
                'Argument to hdf5_class_string does not appear to be a class or'
                ' a child of a class defined by h5py!'
            )

    def add_attribute(self,
                      attribute_key,
                      attribute_value,
                      if_exists='overwrite',
                      **substitutions):
        """
        Adds a single attribute to a dateset or a group.

        Args:
            attribute_key:    The key in _destinations that corresponds to the
                attribute to add. If the key is not one of the recognized keys,
                h5file is not modified and the function silently exits.

            attribute_value:    The value to give the attribute.

            if_exists:    What should be done if the attribute exists? Possible
                values are:

                * ignore:
                    do not update but return the attribute's value.

                * overwrite:
                    Change the value to the specified one.

                * error:
                    raise an exception.

            substitutions:    variables to substitute in HDF5 paths and names.

        Returns:
            unknown:
                The value of the attribute. May differ from attribute_value if
                the attribute already exists, if type conversion is performed,
                or if the file structure does not specify a location for the
                attribute. In the latter case the result is None.
        """

        if attribute_key not in self._file_structure:
            return None

        assert attribute_key in self._elements['attribute']

        attribute_config = self._file_structure[attribute_key]
        parent_path = (attribute_config.parent
                       %
                       substitutions)
        if parent_path not in self:
            parent = self.create_group(parent_path)
        else:
            parent = self[parent_path]

        attribute_name = attribute_config.name % substitutions
        if attribute_name in parent.attrs:
            if (
                    if_exists == 'ignore'
                    or
                    parent.attrs[attribute_name] == attribute_value
            ):
                return parent.attrs[attribute_name]
            elif if_exists == 'error':
                raise HDF5LayoutError(
                    "Attribute '%s/%s.%s' already exists!"
                    %
                    (self.filename, parent_path, attribute_name)
                )
            else:
                assert if_exists == 'overwrite'

        if type(attribute_value) in [str, bytes, numpy.string_]:
            parent.attrs.create(attribute_name,
                                (
                                    attribute_value.encode('ascii')
                                    if type(attribute_value) is str
                                    else attribute_value
                                ))
        else:
            parent.attrs.create(attribute_name,
                                attribute_value,
                                dtype=self.get_dtype(attribute_key))

        return parent.attrs[attribute_name]

    def add_link(self, link_key, if_exists='overwrite', **substitutions):
        """
        Adds a soft link to the HDF5 file.

        Args:
            link_key:    The key identifying the link to create.

            if_exists:    See same name argument to :meth:`add_attribute`.

            substitutions:    variables to substitute in HDF5 paths and names of
                both where the link should be place and where it should point
                to.

        Returns:
            str:
                The path the identified link points to. See if_exists argument
                for how the value con be determined or None if the link was not
                created (not defined in current file structure).

        Raises:
            IOError:    if an object with the same name as the link exists,
                but is not a link or is a link, but does not point to the
                configured target and if_exists == 'error'.
        """

        if link_key not in self._file_structure:
            return None

        assert link_key in self._elements['link']

        link_config = self._file_structure[link_key]

        link_path = link_config.abspath % substitutions
        target_path = link_config.target % substitutions

        if link_path in self:
            existing_class = self.get(link_path, getclass=True, getlink=True)
            if issubclass(existing_class, h5py.SoftLink):
                existing_target_path = self[link_path].path
                if (
                        if_exists == 'ignore'
                        or
                        existing_target_path == target_path
                ):
                    return existing_target_path

                raise IOError(
                    "Unable to create link with key %s: a link at '%s' already"
                    " exists in '%s', and points to '%s' instead of '%s'!"
                    %
                    (
                        link_key,
                        link_path,
                        self.filename,
                        existing_target_path,
                        target_path
                    )
                )
            else:
                raise IOError(
                    "Unable to create link with key %s: a %s at '%s' already"
                    " exists in '%s'!"
                    %
                    (
                        link_key,
                        self.hdf5_class_string(existing_class),
                        link_path,
                        self.filename,
                    )
                )
        self[link_path] = h5py.SoftLink(target_path)
        return target_path

    def _delete_obsolete_dataset(self, dataset_key, **substitutions):
        """
        Delete obsolete HDF5 dataset if it exists and update repacking flag.

        Args:
            dataset_key:    The key identifying the dataset to delete.

        Returns:
            bool:
                Was a dataset actually deleted?

        Raises:
            Error.HDF5:
                if an entry already exists at the target dataset's location
                but is not a dataset.
        """

        if dataset_key not in self._file_structure:
            return False

        self._check_for_dataset(dataset_key, False)

        dataset_config = self._file_structure[dataset_key]
        dataset_path = dataset_config.abspath % substitutions

        if dataset_path in self:
            repack_attribute_config = self._file_structure['repack']
            if repack_attribute_config.parent not in self:
                self.create_group(repack_attribute_config.parent)
            repack_parent = self[repack_attribute_config.parent]
            if repack_attribute_config.name in repack_parent.attrs:
                repack_parent.attrs[repack_attribute_config.name] = (
                    self.attrs[repack_attribute_config.name]
                    +
                    ','
                    +
                    dataset_path
                )
            else:
                repack_parent.create(repack_attribute_config.name,
                                     dataset_path,
                                     dtype=self.get_dtype('repack'))
            del self[dataset_path]
            return True

        return False

    def dump_file_or_text(self,
                          dataset_key,
                          file_contents,
                          if_exists='overwrite',
                          **substitutions):
        """
        Adds a byte-by-byte dump of a file-like object to self.

        Args:
            dataset_key:    The key identifying the dataset to create for the
                file contents.

            file_contents:    See text argument to
                :meth:`_write_text_to_dataset`. None is also a valid value, in
                which case an empty dataset is created.

            if_exists:    See same name argument to add_attribute.

            substitutions:    variables to substitute in the dataset HDF5 path.
        Returns:
            (bool):
                Was the dataset actually created?
        """

        if dataset_key not in self._file_structure:
            return

        self._check_for_dataset(dataset_key, False)

        dataset_path = self._file_structure[dataset_key].abspath % substitutions
        assert self.get_dtype(dataset_key) == numpy.dtype('i1')

        if dataset_path in self:
            if if_exists == 'ignore':
                return
            elif if_exists == 'error':
                raise IOError("Dataset ('%s') '%s' already exists in '%s' and "
                              "overwriting is not allowed!"
                              %
                              (dataset_key, dataset_path, self.filename))
            else:
                self._delete_obsolete_dataset(dataset_key, **substitutions)

        self._write_text_to_dataset(
            text=(
                file_contents
                if file_contents is not None else
                numpy.empty((0,), dtype='i1')
            ),
            h5group=self,
            dset_path=dataset_path,
            creation_args=self.get_dataset_creation_args(dataset_key)
        )

    def add_file_dump(self,
                      dataset_key,
                      fname,
                      if_exists='overwrite',
                      delete_original=True,
                      **substitutions):
        """
        Adds a byte by byte dump of a file to self.

        If the file does not exist an empty dataset is created.

        Args:
            fname:    The name of the file to dump.

            dataset_key:    Passed directly to dump_file_like.

            if_exists:    See same name argument to add_attribute.

            delete_original:    If True, the file being dumped is
                deleted (default).

            substitutions:    variables to substitute in the dataset HDF5 path.
        Returns:
            None.
        """


        created_dataset = self.dump_file_or_text(
            dataset_key,
            (open(fname, 'r') if os.path.exists(fname) else None),
            if_exists,
            **substitutions
        )
        if delete_original and os.path.exists(fname):
            if created_dataset:
                os.remove(fname)
            else:
                raise IOError("Dataset '%s' containing a dump of '%s' not "
                              "created in '%s' but original deletion was "
                              "requested!"
                              %
                              (dataset_key, fname, self.filename))

    def get_file_dump(self, dataset_key, **substitutions):
        """
        Returns as a string (with name attribute) a previously dumped file.

        Args:
            dataset_key:    The key identifying the dataset containing the file
                dump.

            substitutions:    Any arguments that should be substituted in the
                path. Only required if must_exist == True.

        Returns:
            bytes:
                The text of the dumped file.
        """

        self._check_for_dataset(dataset_key, True, **substitutions)
        assert self.get_dtype(dataset_key) == numpy.dtype('i1')

        result = self._read_text_from_dataset(
            self[self._file_structure[dataset_key].abspath],
            as_file=True
        )
        return result

    def get_attribute(self,
                      attribute_key,
                      default_value=None,
                      **substitutions):
        """
        Returns the attribute identified by the given key.

        Args:
            attribute_key:    The key of the attribute to return. It must be one
                of the standard keys.

            default_value:    If this is not None this values is returned if the
                attribute does not exist in the file, if None, not finding the
                attribute rasies IOError.

            substitutions:    Any keys that must be substituted in the path
                (i.e. ap_ind, config_id, ...).

        Returns:
            value:    The value of the attribute.

        Raises:
            KeyError:
                If no attribute with the given key is defined in the current
                files structure or if it does not correspond to an attribute.

            IOError:
                If the requested dataset is not found and no default value was
                given.
        """

        if attribute_key not in self._file_structure:
            raise KeyError(
                "The key '%s' does not exist in the list of configured HDF5 "
                "file structure."
                %
                attribute_key
            )
        if attribute_key not in self._elements['attribute']:
            raise KeyError(
                "The key '%s' does not correspond to an attribute in the "
                "configured HDF5 file structure."
                %
                attribute_key
            )

        attribute_config = self._file_structure[attribute_key]

        parent_path = attribute_config.parent % substitutions
        attribute_name = attribute_config.name % substitutions

        if parent_path not in self:
            if default_value is not None:
                return default_value
            raise IOError(
                "Requested attribute (%s) '%s' from a non-existent path: '%s' "
                "in '%s'!"
                %
                (
                    attribute_key,
                    attribute_name,
                    parent_path,
                    self.filename,
                )
            )
        parent = self[parent_path]
        if attribute_name not in parent.attrs:
            if default_value is not None:
                return default_value
            raise IOError(
                "The attribute (%s) '%s' is not defined for '%s' in '%s'!"
                %
                (attribute_key, attribute_name, parent_path, self.filename)
            )
        return parent.attrs[attribute_name]

    def get_dataset(self,
                    dataset_key,
                    expected_shape=None,
                    default_value=None,
                    **substitutions):
        """
        Return a dataset as a numpy float or int array.

        Args:
            dataset_key:    The key in self._destinations identifying the
                dataset to read.

            expected_shape:    The shape to use for the dataset if an empty
                dataset is found. If None, a zero-sized array is returned.

            default_value:    If not None and the dataset does not exist, this
                value is returned, otherwise if the dataset does not exist an
                exception is raised.

            substitutions:    Any arguments that should be substituted in the
                path.

        Returns:
            numpy.array:
                A numpy int/float array containing the identified dataset from
                the HDF5 file.

        Raises:
            KeyError:
                If the specified key is not in the currently set file structure
                or does not identify a dataset.

            IOError:
                If the dataset does not exist, and no default_value was
                specified
        """

        self._check_for_dataset(dataset_key,
                                default_value is None,
                                **substitutions)

        dataset_config = self._file_structure[dataset_key]
        dataset_path = dataset_config.abspath % substitutions

        if dataset_path not in self:
            return default_value

        dataset = self[dataset_path]
        variable_length_dtype = h5py.check_dtype(vlen=dataset.dtype)
        if variable_length_dtype is not None:
            result_dtype = variable_length_dtype

        if dataset.size == 0:
            result = numpy.empty(
                shape=(dataset.shape
                       if expected_shape is None else
                       expected_shape),
                dtype=result_dtype
            )
            result.fill(numpy.nan)
        elif variable_length_dtype is not None:
            return dataset[:]
        else:
            result = numpy.empty(shape=dataset.shape,
                                 dtype=self.get_dtype(dataset_key))
            dataset.read_direct(result)

        if (
                dataset_config.replace_nonfinite is not None
                and
                result.dtype.kind == 'f'
        ):
            result[result == dataset.fillvalue] = numpy.nan

        return result

    def add_dataset(self,
                    dataset_key,
                    data,
                    if_exists='overwrite',
                    **substitutions):
        """
        Adds a single dataset to self.

        If the target dataset already exists, it is deleted first and the
        name of the dataset is added to the root level Repack attribute.

        Args:
            dataset_key:    The key identifying the dataset to add.

            data:    The values that should be written, a numpy array with
                an appropriate data type.

            replace_nonfinite:    If not None, any non-finite values are
                replaced with this value, it is also used as the fill value for
                the dataset.

            if_exists:    See same name argument to add_attribute.

            substitututions:    Any arguments that should be substituted in the
                dataset path.

        Returns:
            None
        """

        self._check_for_dataset(dataset_key, False)
        dataset_config = self._file_structure[dataset_key]
        dataset_path = dataset_config.abspath % substitutions

        if dataset_path in self:
            if if_exists == 'ignore':
                return
            elif if_exists == 'error':
                raise IOError("Dataset ('%s') '%s' already exists in '%s' and "
                              "overwriting is not allowed!"
                              %
                              (dataset_key, dataset_path, self.filename))
            else:
                self._delete_obsolete_dataset(dataset_key, **substitutions)

        if dataset_config.replace_nonfinite is None:
            fillvalue = None
            data_copy = data
        else:
            fillvalue = dataset_config.replace_nonfinite
            finite = numpy.isfinite(data)
            if finite.all():
                data_copy = data
            else:
                data_copy = numpy.copy(data)
                data_copy[numpy.logical_not(finite)] = fillvalue
        print(repr(self.get_dataset_creation_args(dataset_key)))
        self.create_dataset(dataset_path,
                            data=data_copy,
                            **self.get_dataset_creation_args(dataset_key))

    def __init__(self,
                 fname,
                 mode,
                 layout_version=None):
        """
        Opens the given HDF5 file in the given mode.

        Args:
            fname:    The name of the file to open.

            mode:    The mode to open the file in (see hdf5.File).

            layout_version:    If the file does not exist, this is the version
                of the layout that will be used for its structure. Leave None
                to use the latest defined.

        Returns:
            None
        """

        old_file = os.path.exists(fname)
        if mode[0] != 'r':
            path = os.path.dirname(fname)
            if path:
                try:
                    os.makedirs(path)
                except OSError:
                    if not os.path.exists(path):
                        raise

        try:
            h5py.File.__init__(self, fname, mode)
        except IOError:
            raise HDF5LayoutError(
                'Problem opening %s in mode=%s'%(fname, mode)
                +
                ''.join(format_exception(*exc_info()))
            )

        layout_version_path, layout_version_attr = (
            self._layout_version_attribute
        )

        if old_file:
            layout_version = (
                self[layout_version_path].attrs[layout_version_attr]
            )
        else:
            layout_version = layout_version

        self._file_structure, self._file_structure_version = (
            self._get_file_structure(layout_version)
        )

        if not old_file:
            self[layout_version_path].attrs[layout_version_attr] = (
                self._file_structure_version
            )
#pylint: enable=too-many-ancestors
