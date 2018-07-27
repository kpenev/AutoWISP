#Only a single class is defined so hardly makes sense to split.
#pylint: disable=too-many-lines
"""Define a class for working with HDF5 files."""

from abc import ABC, abstractmethod
from io import StringIO
import os
import os.path
from sys import exc_info
from ast import literal_eval
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

            * data_set: Identifiers for the data sets that could be included in
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

    @staticmethod
    def write_text_to_dataset(text,
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

        try:
            data = numpy.frombuffer(text, dtype='i1')
        except AttributeError:
            if isinstance(text, numpy.ndarray) and text.dtype == 'i1':
                data = text
            else:
                data = numpy.fromfile(text, dtype='i1')

        dataset = h5group.create_dataset(dset_path, data=data, **creation_args)
        for key, value in attributes.items():
            dataset.attrs[key] = value

    @staticmethod
    def read_text_from_dataset(h5dset, as_file=False):
        r"""
        Reads a text from an HDF5 dataset.

        The inverse of :meth:`write_text_to_dataset`\ .

        Args:
            h5dset:    The dataset containing the text to read.

        Returns:
            text:    Numpy byte array (dtype='i1') containing the text.
        """

        text = numpy.empty((h5dset.len(),), dtype='i1')
        if h5dset.len() != 0:
            h5dset.read_direct(text)
        if as_file:
            return StringIO(text.data)

        return text

    @staticmethod
    def write_fitsheader_to_dataset(fitsheader, *args, **kwargs):
        r"""
        Adds a FITS header to an HDF5 file as a dataset.

        Args:
            fitsheader:    The header to save (fits.Header instance).

            args:    Passed directly to :meth:`write_text_to_dataset`\ .

            kwargs:    Passed directly to :meth:`write_text_to_dataset`\ .

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
        HDF5File.write_text_to_dataset(fitsheader_array, *args, **kwargs)

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
        try:
            header = fits.Header.fromfile(StringIO(fitsheader_array.data),
                                          endcard=False,
                                          padding=False)
        except:
            newlines = numpy.empty(fitsheader_array.size/80, dtype='i1')
            newlines.fill(numpy.frombuffer('\n', dtype='i1')[0])
            fitsheader_array = numpy.insert(fitsheader_array,
                                            slice(80, None, 80),
                                            numpy.frombuffer('\n', dtype='i1'))
            header = fits.Header.fromfile(StringIO(fitsheader_array.data))
        return header

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

        #pylint false positive
        #pylint: disable=no-member
        for (element_type, recognized) in cls.elements.items():
        #pylint: enable=no-member
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

        for dataset_key in self._elements['data_set']:
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

    @classmethod
    def get_version_dtype(cls, element_id, version=None):
        """
        What get_dtype would return for LC configured with the given version.

        Args:
            element_id:    The string identifier for the quantity to return the
                data type for.

            version:    The structure version for which to return the data type.
                If None, uses the latest configured version.

        Returns:
            dtype:    A numpy style data type to use for the quantity in LCs.
        """

        if version is None:
            destination = cls._default_destinations
        else:
            destination = cls.destination_versions[version]
        return destination[element_id]['creation_args']['dtype']

    def get_dtype(self, element_id):
        """Return numpy style data type string for the given element_id."""

        return self._destinations[element_id]['creation_args']['dtype']

    def add_attribute(self,
                      attribute_key,
                      attribute_value,
                      attribute_dtype=None,
                      if_exists='overwrite',
                      logger=None,
                      log_extra=dict(),
                      **substitutions):
        """
        Adds a single attribute to a dateset or a group.

        Args:
            attribute_key:    The key in _destinations that corresponds to the
                attribute to add. If the key is not one of the recognized keys,
                h5file is not modified and the function silently exits.

            attribute_value:    The value to give the attribute.

            attribute_dtype:    Data type for the new attribute, None to
                determine automatically.

            if_exists:    What should be done if the attribute exists? Possible
                values are:

                * ignore:    do not update but return the attribute's value.

                * overwrite:    Change the value to the specified one.

                * error: raise an exception.

            logger:    An object to pass log messages to.
            log_extra:    Extra information to attach to the log messages.

            substitutions:    variables to substitute in HDF5 paths and names.

        Returns:
            None.
        """

        if attribute_key not in self._destinations:
            if logger:
                logger.debug(
                    "Not adding '%s' attribute, since no destination is defined"
                    " for it."
                    %
                    attribute_key
                )
            return
        destination = self._destinations[attribute_key]
        parent_path = destination['parent'] % substitutions
        if parent_path not in self:
            if destination['parent_type'] != 'group':
                raise HDF5LayoutError(
                    "Attempting to add a attribute to non-existant %s ('%s') "
                    "in '%s'!"
                    %
                    (
                        destination['parent_type'],
                        parent_path,
                        self.filename
                    )
                )
            parent = self.create_group(parent_path)
        else:
            parent = self[parent_path]
        if logger:
            logger.debug(
                "Defining '%s%s.%s'='%s'"
                %
                (
                    self.filename,
                    destination['parent'] % substitutions,
                    destination['name'] % substitutions,
                    attribute_value
                ),
                extra=log_extra
            )
        attribute_name = destination['name'] % substitutions
        if attribute_name in parent.attrs:
            if if_exists == 'ignore':
                return parent.attrs[attribute_name]
            elif if_exists == 'error':
                raise HDF5LayoutError(
                    "Attribute '%s/%s.%s' already exists!"
                    %
                    (self.filename, parent_path, attribute_name)
                )
            else:
                assert if_exists == 'overwrite'

        parent.attrs.create(attribute_name,
                            attribute_value,
                            dtype=attribute_dtype)

    def add_link(self, target, name, logger=None, log_extra=dict()):
        """
        Adds a soft link to the HDF5 file.

        Args:
            target:    The path to create a soft link to.

            name:    The name to give to the link. Overwritten if it existts and
                is a link.

        Returns:
            None

        Raises:
            Error.HDF5:    if an object with the same name as the link exists,
                but is not a link.
        """

        if logger:
            logger.debug("Linking '%s' -> '%s' in '%s'"
                         %
                         (name, target, self.filename),
                         extra=log_extra)
        if name in self:
            if self.get(name, getclass=True, getlink=True) == h5py.SoftLink:
                if logger:
                    logger.debug("Removing old symlink '%s'" % name,
                                 extra=log_extra)
                del self[name]
            else:
                raise HDF5LayoutError(
                    "An object named '%s' already exists in '%s', and is not"
                    " a link. Not overwriting!"
                    %
                    (name, self.filename)
                )
        self[name] = h5py.SoftLink(target)

    def _delete_obsolete_dataset(self,
                                 parent,
                                 name,
                                 logger=None,
                                 log_extra=dict()):
        """
        Delete obsolete HDF5 dataset if it exists and update repacking flag.

        Args:
            parent:    The parent group this entry belongs to.

            name:    The name of the entry to check and delete. If the entry is
                not a dataset, an error is raised.

            logger:    An object to issue log messages to.

            log_extra:    Extra information to add to log messages.

        Returns:
            None

        Raises:
            Error.HDF5:    if an entry with the given name exists under parent,
                but is not a dataset.
        """

        if name in parent:
            if logger:
                logger.warning("Deleteing obsolete dataset '%s/%s' in '%s'"
                               %
                               (parent.name, name, self.filename))
            if 'Repack' in self:
                self.attrs['Repack'] = (
                    self.attrs['Repack']
                    +
                    ',%s/%s' % (parent, name)
                )
            else:
                self.attrs.create('Repack', bytes(',%s/%s' % (parent, name)))
            del parent[name]

    def dump_file_like(self,
                       file_like,
                       destination,
                       link_name=False,
                       logger=None,
                       external_log_extra=dict(),
                       log_dumping=True):
        """
        Adds a byte-by-byte dump of a file-like object to self.

        Args:
            file_like:    A file-like object to dump.

            destination:    The path in self to use for the dump.

            link_name:    If this argument converts to True, a link with the
                given name is created pointing to destination.

            logger:    An object to emit log messages to.

            external_log_extra:    extra information to add to log message.

        Returns:
            None.
        """

        if destination['parent'] not in self:
            parent = self.create_group(destination['parent'])
        else:
            parent = self[destination['parent']]
        self._delete_obsolete_dataset(parent,
                                      destination['name'],
                                      logger,
                                      external_log_extra)
        text_to_dataset(
            (
                file_like
                if file_like is not None else
                numpy.empty((0,), dtype='i1')
            ),
            parent,
            destination['name'],
            creation_args=destination['creation_args']
        )
        if link_name:
            self.add_link(
                destination['parent'] + '/' + destination['name'],
                link_name,
                logger=logger,
                log_extra=log_extra
            )

    def add_file_dump(self,
                      fname,
                      destination,
                      link_name=False,
                      delete_original=True,
                      logger=None,
                      external_log_extra=dict()):
        """
        Adds a byte by byte dump of a file to the data reduction file.

        If the file does not exist an empty dataset is created.

        Args:
            fname:    The name of the file to dump.

            destination:    Passed directly to dump_file_like.

            link_name:    Passed directly to dump_file_like.

            delete_original:    If True, the file being dumped is
                deleted (default).

            logger:    An object to emit log messages to.

            external_log_extra:    extra information to add to log message.

        Returns:
            None.
        """

        if logger:
            logger.debug(
                (
                    "Adding dump of '%s' to '%s' as '%s/%s'"
                    %
                    (
                        fname,
                        self.filename,
                        destination['parent'],
                        destination['name']
                    )
                    +
                    (" and linking as '%s'" % link_name if link_name else '')
                ),
                extra=dict(log_extra.items() + external_log_extra.items())
            )
        self.dump_file_like(
            (open(fname, 'r') if exists(fname) else None),
            destination,
            link_name,
            logger,
            external_log_extra,
            log_dumping=False
        )
        if delete_original and exists(fname):
            os.remove(fname)

    def get_file_dump(self, dump_key):
        """
        Returns as a string (with name attribute) a previously dumped file.

        Args:
            dump_key:    The key in self._destinations identifying the file
                to extract.

        Returns:
            dump:    The text of the dumped file.
        """

        if dump_key not in self._destinations:
            raise HDF5LayoutError(
                "The key '%s' does not exist in the list of configured data "
                "reduction file entries."
                %
                dump_key
            )
        destination = self._destinations[dump_key]
        dset_name = destination['parent'] + '/' + destination['name']
        if dset_name not in self:
            raise HDF5LayoutError("No '%s' dataset found in data reduction '%s'"
                                  %
                                  (dset_name, self.filename))
        result = text_from_dataset(self[dset_name], as_file=True)
        result.name = self.filename + '/' + dset_name
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
                attribute rasies Error.Sanity.

            substitutions:    Any keys that must be substituted in the path
                (i.e. ap_ind, config_id, ...).

        Returns:
            value:    The value of the attribute.
        """

        if attribute_key not in self._destinations:
            raise HDF5LayoutError(
                "The key '%s' does not exist in the list of configured data "
                "reduction file entries."
                %
                attribute_key
            )
        destination = self._destinations[attribute_key]
        parent_path = destination['parent'] % substitutions
        attribute_name = destination['name'] % substitutions
        if parent_path not in self:
            if default_value is not None:
                return default_value
            raise HDF5LayoutError(
                "Requested attribute '%s' from a non-existent %s: '%s/%s'!"
                %
                (
                    attribute_name,
                    destination['parent_type'],
                    self.filename,
                    parent_path
                )
            )
        parent = self[parent_path]
        if attribute_name not in parent.attrs:
            if default_value is not None:
                return default_value
            raise HDF5LayoutError(
                "The attribute '%s' is not defined for '%s/%s'!"
                %
                (attribute_name, self.filename, parent_path)
            )
        return parent.attrs[attribute_name]

    def get_single_dataset(self,
                           dataset_key,
                           sub_entry=None,
                           expected_shape=None,
                           optional=None,
                           **substitute):
        """
        Return a single dataset as a numpy float or int array.

        Args:
            dataset_key:    The key in self._destinations identifying the
                dataset to read.

            sub_entry:    If the dataset_key does not identify a single dataset,
                this value is used to select from among the multiple possible
                datasets (e.g. 'field' or 'source' for source IDs)

            expected_size:    The size to use for the dataset if an empty
                dataset is found. If None, a zero-sized array is returned.

            optional:    If not None and the dataset does not exist, this value
                is returned, otherwise if the dataset does not exist an
                exception is raised.

            substitute:    Any arguments that should be substituted in the path
                (e.g. ap_ind or config_id).

        Returns:
            numpy.array:
                A numpy int/float array containing the identified dataset from
                the HDF5 file.
        """

        key = (('%(parent)s/%(name)s' % self._destinations[dataset_key])
               %
               substitute)
        if key not in self:
            if optional is not None:
                return optional
            raise HDF5LayoutError(
                "Requested dataset '%s' does not exist in '%s'!"
                %
                (key, self.filename)
            )
        dataset = self[key]
        if sub_entry is not None:
            dataset = dataset[sub_entry]
        variable_length_dtype = h5py.check_dtype(vlen=dataset.dtype)
        if variable_length_dtype is not None:
            result_dtype = variable_length_dtype
        elif numpy.can_cast(dataset.dtype, int):
            result_dtype = int
        else:
            result_dtype = float
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
            result = numpy.empty(shape=dataset.shape, dtype=result_dtype)
            dataset.read_direct(result)
        if (
                'replace_nonfinite' in self._destinations[dataset_key]
                and
                (
                    dataset.fillvalue != 0
                    or
                    self._destinations[dataset_key]['replace_nonfinite'] == 0
                )
        ):
            result[result == dataset.fillvalue] = numpy.nan
        else:
            assert not dataset.fillvalue
        return result

    def add_single_dataset(self,
                           parent,
                           name,
                           data,
                           creation_args,
                           replace_nonfinite=None,
                           logger=None,
                           log_extra=dict(),
                           **kwargs):
        """
        Adds a single dataset to self.

        If the target dataset already exists, it is deleted first and the
        name of the dataset is added to the root level Repack attribute.

        Args:
            parent:    The full path of the group under which to place the new
                dataset (created if it does not exist).

            name:    The name of the dataset.

            data:    The values that should be written, a numpy array with
                appropriate type already set.

            creation_args:    Additional arguments to pass to the create_dataset
                method.

            replace_nonfinite:    If not None, any non-finite values are
                replaced with this value, it is also used as the fill value for
                the dataset.

            logger:    An object to send log messages to.

            log_extra:    Extra information to add to log messages

            kwargs:    Ignored.

        Returns:
            None
        """

        if logger:
            logger.debug("Creating dataset '%s/%s' in '%s'"
                         %
                         (parent, name, self.filename),
                         extra=log_extra)
        if parent not in self:
            parent_group = self.create_group(parent)
        else:
            parent_group = self[parent]
        self._delete_obsolete_dataset(parent_group, name, logger, log_extra)
        fillvalue = None
        if replace_nonfinite is None:
            data_copy = data
        else:
            finite = numpy.isfinite(data)
            fillvalue = replace_nonfinite
            if finite.all():
                data_copy = data
            else:
                data_copy = numpy.copy(data)
                data_copy[numpy.logical_not(finite)] = replace_nonfinite

        if fillvalue is None:
            parent_group.create_dataset(name,
                                        data=data_copy,
                                        **creation_args)
        else:
            parent_group.create_dataset(name,
                                        data=data_copy,
                                        fillvalue=fillvalue,
                                        **creation_args)

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
