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
import logging

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

        _hat_id_prefixes (numpy.array):    A list of the currently recognized
            HAT-ID prefixes, with the correct data type ready for adding as a
            dataset.
    """

    @classmethod
    @abstractmethod
    def _get_root_tag_name(cls):
        """The name of the root tag in the layout configuration."""

    @classmethod
    @abstractmethod
    def _product(cls):
        """The pipeline key of the product held in this type of HDF5 files."""

    @property
    def _layout_version_attribute(self):
        """
        Return path, name of attribute in the file holding the layout version.
        """

        return '/', 'LayoutVersion'

    @property
    @abstractmethod
    def elements(self):
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

    @classmethod
    @abstractmethod
    def _get_file_structure(cls, version=None):
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
                lists in self.elements and the value is an object with
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
                         for dataset_key in self.elements['dataset']]

        for attribute_key in self.elements['attribute']:
            attribute = self._file_structure[attribute_key]
            attribute.parent_must_exist = attribute.parent in dataset_paths

    def _write_text_to_dataset(self,
                               dataset_key,
                               text,
                               if_exists='overwrite',
                               **substitutions):
        r"""
        Adds ASCII text/file as a dateset to an HDF5 file.

        Args:
            dataset_key:    The key identifying the dataset to add.

            text:    The text or file to add. If it is an open file, the
                contents is dumped, if it is a python2 string or a python3
                bytes, the value is stored.

            if_exists:    See add_dataset().

            substitututions:    Any arguments that should be substituted in the
                dataset path.

        Returns:
            None
        """

        if isinstance(text, bytes):
            data = numpy.frombuffer(text, dtype='i1')
        elif isinstance(text, numpy.ndarray) and text.dtype == 'i1':
            data = text
        else:
            data = numpy.fromfile(text, dtype='i1')

        self.add_dataset(dataset_key,
                         data,
                         if_exists=if_exists,
                         **substitutions)

    def write_fitsheader_to_dataset(self,
                                    dataset_key,
                                    fitsheader,
                                    **kwargs):
        r"""
        Adds a FITS header to an HDF5 file as a dataset.

        Args:
            dataset_key(str):    The key identifying the dataset to add the
                header to.

            fitsheader(fits.Header):    The header to save.

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
            fitsheader_string = b''.join(
                card.image.encode('ascii') for card in fitsheader.cards
            )
        fitsheader_array = numpy.frombuffer(fitsheader_string, dtype='i1')
        self._write_text_to_dataset(dataset_key, fitsheader_array, **kwargs)

    def read_fitsheader_from_dataset(self, dataset_key, **substitutions):
        """
        Reads a FITS header from an HDF5 dataset.

        The inverse of :meth:`write_fitsheader_to_dataset`.

        Args:
            h5dset:    The dataset containing the header to read.

        Returns:
            fits.Header:
                The FITS header contained in the given dataset.
        """

        fitsheader_array = self.get_dataset(dataset_key, **substitutions)
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
                "The key '%s' does not exist in the list of configured %s file "
                "entries."
                %
                (dataset_key, self._product())
            )

        if (
                dataset_key not in self.elements['dataset']
                and
                dataset_key not in self.elements['link']
        ):
            raise KeyError(
                "The key '%s' does not identify a dataset or link in '%s'"
                %
                (dataset_key, self.filename)
            )

        if must_exist:
            dataset_path = (self._file_structure[dataset_key].abspath
                            %
                            substitutions)
            if dataset_path not in self:
                raise IOError(
                    "Requried dataset ('%s') '%s' does not exist in '%s'"
                    %
                    (dataset_key, dataset_path, self.filename)
                )

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

        #All implementations of _elemnts are required to make them dict-like.
        #pylint: disable=no-member
        for (element_type, recognized) in cls.elements.items():
            if element_id.rstrip('.') in recognized:
                return element_type
        #pylint: enable=no-member

        raise KeyError('Unrecognized element: ' + repr(element_id))

    def get_element_path(self, element_id, **substitutions):
        """
        Return the path to the given element (.<attr> for attributes).

        Args:
            substitutions:    Arguments that should be substituted in the path.
                If none are given, the path is returned without substitutions.

        Returns:
            str:
                A string giving the path the element does/will have in the file.
        """

        for (element_type, recognized) in self.elements.items():
            if element_id.rstrip('.') in recognized:
                if element_type == 'attribute':
                    attribute_config = self._file_structure[element_id]
                    path_template = (attribute_config.parent
                                     +
                                     '.'
                                     +
                                     attribute_config.name)
                else:
                    path_template = self._file_structure[element_id].abspath

        if substitutions:
            return path_template % substitutions
        return path_template

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
            if len(path) == 1 and path[0] == '':
                return parent
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

        for dataset_key in self.elements['dataset']:
            dataset = self._file_structure[dataset_key]
            path = dataset.abspath.lstrip('/').split('/')[:-1]
            add_dataset(require_parent(path, True), dataset)

        for attribute_key in self.elements['attribute']:
            attribute = self._file_structure[attribute_key]
            path = attribute.parent.lstrip('/').split('/')
            add_attribute(require_parent(path, False), attribute)

        for link_key in self.elements['link']:
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

        return result

    #The path_substitutions arg is used by overloading functions.
    #pylint: disable=unused-argument
    def get_dataset_creation_args(self, dataset_key, **path_substitutions):
        """
        Return all arguments to pass to create_dataset() except the content.

        Args:
            dataset_key:    The key identifying the dataset to delete.

            path_substitutions:    In theory the dataset creation arguments can
                depend on the full dataset path (c.f. srcextract.sources).

        Returns:
            dict:
                All arguments to pass to create_dataset() or require_dataset()
                except: name, shape and data.
        """

        self._check_for_dataset(dataset_key, False)

        dataset_config = self._file_structure[dataset_key]
        result = dict(shuffle=dataset_config.shuffle)

        dtype = self.get_dtype(dataset_key)
        if dtype is not None:
            result['dtype'] = dtype

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

        if dataset_config.replace_nonfinite is not None:
            result['fillvalue'] = dataset_config.replace_nonfinite

        if dataset_key in ['catalogue.columns', 'srcproj.columns']:
            column = path_substitutions[dataset_key.split('.')[0]
                                        +
                                        '_column_name']
            if column in ['hat_id_prefix',
                          'hat_id_field',
                          'hat_id_source',
                          'objtype',
                          'doublestar',
                          'sigRA',
                          'sigDec',
                          'phqual',
                          'magsrcflag',
                          'enabled',
                          'DESIGNATION',
                          'phot_variable_flag',
                          'datalink_url',
                          'epoch_photometry_url',
                          'libname_gspphot',
                          'pmra',
                          'pmdec',
                          'phot_bp_mean_mag',
                          'phot_rp_mean_mag',
                          'phot_bp_mean_flux',
                          'phot_rp_mean_flux',
                          'phot_bp_mean_flux_error',
                          'phot_rp_mean_flux_error',
                          'phot_bp_rp_excess_factor']:
                result['compression'] = 'gzip'
                result['compression_opts'] = 9
                result['shuffle'] = True
            elif column in ['RA', 'Dec', 'RA_orig', 'Dec_orig']:
                del result['compression']
                result['scaleoffset'] = 7
            elif column in ['xi', 'eta', 'x', 'y']:
                del result['compression']
                result['scaleoffset'] = 6
            elif (
                column in ['J', 'H', 'K',
                           'B', 'V', 'R', 'I',
                           'u', 'g', 'r', 'i', 'z']
                or
                column.endswith('mag')
            ):
                del result['compression']
                result['scaleoffset'] = 3
            elif column in ['dist',
                            'epochRA', 'epochDec',
                            'sigucacmag',
                            'errJ', 'errH', 'errK']:
                del result['compression']
                result['scaleoffset'] = 2
            elif column in 'source_id' or column.endswith('_n_obs'):
                del result['compression']
                result['dtype'] = numpy.dtype('uint64')
                result['scaleoffset'] = 0
            else:
                del result['compression']
                result['scaleoffset'] = 1

        return result
    #pylint: enable=unused-argument

    @staticmethod
    def hdf5_class_string(hdf5_class):
        """Return a string identifier of the given hdf5 class."""

        if issubclass(hdf5_class, h5py.Group):
            return "group"
        if issubclass(hdf5_class, h5py.Dataset):
            return "dataset"
        if issubclass(hdf5_class, h5py.HardLink):
            return "hard link"
        if issubclass(hdf5_class, h5py.SoftLink):
            return "soft link"
        if issubclass(hdf5_class, h5py.ExternalLink):
            return "external link"
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

        assert attribute_key in self.elements['attribute']

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
            #TODO: handle  multi-valued attributes correctly.
            if (
                    if_exists == 'ignore'
                    or
                    (
                        parent.attrs[attribute_name]
                        ==
                        numpy.asarray(attribute_value)
                    ).all()
            ):
                return parent.attrs[attribute_name]
            if if_exists == 'error':
                raise HDF5LayoutError(
                    "Attribute '%s/%s.%s' already exists!"
                    %
                    (self.filename, parent_path, attribute_name)
                )
            assert if_exists == 'overwrite'

        if isinstance(attribute_value, (str, bytes, numpy.string_)):
            parent.attrs.create(attribute_name,
                                (
                                    attribute_value.encode('ascii')
                                    if isinstance(attribute_value, str)
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

        assert link_key in self.elements['link']

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
                repack_parent.attrs.create(repack_attribute_config.name,
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

        self._write_text_to_dataset(
            dataset_key=dataset_key,
            text=(
                file_contents
                if file_contents is not None else
                numpy.empty((0,), dtype='i1')
            ),
            if_exists=if_exists,
            **substitutions
        )
        return True

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
        if attribute_key not in self.elements['attribute']:
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

            default_value:    If the dataset does not exist, this value is
                returned.

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
            result = numpy.full(
                shape=(dataset.shape
                       if expected_shape is None else
                       expected_shape),
                fill_value=numpy.nan
            )
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

    def get_dataset_shape(self, dataset_key, **substitutions):
        """Return the shape of the given dataset."""

        dataset_path = self._file_structure[dataset_key].abspath % substitutions

        if dataset_path not in self:
            return None

        return self[dataset_path].shape


    @staticmethod
    def _replace_nonfinite(data, expected_dtype, replace_nonfinite):
        """Return (copy of) data with non-finite values replaced."""

        if (
                data.dtype.kind == 'S'
                and
                (
                    (
                        expected_dtype is not None
                        and
                        numpy.dtype(expected_dtype).kind == 'f'
                    )
                    or
                    (data == b'NaN').all()
                )
        ):
            assert (data == b'NaN').all() or (data == b'None').all()
            return numpy.full(
                fill_value=(replace_nonfinite or numpy.nan),
                dtype=numpy.float64,
                shape=data.shape
            )

        if replace_nonfinite is None:
            return data

        finite = numpy.isfinite(data)
        if finite.all():
            return data
        data_copy = numpy.copy(data)
        data_copy[numpy.logical_not(finite)] = replace_nonfinite
        return data_copy

    def add_dataset(self,
                    dataset_key,
                    data,
                    *,
                    if_exists='overwrite',
                    unlimited=False,
                    shape=None,
                    dtype=None,
                    **substitutions):
        """
        Adds a single dataset to self.

        If the target dataset already exists, it is deleted first and the
        name of the dataset is added to the root level Repack attribute.

        Args:
            dataset_key:    The key identifying the dataset to add.

            data:    The values that should be written, a numpy array with
                an appropriate data type or None if an empty dataset should be
                created.

            if_exists:    See same name argument to add_attribute.

            unlimited(bool):    Should the first dimension of the dataset be
                unlimited (i.e. data can be added later)?

            shape(tuple(int,...)):    The shape of the dataset to create if data
                is None, otherwise the shape of the data is used. Just like if
                data is specified, the first dimension will be ignored if
                unlimited is True. It is an error to specify both data and
                shape!

            dtype:    The data type for the new dataset if the data is None. It
                is an error to specify both dtype and data!

            substitututions:    Any arguments that should be substituted in the
                dataset path.

        Returns:
            None
        """

        self._check_for_dataset(dataset_key, False)
        dataset_config = self._file_structure[dataset_key]
        dataset_path = dataset_config.abspath % substitutions

        if dataset_path in self:
            print(
                'Dataset {0!r} already existis in {1!r}!'.format(
                    dataset_path,
                    self.filename
                )
            )
            if if_exists == 'ignore':
                return
            if if_exists == 'error':
                raise IOError("Dataset ('%s') '%s' already exists in '%s' and "
                              "overwriting is not allowed!"
                              %
                              (dataset_key, dataset_path, self.filename))
            self._delete_obsolete_dataset(dataset_key, **substitutions)

        creation_args = self.get_dataset_creation_args(dataset_key,
                                                       **substitutions)

        if data is None:
            data_copy = None
        else:
            data_copy = self._replace_nonfinite(
                data,
                creation_args.get('dtype'),
                dataset_config.replace_nonfinite
            )

        if data is not None:
            assert shape is None
            assert dtype is None
            shape = data.shape
            dtype = data_copy.dtype

        if unlimited:
            shape_tail = shape[1:]

            if hasattr(self, '_chunk_size'):
                #pylint: disable=no-member
                creation_args['chunks'] = (self._chunk_size,) + shape_tail
                #pylint: enable=no-member
            else:
                creation_args['chunks'] = True

            creation_args['maxshape'] = (None,) + shape_tail

        if dtype.kind == 'S':
            assert creation_args.get('dtype', numpy.bytes_) == numpy.bytes_
            creation_args['dtype'] = h5py.special_dtype(vlen=bytes)

        if 'scaleoffset' in creation_args:
            assert data is None or numpy.isfinite(data_copy).all()

        logging.getLogger(__name__).debug(
            'Creating dataset %s with shape %s, input dtype %s, '
            'creation_args %s',
            dataset_path,
            shape,
            repr(None if data_copy is None else data_copy.dtype),
            repr(creation_args)
        )
        self.create_dataset(
            dataset_path,
            data=data_copy,
            shape=shape,
            **creation_args
        )

    def __init__(self,
                 fname=None,
                 mode=None,
                 layout_version=None,
                 **kwargs):
        """
        Opens the given HDF5 file in the given mode.

        Args:
            fname:    The name of the file to open.

            mode:    The mode to open the file in (see hdf5.File).

            layout_version:    If the file does not exist, this is the version
                of the layout that will be used for its structure. Leave None
                to use the latest defined.

            kwargs:    Any additional arguments. Passed directly to h5py.File.

        Returns:
            None
        """

        if fname is None:
            assert mode is None
            super().__init__('memory_only',
                             mode='w',
                             driver='core',
                             backing_store=False)
        else:
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
                super().__init__(fname, mode, **kwargs)
            except IOError as details:
                raise HDF5LayoutError(
                    f'Problem opening {fname:s} in mode={mode:s}'
                    +
                    ''.join(format_exception(*exc_info()))
                ) from details

        layout_version_path, layout_version_attr = (
            self._layout_version_attribute
        )


        if fname is not None and old_file:
            layout_version = (
                self[layout_version_path].attrs[layout_version_attr]
            )

        (
            self._defined_elements,
            self._file_structure,
            self._file_structure_version
        ) = self._get_file_structure(layout_version)

        if fname is not None and not old_file:
            self[layout_version_path].attrs[layout_version_attr] = (
                self._file_structure_version
            )


    @staticmethod
    def collect_columns(destination,
                        name_head,
                        name_tail,
                        dset_name,
                        values):
        """
        If dataset is 1D and name starts and ends as given, add to destination.

        This function is intended to be passed to h5py.Group.visititems() after
        fixing the first 3 arguments using functools.partial.

        Args:
            destination(pandas.DataFrame):    The DataFrame to add matching
                datasets to. Datasets are added with column names given by the
                part of the name between `name_head` and `name_tail`.

            name_head(str):    Only datasets whose names start with this will be
                included.

            name_tail(str):    Only datasets whose names end with this will be
                included.

            dset_name(str):    The name of the dataset.

            values(array-like):    The values to potentially add as the new
                column.

        Returns:
            None
        """

        if (
                isinstance(values, h5py.Dataset)
                and
                dset_name.startswith(name_head)
                and
                dset_name.endswith(name_tail)
                and
                len(values.shape) == 1
        ):
            column_name = dset_name[len(name_head):]
            if name_tail:
                column_name = column_name[:-len(name_tail)]
            enum_transform = h5py.check_enum_dtype(values.dtype)
            if enum_transform is None:
                insert_values = values
            else:
                insert_values = numpy.empty(
                    values.shape,
                    dtype='S' + str(max(map(len, enum_transform.keys())))
                )
                for new, old in enum_transform.items():
                    insert_values[values[:] == old] = new.encode('ascii')
            destination.insert(len(destination.columns),
                               column_name,
                               insert_values)
#pylint: enable=too-many-ancestors
