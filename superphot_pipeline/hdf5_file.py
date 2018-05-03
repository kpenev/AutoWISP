#Only a single class is defined so hardly makes sense split.
#pylint: disable=too-many-lines
"""Define a class for working with HDF5 files."""

from abc import ABC, abstractmethod
import h5py
import numpy
from astropy.io import fits
from io import StringIO
import re
import xml.dom.minidom as dom
import os
import os.path
from traceback import format_exception, print_exception
from sys import exc_info
from ast import literal_eval

from superphot_pipeline.pipeline_exceptions import HDF5LayoutError

git_id = '$Id$'

#This is a h5py issue not an issue with this module
#pylint: disable=too-many-ancestors
class HDF5File(ABC, h5py.File):
    """
    Base class for HDF5 pipeline products.

    Supports defining the structure from the database or XML file, as  well as
    generating markdown, and XML files describing the structure.

    Implements backwards compatibility for different versions of the structure
    of files.
    """

    _destination_versions = dict()

    @classmethod
    @abstractmethod
    def get_layout_root_tag_name(cls):
        """The name of the root tag in the layout configuration."""

    @property
    @abstractmethod
    def elements(self):
        """
        Identifying strings for the recognized elements of the HDF5 file.

        Shoul be a dictionary-like object with values being a set of strings
        containing the identifiers of the HDF5 elements and keys:

            * dataset: Identifiers for the datasets that could be included in
                the file.

            * attribute: Identifiers for the attributes that could be included
                in the file.

            * link: Identifiers for the links that could be included in
                the file.
        """
    @property
    @abstractmethod
    def element_uses(self):
        """
        A dictionary specifying what each dataset or property is used for.

        This structure has two keys: ``'dataset'`` and ``'attribute'`` each of
        which should contain a dictionary with keys ``self.elements['dataset']``
        or ``self.elements['attribute']`` and values are lists of strings
        specifying uses (only needed for generating documentation).
        """

    @property
    def default_destinations(self):
        """
        Dictionary of where to place newly created elements in the HDF5 file.

        There is an entry for all non-group elements as defined by
        self.get_element_type(). Each entry is a dictionary:

            * parent: The path to the parent group/dataset where the new
                entry will be created.

            * parent_type: The type of the parent - 'group' or 'dataset'.

            * name: The name to give to the new dataset. It may contain
                a substitution of %(ap_ind)?.

            * creation_args: For datasets only. Should specify additional
                arguments for the create_dataset method.

            * replace_nonfinite: For floating point datasets only. Specifies
                a value with which to replace any non-finite dataset entries
                before writing to the file. (Workaround the scaleoffset filter
                problem with non-finite values). After extracting a dataset, any
                values found to equal this are replaced by not-a-number.
        """

        return self._default_destinations

    @property
    def destinations(self):
        r"""
        Specifies the destinations for self.elements in the current file.

        See :attr:`default_destinations`\ .
        """

        return self._destinations

    @property
    @classmethod
    def destination_versions(cls):
        """
        Destiantions for self.elements for all known file structure versions.

        This is a dictionary indexed by configuration version number containing
        entries identical to self.default_destinations for each configured
        version of the HDF5 structure.
        """

        return cls._destination_versions

    @staticmethod
    def write_text_to_dataset(text,
                              h5group,
                              dset_path,
                              creation_args=None,
                              **attributes):
        """
        Adds ASCII text/file as a dateset to an HDF5 file.

        Args:
            text:    The text or file to add. If it is an open file, the
                contents is dumped, if it is a python2 string or a python3
                bytes, the value is stored.

            h5group:    An HDF5 group (could be the root group, i.e. an
                h5py.File opened for writing).

            dset_path:    The path for the new dataset, either absolute or
                relative to h5group.

            creation_args:    Keyword arguments to pass to
                :meth:`create_dataset`\ . If
                ``None``, defaults to ``dict(compression='gzip',
                compression_opts=9)``.

            compression_opts:    see same name argument
                in h5py.File.create_dataset.

            attributes:    Added as attributes with the same name to the
                the dataset.

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
        """
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
        """
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

    @staticmethod
    def get_hdf5_dtype(dtype_string, hdf5_element_type):
        """
        Return the dtype argument to when creating an HDF5 entry.

        Args:
            dtype_string:    The string from the XML or database configuration
                specifying the type.

            hdf5_element_type:    What is the element being created - dataset
                or attribute.

        Returns:
            dtype:    Whatever should be passed as the dtype argument when
                creating the given entry in the HDF5 file.
        """

        if dtype_string != 'S' or hdf5_element_type == 'attribute':
            return numpy.dtype(dtype_string)

        return h5py.special_dtype(vlen=str)

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
        return 'group'

    @classmethod
    def get_documentation_order(cls, element_type, element_id):
        """
        Return a sorting key for the elements in a documentation level.

        Args:
            element_type:    The type of entry this element corresponds to in
                the HDF5 file (group, dataset, attribute or link).

            element_id:    The identifying string of this element.

        Returns:
            sort_key:    An integer such that elements for which lower values
                are returned should appear before elements with higher values in
                the documentation if the two are on the same level.
        """

        offset = 1
        for check_type in ['attribute', 'group', 'link', 'dataset']:
            if element_type == check_type:
                return (
                    offset
                    +
                    (0 if element_type == 'group'
                     else cls.elements[element_type].index(element_id))
                )
            if element_type == 'group':
                offset += 1000
            else:
                offset += 100 * len(cls.elements[element_type])
        raise HDF5LayoutError("Unrecognized element type: '%s'"
                              %
                              element_type)

    @classmethod
    def _add_paths(cls, xml_part, parent_path='/', parent_type='group'):
        """
        Add the paths in a part of an XML document.

        Args:
            xml_part:    A part of an XML docuemnt to parse (parsed
                through xml.dom.minidom).

            parent_path:    The path under which xml_part lives.

            parent_type:    The type of entry the parent is ('group'
                or 'dataset').

        Returns:
            None
        """

        def parse_compression(compression_str):
            """
            Parses a string defining how to compress a dataset.

            Args:
                compression_str:    The string defiring the compression. Allowed
                    values (can be combined with ';'):

                    * gzip:<level>:    uses gzip based compression of the
                        specified level

                    * shuffle:    enables the shuffle filter.

                    * scaleoffset:<precision>:    Uses the scale-offset filter.
                        The precision is ignored if the dataset contains
                        integral values and if it contains floating point values
                        it specifies the number of digits after the decimal
                        point to preserve.

            Returns:
                compression_args:    A dictionary of extra arguments to pass to
                    the h5py create_dataset function in order to enable the
                    specified compression.
            """

            result = dict()
            for subfilter in compression_str.split(';'):
                if subfilter == 'shuffle':
                    result['shuffle'] = True
                else:
                    method, options = subfilter.split(':')
                    if method == 'scaleoffset':
                        result.update(scaleoffset=int(options))
                    elif method == 'gzip':
                        result.update(compression='gzip',
                                      compression_opts=int(options))
                    else:
                        raise HDF5LayoutError(
                            "Unrecognized compression filter '%s'!"
                            %
                            method
                        )
            return result

        def get_name(xml_element):
            """Return the tag of the XML element replacing if necessary."""

            if xml_element.hasAttribute('element_name'):
                return xml_element.getAttribute('element_name')
            return xml_element.tagName

        if not xml_part.hasAttribute('type'):
            raise HDF5LayoutError(
                "%sStructure configuration contains elemenst with no 'type' "
                "attribute."
                %
                cls.get_layout_root_tag_name
            )
        part_type = xml_part.getAttribute('type')

        if get_name(xml_part) == cls.get_layout_root_tag_name():
            if part_type != 'group':
                raise HDF5LayoutError(
                    "Root entry of %sStructure configuration has type='%s' "
                    "instead of 'group'"
                    %
                    (cls.get_layout_root_tag_name(), part_type)
                )
            part_path = '/'
        elif parent_path == '/':
            part_path = '/'+get_name(xml_part)
        else:
            part_path = parent_path+'/' + get_name(xml_part)

        if part_type not in ['group', 'attribute', 'dataset', 'link']:
            raise HDF5LayoutError(
                "%sStructure configuration contains '%s' with type '%s', only "
                "'group', 'attribute', 'dataset' and 'link' are allowed!"
                %
                (cls.get_layout_root_tag_name(), part_type, part_path)
            )

        if xml_part.hasAttribute('value'):
            if part_type == 'group':
                raise HDF5LayoutError(
                    "Value defined for the group '%s' in %sStructure "
                    "configuration."
                    %
                    (part_path, cls.get_layout_root_tag_name())
                )
            else:
                part_value = xml_part.getAttribute('value')
                cls._default_destinations[part_value] = dict(
                    parent=parent_path,
                    parent_type=parent_type,
                    name=get_name(xml_part)
                )
                this_destination = cls._default_destinations[part_value]
                if xml_part.hasAttribute('compression'):
                    if part_type != 'dataset':
                        raise HDF5LayoutError(
                            "Compression defined for the %s '%s' in "
                            "%sStructure configuration. Only datasets can "
                            "be compressed!"
                            %
                            (part_type, part_path,
                             cls.get_layout_root_tag_name())
                        )
                    else:
                        this_destination['creation_args'] = parse_compression(
                            xml_part.getAttribute('compression')
                        )
                else:
                    this_destination['creation_args'] = dict()

                if xml_part.hasAttribute('replace_nonfinite'):
                    assert part_type == 'dataset'
                    this_destination['replace_nonfinite'] = literal_eval(
                        xml_part.getAttribute('replace_nonfinite')
                    )
                if xml_part.hasAttribute('dtype'):
                    this_destination['creation_args'].update(
                        dtype=cls.get_hdf5_dtype(
                            xml_part.getAttribute('dtype'),
                            part_type
                        )
                    )

        if xml_part.hasChildNodes():
            if xml_part.getAttribute('type') == 'attribute':
                raise HDF5LayoutError(
                    "Configuration for %s file has structure under the "
                    "attribute '%s'!"
                    %
                    (cls.get_layout_root_tag_name(), part_path)
                )
            child = xml_part.firstChild
            while child:
                if not hasattr(child, 'data') or child.data.strip():
                    child_type = child.getAttribute('type')
                    if part_type == 'dataset' and child_type != 'attribute':
                        raise HDF5LayoutError(
                            "HDF5 file structure  configuration involves a "
                            "'%s' ('%s') under the dataset '%s', only "
                            "attributes are allowed."
                            %
                            (
                                child_type,
                                get_name(child),
                                part_path
                            )
                        )
                    cls._add_paths(child, part_path, part_type)
                child = child.nextSibling

    #TODO: Fix to work with markdown rather than markup.
    def generate_wiki(self, xml_part, current_indent='', format_for='TRAC'):
        """
        Returns the part of the wiki corresponding to a part of the XML tree.

        Args:
            xml_part:    The part of the XML tree to wikify.

            current_indent:    The indent to use for the root element
                of xml_part.

        Returns:
            str:
                the wiki text to add (newlines and all).
        """

        camel_case_rex = re.compile('[A-Z][a-z]+([A-Z][a-z]+)+')

        def camel_case(string):
            """Returns True iff the given string is CamelCase."""

            match = camel_case_rex.match(string)
            return match is not None and match.end() == len(string)

        def format_description(description):
            """The string to add to the wiki for the given description."""

            if format_for == 'TRAC':
                return '- [[span(%s, style=color: grey)]]' % description
            return description


        def format_group(name, description=""):
            """
            Returns the string to represent a group.

            Args:
                name:    The name of the group.

            Returns:
                str:
                    The properly formatted string stating the name of
                    the given group, as it should be added to the wiki.
            """

            if format_for == 'TRAC':
                if camel_case(name):
                    name = '!' + name
                result = (
                    '- [[span(%s, style=color: orange; font-size: 150%%, '
                    'id=anchor)]]'
                    %
                    name
                )
            else:
                result = '<li><b>' + name + '</b>'
            return result + format_description(description)

        def format_dataset(name, used_by, description, has_attributes):
            """
            Returns the string to represent a dataset.

            Args:
                name:    The name of the dataset.

                used_by:    A list of "things" that use the dataset.

                description:    A brief description of the dataset.

                has_attributes:    Does the dataset have attributes.

            Returns:
                str:
                    The properly formatted string containing all
                    the supplied information about the dataset, as it should be
                    added to the wiki.
            """

            if format_for == 'TRAC':
                if used_by:
                    result = "- '''__%s__'''^%s^: " % (name, ', '.join(used_by))
                else:
                    result = '- __%s__: ' % name
                return result + format_description(description)
            else:
                result = '<li><u>' + name + '</u>: '
                if used_by:
                    result += '<sup>' + ', '.join(used_by) + '</sup>: '
                result += format_description(description)
                if not has_attributes:
                    result += '</li>'
                return result

        def format_attribute(name, used_by, description=""):
            """
            Returns the string to represent an attribute.

            Args:
                name:    The name of the attribute.

                used_by:    A list of "things"  that use the attribute

                description:    A brief description of the attribute (optional).

            Returns:
                str:
                    The properly formatted string containing all
                    the supplied information about the attribute, as it should
                    be added to the wiki.
            """

            formatted_name = ('!' + name if camel_case(name) else name)
            if format_for == 'TRAC':
                if used_by:
                    result = "- '''%s'''^%s^: " % (formatted_name,
                                                   ', '.join(used_by))
                else:
                    result = '- ' + formatted_name + ': '
                return result + format_description(description)
            else:
                return ('<li>'
                        +
                        name
                        +
                        ': '
                        +
                        format_description(description)
                        +
                        '</li>')

        def format_link(name, description):
            """
            Returns the string to represent a link.

            Args:
                name:    The name of the link.

                description:    A brief description of the link.

            Returns:
                str:
                    The properly formatted string containing all the
                    supplied information about the link, as it should be added
                    to the wiki.
            """

            formatted_name = ('!' + name if camel_case(name) else name)
            if format_for == 'TRAC':
                return ('- [[span(%s, style=color: DarkTurquoise)]]: '
                        %
                        formatted_name
                        +
                        format_description(description))
            return ('<li><i>'
                    +
                    name
                    +
                    '</i>: '
                    +
                    format_description(description)
                    +
                    '</li>')

        def format_header():
            """The header of the wiki page (legend, description, ...)"""

            if format_for == 'TRAC':
                legend_start = '= Legend: ='
                legend_end = ''
                format_start = '= Single Frame HDF5 File version %s =\n\n'
            else:
                legend_start = '# Legend:\n<ul>'
                legend_end = '    </ul>\n</ul>'
                format_start = '# Single Frame HDF5 File version %s\n\n'

            legend_end = ('' if format_for == 'TRAC' else '\n</ul>')
            result = (legend_start + '\n' + '    ' + format_group('group'))

            if format_for == 'GitHub':
                result += '\n    <ul>'

            result += ('\n'
                       +
                       ('        ' + format_dataset('dataset',
                                                    ['[usedby', '[...]]'],
                                                    'description',
                                                    False))
                       +
                       '\n'
                       +
                       ('        ' + format_attribute('attribute',
                                                      ['[usedby', '[...]]'],
                                                      '[description]'))
                       +
                       '\n'
                       +
                       '        ' + format_link('link', 'description'))

            if format_for == 'GitHub':
                result += '\n    </ul>'
            return (
                result
                +
                legend_end
                +
                '\n\nElements which depend on the aperture used for aperture'
                'photometry must contain a substitution of %(ap_ind)s '
                'in their names, unless only a single aperture is '
                'being used.\n\nElements which are required by the pipeline'
                ' are bold.\n\n'
                +
                format_start%xml_part.getAttribute('version')
            )

        def sort_children():
            """Orders all the child nodes as they should be processed."""

            child_nodes = []
            child = xml_part.firstChild
            while child:
                child_nodes.append(child)
                child = child.nextSibling
            result = sorted(
                child_nodes,
                key=lambda child: self.get_documentation_order(
                    child.getAttribute('type'),
                    child.getAttribute('value')
                )
            )
            return result

        part_type = xml_part.getAttribute('type')
        part_description = xml_part.getAttribute('description')
        if xml_part.tagName == self.get_layout_root_tag_name():
            result = format_header()
        else:
            hdf5_name = xml_part.tagName
            key = xml_part.getAttribute('value')
            result = '\n' + current_indent
            if part_type == 'group':
                result += format_group(hdf5_name, part_description)
            elif part_type == 'dataset':
                users = self.element_uses['dataset'].get(key, [])
                result += format_dataset(hdf5_name,
                                         users,
                                         part_description,
                                         xml_part.hasChildNodes())
            elif part_type == 'attribute':
                result += format_attribute(
                    hdf5_name,
                    self.element_uses['attribute'].get(key, []),
                    part_description
                )
            elif part_type == 'link':
                result += format_link(hdf5_name, part_description)

        if xml_part.hasChildNodes():
            if format_for != 'TRAC':
                result += '\n' + current_indent + '<ul>'
            new_indent = current_indent + '    '
            for child in sort_children():
                result += self.generate_wiki(child, new_indent, format_for)
            if format_for != 'TRAC':
                result += '\n' + current_indent + '</ul></li>'
        return result

    def _wiki_other_version_links(self, version_list, target_version):
        """Text to add to wiki to link to other configuration versions."""

        result = "\n\n= Other Versions: =\n"
        for version in version_list:
            if version != target_version:
                result += (
                    '\n * '
                    +
                    'Version %(ver)s: [wiki:%(product)sFormat_v%(ver)s]'
                    %
                    dict(version=version,
                         product=self.get_layout_root_tag_name())
                )
        return result

    @classmethod
    def configure_from_xml(cls, xml, project_id=0, make_default=False):
        """
        Defines the file structure from an xml.dom.minidom document.

        Args:
            xml:    The xml.dom.minidom document defining the structure.

            project_id:    The project ID this configuration applies to.

            make_default:    Should this configuration be saved as the
                default one?

        Returns:
            None
        """

        if hasattr(cls, '_add_custom_elements'):
            #pylint false positive: we check for this method before calling.
            #pylint: disable=no-member
            cls._add_custom_elements()
            #pylint: enable=no-member
        version = int(xml.firstChild.getAttribute('version'))
        cls._add_paths(xml.firstChild)
        cls._destination_versions[version] = (
            cls._default_destinations
        )
        cls._structure_project_id = None
        cls._default_project_id = None
        if make_default:
            cls._default_version = version
            cls._project_id = project_id
        else:
            cls._default_destinations = dict()
        cls._configured_from_db = False

    @classmethod
    def configure_from_db(cls,
                          db,
                          target_project_id=0,
                          target_version=None,
                          datatype_from_db=False,
                          save_to_file=None,
                          update_trac=False,
                          generate_markdown=False):
        """
        Reads the defined the structure of the file from the database.

        Args:
            db:    An instance of CalibDB connected to the calibration database.

            target_project_id:    The project ID to configure for (falls back to
                the configuration for project_id=0 if no configuration is found
                for the requested value. Default: 0.

            target_version:    The configuration version to set as default. If
                None (default), the largest configuration value found is used.

            datatype_from_db:    Should the information about data type be read
                form the database?

            save_to_file:    If not None, a file with the given names is created
                contaning an XML representation of the HDF5 file structure
                costructed from the database.

            generate_markdown:    Generates markdown files suitable for
                committing to a GitHub repository as documentation. If given,
                this argument should be a directory where the files should be
                saved. Otherwise it should be something that tests as False.

        Returns:
            None
        """

        def build_xml(db_config):
            """
            Builds a DOM structure from the database configuration.

            Args:
                db_config:    The entries in the database defining
                    the configuration.

            Returns:
                xml.dom.minidom:
                    A document with the configuration.
            """

            def create_elements(dom_document):
                """
                Creates DOM elements for all entries with no hierarchy.

                Args:
                    dom_document:    The DOM document to create elements with.

                Returns:
                    dict:
                        A dictionary indexed by
                        %(component)s.%(element)s of DOM elements for each entry
                        with all their attributes properly set.
                """

                hdf5_structure = dom_document.createElement(
                    cls.get_layout_root_tag_name()
                )
                hdf5_structure.setAttribute('type', 'group')

                result = {}
                result['.'] = dict(element=hdf5_structure)
                for db_entry in db_config:
                    if datatype_from_db:
                        (
                            component,
                            element,
                            parent,
                            h5name,
                            datatype,
                            compression,
                            replace_nonfinite,
                            description
                        ) = db_entry
                    else:
                        (
                            component,
                            element,
                            parent,
                            h5name,
                            compression,
                            replace_nonfinite,
                            description
                        ) = db_entry
                    element_id = component + '.' + element

                    if '.' not in parent:
                        parent = component + '.' + parent
                        if parent == element_id:
                            parent = '.'

                    new_element = dom_document.createElement(
                        h5name.replace(
                            '%', ''
                        ).replace(
                            '(', ''
                        ).replace(
                            ')', ''
                        )
                    )
                    element_type = cls.get_element_type(element_id.rstrip('.'))
                    new_element.setAttribute('type', element_type)
                    if not h5name.isalnum():
                        new_element.setAttribute('element_name', h5name)
                    if element_type != 'group':
                        new_element.setAttribute('value',
                                                 element_id.rstrip('.'))
                        if datatype_from_db:
                            new_element.setAttribute('dtype', datatype)
                        if (
                                element_type == 'dataset'
                                and
                                replace_nonfinite is not None
                        ):
                            new_element.setAttribute('replace_nonfinite',
                                                     repr(replace_nonfinite))
                    if compression is not None:
                        new_element.setAttribute('compression', compression)
                    new_element.setAttribute('description', description)
                    result[element_id] = dict(element=new_element,
                                              parent=parent)
                return result

            result = dom.Document()
            element_dict = create_elements(result)
            for element_id, element in element_dict.items():
                if element_id != '.':
                    parent = element['parent']
                    if parent not in element_dict:
                        if parent[-1] != '.':
                            raise HDF5LayoutError(
                                "Requested that '%s' be placed under '%s' but "
                                "'%s' is not configured."
                                %
                                (element_id, parent, parent)
                            )
                        else:
                            parent = '.'
                    element_dict[parent]['element'].appendChild(
                        element['element']
                    )
            result.appendChild(element_dict['.']['element'])
            return result

        if hasattr(cls, '_add_custom_elements'):
            cls._add_custom_elements()
        version_list = None
        structure_project_id = target_project_id
        default_project_id = target_project_id
        while version_list is None:
            version_list = db(
                (
                    'SELECT `version` FROM `%s` WHERE `project_id`=%%s GROUP BY'
                    ' `version`'
                    %
                    cls._db_table
                ),
                (structure_project_id,),
                no_simplify=True
            )
            if version_list is None:
                if structure_project_id == 0:
                    raise HDF5LayoutError(
                        'No %s structure defined for project id %d or 0!'
                        %
                        (cls.get_layout_root_tag_name(), target_project_id)
                    )
                structure_project_id = 0
            else:
                version_list = [r[0] for r in version_list]
        if target_version is None:
            target_version = max(version_list)
        if target_version not in version_list:
            raise HDF5LayoutError(
                "No configuration found for project ID %d, version %d in `%s`"
                %
                (target_project_id, target_version, cls._db_table)
            )
        if update_trac:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            ssl_context.verify_flags = ssl.VERIFY_DEFAULT
            wiki_server = xmlrpclib.ServerProxy(
                'https://kpenev:shakakaa@hat.astro.princeton.edu/projects/'
                'HATreduc/login/rpc',
                context=ssl_context
            ).wiki
        for version in version_list:
            query = 'SELECT `component`, `element`, `parent`, `hdf5_name`,'
            if datatype_from_db:
                query += ' `dtype`,'
            query += (' `compression`, `replace_nonfinite`, `description` '
                      'FROM `%s` WHERE `project_id`=%%s AND `version`=%%s')
            db_config = db(query % cls._db_table,
                           (structure_project_id, version),
                           no_simplify=True)
            xml = build_xml(db_config)
            xml.firstChild.setAttribute('version', str(version))
            if save_to_file is not None:
                f = open(save_to_file % dict(PROJID=structure_project_id,
                                             VERSION=version),
                         'w')
                f.write(xml.toprettyxml())
                f.close()
            if update_trac or generate_markdown:
                page_name = cls.get_layout_root_tag_name() + 'Format'
                page_text = cls.generate_wiki(
                    xml.firstChild,
                    format_for=('TRAC' if update_trac else 'GitHub')
                )
                if version != target_version:
                    page_name += '_v' + str(version)
                elif update_trac:
                    page_text += cls._wiki_other_version_links(
                        version_list,
                        version
                    )
                if update_trac:
                    wiki_server.putPage(page_name, page_text, dict())
                else:
                    f = open(join_paths(generate_markdown, page_name + '.md'),
                             'w')
                    f.write(page_text)
                    f.close()
            cls.configure_from_xml(xml,
                                   structure_project_id,
                                   version == target_version)
        cls._structure_project_id = structure_project_id
        cls._default_project_id = default_project_id
        cls._calibration_station = db.station
        cls.configure_filenames(db)
        cls._configured_from_db = True

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
                 project_id=None,
                 db_config_version=None):
        """Opens the given HDF5 file in the given mode."""

        old_file = exists(fname)
        if mode[0] != 'r':
            path = dirname(fname)
            try:
                os.makedirs(path)
            except OSError:
                if not exists(path):
                    raise
        try:
            h5py.File.__init__(self, fname, mode)
        except IOError as error:
            raise HDF5LayoutError(
                'Problem opening %s in mode=%s'%(fname, mode)
                +
                ''.join(format_exception(*exc_info()))
            )
        if mode[0] == 'r' or (mode == 'a' and old_file):
            found_project_id = False
            if self._configured_from_db:
                for self._destinations\
                        in\
                        self.destination_versions.values():
                    try:
                        structure_project_id = HDF5File.get_attribute(
                            self,
                            'file_structure.project_id',
                            project_id=project_id
                        )
                        found_project_id = True
                        break
                    except Error.HDF5:
                        pass
                assert found_project_id
                if self._structure_project_id is not None:
                    assert structure_project_id == self._structure_project_id
                self._project_id = (
                    self._default_project_id
                    if project_id is None else
                    project_id
                )
                self._version = HDF5File.get_attribute(
                    self,
                    'file_structure.version',
                    project_id=project_id
                )
                self._destinations = self.destination_versions[
                    self._version
                ]
            else:
                self._project_id = None
                self._version = None
                self._destinations = self._default_destinations
        else:
            if project_id is None:
                self._project_id = self._default_project_id
            else:
                self._project_id = project_id
            self._version = (self._default_version
                             if db_config_version is None else
                             db_config_version)
            self._destinations = self.destination_versions[self._version]
            self.add_attribute(
                'file_structure.project_id',
                (
                    0
                    if self._structure_project_id is None else
                    self._structure_project_id
                )
            )
            self.add_attribute('file_structure.version', self._version)
#pylint: enable=too-many-ancestors
