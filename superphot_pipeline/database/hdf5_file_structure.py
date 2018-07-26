"""Define HDF5 file setting its structure from a database."""

from io import BytesIO
from sqlalchemy.orm import subqueryload

from superphot_pipeline.hdf5_file import HDF5File
from superphot_pipeline.database.interface import db_session_scope
from superphot_pipeline.database.data_model import\
    HDF5Product,\
    HDF5StructureVersion

#This is a h5py issue not an issue with this module
#pylint: disable=too-many-ancestors
class HDF5FileDatabaseStructure(HDF5File):
    """HDF5 file with structure specified through the database."""

    @property
    def _elements(self):
        """See :meth:HDF5File._elements for description."""

        return self._defined_elements

    def _get_file_structure(self, version=None):
        """See :meth:HDF5File._get_file_structure for description."""

        def fill_elements(structure):
            """Fill self._defined_elements with all defined pipeline keys."""

            for element_type in ['data_set', 'attribute', 'link']:
                self._defined_elements[element_type] = [
                    element.pipeline_key
                    for element in getattr(structure.structure_versions[0],
                                           element_type + 's')
                ]

        def create_result(structure):
            """Create the final result of the parent function."""

            result = dict()
            for element_type in ['data_sets', 'attributes', 'links']:
                for element in getattr(structure.structure_versions[0],
                                       element_type):
                    result[element.pipeline_key] = element

            return result

        with db_session_scope() as db_session:
            structure = db_session.query(
                HDF5Product
            ).options(
                subqueryload(
                    HDF5Product.structure_versions
                ).subqueryload(
                    HDF5StructureVersion.data_sets
                )
            ).options(
                subqueryload(
                    HDF5Product.structure_versions
                ).subqueryload(
                    HDF5StructureVersion.attributes
                )
            ).options(
                subqueryload(
                    HDF5Product.structure_versions
                ).subqueryload(
                    HDF5StructureVersion.links
                )
            ).filter(
                HDF5Product.pipeline_key == self._product
            ).filter(
                HDF5StructureVersion.version == 0
            ).one()

            db_session.expunge_all()

        fill_elements(structure)
        return create_result(structure)

    def __init__(self, product, *args, **kwargs):
        """
        Open a file containing the given pipeline product.

        All arguments other than product are passed directly to
        HDF5File.__init__().
        """

        self._defined_elements = dict()
        self._product = product
        super().__init__(*args, **kwargs)

class DataReductionFile(HDF5FileDatabaseStructure):
    """Data reduction file with structure specified through the database."""

    @classmethod
    def _get_root_tag_name(cls):
        """The name of the root tag in the layout configuration."""

        return 'DataReduction'

    def __init__(self, *args, **kwargs):
        """See HDF5File for description of arguments."""

        super().__init__('data_reduction', *args, **kwargs)

#pylint: enable=too-many-ancestors

if __name__ == '__main__':
    dr_file = DataReductionFile('test.hdf5', 'w-')

    from xml.dom import minidom
    from xml.etree import ElementTree

    with open('example_structure.xml', 'wb') as xml:
        xml.write(
            b'<?xml-stylesheet type="text/xsl" href="hdf5_file_structure.xsl"?>'
            b'\n'
        )
        xml.write(
            minidom.parseString(
                ElementTree.tostring(
                    dr_file.layout_to_etree()
                )
            ).toprettyxml(indent='    ', encoding='UTF-8')
        )
