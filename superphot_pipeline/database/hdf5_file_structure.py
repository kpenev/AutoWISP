"""Define HDF5 file setting its structure from a database."""

from sqlalchemy.orm import contains_eager

from superphot_pipeline.hdf5_file import HDF5File
from superphot_pipeline.database.interface import db_session_scope

#Pylint false positive due to quirky imports.
#pylint: disable=no-name-in-module
from superphot_pipeline.database.data_model import\
    HDF5Product,\
    HDF5StructureVersion
#pylint: enable=no-name-in-module

#This is a h5py issue not an issue with this module
#pylint: disable=too-many-ancestors

#Class intentionally left abstract.
#pylint: disable=abstract-method
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

            for element_type in ['dataset', 'attribute', 'link']:
                self._defined_elements[element_type] = [
                    element.pipeline_key
                    for element in getattr(structure.structure_versions[0],
                                           element_type + 's')
                ]

        def create_result(structure):
            """Create the final result of the parent function."""

            result = dict()
            for element_type in ['datasets', 'attributes', 'links']:
                for element in getattr(structure.structure_versions[0],
                                       element_type):
                    result[element.pipeline_key] = element

            return result, str(structure.structure_versions[0].version)

        with db_session_scope() as db_session:
            query = db_session.query(
                HDF5Product
            ).join(
                HDF5Product.structure_versions
            ).options(
                contains_eager(
                    HDF5Product.structure_versions
                ).subqueryload(
                    HDF5StructureVersion.datasets
                )
            ).options(
                contains_eager(
                    HDF5Product.structure_versions
                ).subqueryload(
                    HDF5StructureVersion.attributes
                )
            ).options(
                contains_eager(
                    HDF5Product.structure_versions
                ).subqueryload(
                    HDF5StructureVersion.links
                )
            ).filter(
                HDF5Product.pipeline_key == self._product
            )

            if version is None:
                structure = query.order_by(
                    HDF5StructureVersion.version.desc()
                ).first()
            else:
                structure = query.filter(
                    HDF5StructureVersion.version == version
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
#pylint: enable=abstract-method

#pylint: enable=too-many-ancestors
