"""Define HDF5 file setting its structure from a database."""

from sqlalchemy.orm import contains_eager

from autowisp.hdf5_file import HDF5File
from autowisp.database.interface import Session

# Pylint false positive due to quirky imports.
# pylint: disable=no-name-in-module
from autowisp.database.data_model import HDF5Product, HDF5StructureVersion

# pylint: enable=no-name-in-module

# This is a h5py issue not an issue with this module
# pylint: disable=too-many-ancestors


# Class intentionally left abstract.
# pylint: disable=abstract-method
class HDF5FileDatabaseStructure(HDF5File):
    """HDF5 file with structure specified through the database."""

    @property
    def elements(self):
        """See :meth:HDF5File.elements for description."""

        return self._defined_elements

    @classmethod
    def get_file_structure(cls, version=None):
        """See :meth:HDF5File.get_file_structure for description."""

        def get_defined_elements(structure):
            """Fill cls._defined_elements with all defined pipeline keys."""

            defined_elements = {}
            for element_type in ["dataset", "attribute", "link"]:
                defined_elements[element_type] = set(
                    element.pipeline_key
                    for element in getattr(
                        structure.structure_versions[0], element_type + "s"
                    )
                )
            return defined_elements

        def create_result(structure):
            """Create the final result of the parent function."""

            file_structure = {}
            for element_type in ["datasets", "attributes", "links"]:
                for element in getattr(
                    structure.structure_versions[0], element_type
                ):
                    file_structure[element.pipeline_key] = element

            return (
                get_defined_elements(structure),
                file_structure,
                str(structure.structure_versions[0].version),
            )

        if version is not None:
            version = int(version)
        if not hasattr(cls, "_file_structure"):
            cls._file_structure = {}
        if version in cls._file_structure:
            return cls._file_structure[version]

        # False positive
        # pylint: disable=no-member
        with Session.begin() as db_session:
            # pylint: enable=no-member
            query = (
                db_session.query(HDF5Product)
                .join(HDF5Product.structure_versions)
                .options(
                    contains_eager(HDF5Product.structure_versions).subqueryload(
                        HDF5StructureVersion.datasets
                    )
                )
                .options(
                    contains_eager(HDF5Product.structure_versions).subqueryload(
                        HDF5StructureVersion.attributes
                    )
                )
                .options(
                    contains_eager(HDF5Product.structure_versions).subqueryload(
                        HDF5StructureVersion.links
                    )
                )
                .filter(HDF5Product.pipeline_key == cls._product())
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

        cls._file_structure[version] = create_result(structure)
        if version is None:
            cls._file_structure[
                int(structure.structure_versions[0].version)
            ] = cls._file_structure[None]
        return cls._file_structure[version]


# pylint: enable=abstract-method

# pylint: enable=too-many-ancestors
