from sqlalchemy.orm import contains_eager
from autowisp.hdf5_file import HDF5File
import os
import numpy as np
import scipy as sp
from autowisp.database.interface import Session
from autowisp.database.hdf5_file_structure import HDF5FileDatabaseStructure

# Pylint false positive due to quirky imports.
# pylint: disable=no-name-in-module
from autowisp.database.data_model import HDF5Product, HDF5StructureVersion

# pylint: enable=no-name-in-module

# This is a h5py issue not an issue with this module
# pylint: disable=too-many-ancestors

# Class intentionally left abstract.
# pylint: disable=abstract-method


class DataReduction(HDF5FileDatabaseStructure):
    """The initial goal is to convert one frame to an hdf5 dataset and add attributes"""

    def __init__(self, fname, mode):

        self.fistarcolumn_names = [
            "id",
            "x",
            "y",
            "bg",
            "amp",
            "s",
            "d",
            "k",
            "fwhm",
            "ellip",
            "pa",
            "flux",
            "ston",
            "npix",
        ]
        super().__init__("data_reduction", fname, mode)

    def add_fistar(self, filename):
        """
        Creates datasets out of the columns in an extracted source file

        Args:
            filename: Name of the extracted source file


        Returns:
             None
        """
        transitarray = np.genfromtxt(filename, names=self.fistarcolumn_names)
        for column_name in self.fistarcolumn_names:
            self.add_dataset(
                dataset_key="srcextract.sources",
                data=transitarray[column_name],
                srcextract_version=0,
                srcextract_column_name=column_name,
            )

    def add_catalogue(self, filename):
        """
        Creates datasets out of the columns in a catalogue file

        Args:
            filename: Name of the catalogue file


        Returns:
             None
        """
        transitarray = np.genfromtxt(
            filename, dtype=None, names=True, deletechars=""
        )
        transitarray.dtype.names = [
            name.split("[", 1)[0] for name in transitarray.dtype.names
        ]
        for column_name in transitarray.dtype.names:
            self.add_dataset(
                dataset_key="catalogue.columns",
                data=transitarray[column_name],
                catalogue_version=0,
                catalogue_column_name=column_name,
            )

    def add_match(self, matchfilename, cataloguefilename, fistarfilename):
        """
        Creates a dataset for showing the correspondence between catalogue and extracted sources

        Args:
            matchfilename:    Name of the match file

            cataloguefilename:    Name of the catalogue file

            fistarfilename:    Name of the extracted sources file

        Returns:
             None
        """
        fistararray = np.genfromtxt(
            fistarfilename, names=self.fistarcolumn_names
        )
        cataloguearray = np.genfromtxt(
            cataloguefilename, dtype=None, names=True, deletechars=""
        )
        cataloguearray.dtype.names = [
            name.split("[", 1)[0] for name in cataloguearray.dtype.names
        ]
        catalogue_cols = len(cataloguearray.dtype.names)
        matcharray = np.genfromtxt(
            matchfilename,
            dtype=None,
            names=["cat_id", "fistar_id"],
            usecols=(0, catalogue_cols),
        )
        fistarsorter = sp.argsort(fistararray["id"])
        cataloguesorter = sp.argsort(cataloguearray["ID"])
        match = np.empty([matcharray.size, 2], dtype=int)
        match[:, 0] = cataloguesorter[
            sp.searchsorted(
                cataloguearray["ID"],
                matcharray["cat_id"],
                sorter=cataloguesorter,
            )
        ]
        match[:, 1] = fistarsorter[
            sp.searchsorted(
                fistararray["id"], matcharray["fistar_id"], sorter=fistarsorter
            )
        ]
        self.add_dataset(
            dataset_key="skytoframe.matched", data=match, skytoframe_version=0
        )

    @classmethod
    def _get_root_tag_name(cls):
        return "Data Reduction"


if os.path.exists("fname5"):
    os.remove("fname5")
# A1=DataReduction('fname5','a')
# A1.add_fistar('10-465009_2_G2.fistar')
# A1.close()
A1 = DataReduction("fname5", "a")
A1.add_match(
    "10-465258_2_R1.fistar.match", "test.ucac4", "10-465258_2_R1.fistar"
)
A1.close()
