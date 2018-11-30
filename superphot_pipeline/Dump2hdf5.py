from sqlalchemy.orm import contains_eager
from superphot_pipeline.hdf5_file import HDF5File
import os
import numpy as np
import scipy as sp
from superphot_pipeline.database.interface import db_session_scope
from superphot_pipeline.database.hdf5_file_structure import HDF5FileDatabaseStructure
#Pylint false positive due to quirky imports.
#pylint: disable=no-name-in-module
from superphot_pipeline.database.data_model import HDF5Product,HDF5StructureVersion
#pylint: enable=no-name-in-module

#This is a h5py issue not an issue with this module
#pylint: disable=too-many-ancestors

#Class intentionally left abstract.
#pylint: disable=abstract-method

class DataReduction(HDF5FileDatabaseStructure):
    """The initial goal is to convert one frame to an hdf5 dataset and add attributes"""
    def __init__(self,fname,mode):

        self.fistarcolumn_names=['id', 'x', 'y', 'bg', 'amp', 's', 'd', 'k', 'fwhm', 'ellip', 'pa',
                                                 'flux', 'ston', 'npix']
        super().__init__('data_reduction',fname,mode)
    def add_fistar(self,filename):
        transitarray = np.genfromtxt(filename,names=self.column_names)
        for column_name in self.fistarcolumn_names:
            self.add_dataset(dataset_key='srcextract.sources',data=transitarray[column_name],srcextract_version=0,srcextract_column_name=column_name)

    def add_catalogue(self, filename):
        transitarray = np.genfromtxt(filename, dtype=None, name=True, deletechars='')
        transitarray.dtype.names = [name.split('[', 1)[0] for name in transitarray.dtype.names]
        for column_name in transitarray.dtype.names:
            self.add_dataset(dataset_key='srcextract.sources', data=transitarray[column_name], srcextract_version=0,
                             srcextract_column_name=column_name)
    def add_match(self,filename,catalogue_cols,cataloguefilename, fistarfilename):
        fistararray = np.genfromtxt(fistarfilename, names=self.column_names)

        cataloguearray = np.genfromtxt(cataloguefilename, dtype=None, name=True, deletechars='')

        cataloguearray.dtype.names = [name.split('[', 1)[0] for name in cataloguearray.dtype.names]

        matcharray=np.genfromtxt(filename,dtype=None,name=['cat_id','fistar_id'],usecols=(0,catalogue_cols))

        sortedcataloguearray=sp.argsort(cataloguearray[:,0],axis=-1,'quicksort')

        sortedfistararray = sp.argsort(fistararray[:,0], axis=-1, 'quicksort')

        sortedcataloguearray1=sp.searchsorted(matcharray[:,0],unsortedcataloguearray)

        sortedfistararray1=sp.searchsorted(matcharray[:,1],unsortedfistararray)

        matchsorted=np.concatenate(sortedcataloguearray,sortedfistararray)

        for i in range(0,2)
            self.add_dataset(dataset_key='srcextract.sources', data=matchsorted[i], srcextract_version=0,
                             srcextract_column_name=column_name)

    @classmethod
    def _get_root_tag_name(cls):
        return 'Data Reduction'
if os.path.exists('fname5'):
    os.remove('fname5')
#A1=DataReduction('fname5','a')
#A1.add_fistar('10-465009_2_G2.fistar')
#A1.close()
A1=DataReduction('fname5','a')
A1.match()
