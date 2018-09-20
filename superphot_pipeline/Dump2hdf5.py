from sqlalchemy.orm import contains_eager
from superphot_pipeline.hdf5_file import HDF5File
import numpy as np
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

class Conversion(HDF5FileDatabaseStructure):
    """The initial goal is to convert one frame to an hdf5 dataset and add attributes"""
    def __init__(self,fname,mode):
        self.Transitarray=np.genfromtxt('10-465009_2_G2.fistar',names=['id','x','y','bg','amp','s','d','k','fwhm','ellip','pa','flux','s/n','npix'])
        print (self.Transitarray)
        super().__init__('datareduction',fname,mode)
    @classmethod
    def _get_root_tag_name(cls):
        return 'Data Reduction'

