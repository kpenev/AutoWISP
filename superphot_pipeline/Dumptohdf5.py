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
import hdf5_file
from superphot_pipeline.pipeline_exceptions import HDF5LayoutError
class makedatasets(ABC,h5py.File):
	 object=np.genfromtxt('10-465030_2_G1.fistar', name={id,x,y,bg,amp,s,d,k,fwhm,ellip,pa,flux,s/n,npix})
	 """ takes data from a test file(obviously needs to be generalized) and creates an n d array with the specified oolumnnames"""
	 f=h5py.File('fistar.hdf5','w')
	 dset=f.create_dataset("object",data=arr)
	 

