# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 12:23:59 2021
@author: Ashkan

Goal: Calculating the transformation matrix as:
    (Ra,Dec) > projection(gnemonic) > (zeta,eta) > transformation (T) > (x, y)
(Ra,Dec): on the sky which will be transformed to zeta(or xi) and eta
T: Transformation Matrix which we are looking for
(x,y): coordinates on the frame

This will be done in 3 major steps:
    1. Get initial Transformation (T) using astrometry.net
    2. Building a KD tree
    3. Finding new T with much more matched sources
"""

import glob
import numpy as np
import math
from astropy.io import fits as pyfits
from scipy import linalg
from scipy import spatial
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
#from superphot_pipeline import DataReductionFile

#from superphot_pipeline.astrometry.map_projections import gnomonic_projection
#from superphot_pipeline.astrometry.map_projections import gnomonic_projection_inv


# should be used from the PhotometryPipeline/superphot_pipeline/astrometry/map_projections.py but there are
# some edits, like we do not have output there but here we do

def read_inputs():
    """

    Read all except DR file
    """
    #corr_file = input("Enter the name of the correlation file: ")   #corr.csv
    #fistar_file = input("Enter the name of the fistar file: ")      #10-465000_2_G1_250.fistar
    #catalog_file = input("Enter the name of the catalog file: ")    #cat_sources.ucac4
    #order = int(input("Enter the desired order of transformation: "))    # ex: 3
    
    corr_file = "corr.csv"
    fistar_file = "10-465000_2_G1_250.fistar"
    catalog_file = "cat_sources.ucac4"
    order = 2   # ex: 3
    
   
    xy_corr = np.genfromtxt(corr_file, delimiter=",", usecols=(0, 1))
    radec_corr = np.genfromtxt(corr_file, delimiter=",", usecols=(6, 7))
    xy_extracted = np.genfromtxt(fistar_file, usecols=(1, 2))   
    radec_catalog = np.genfromtxt(catalog_file, skip_header = 1, usecols=(1, 2))   
    #x_corr = xy_corr[:,0:1]
    #y_corr = xy_corr[:,1:2]
    #a,b = xy_extracted.shape
    #c,d = radec.shape
    return xy_corr, radec_corr, xy_extracted, radec_catalog, order
    

def read_dr_file():
    """
    Read some variable (existing in the header of the fits file) from the dr file 
    """
    #dr_fname = input("Enter the name of the DR file: ")
    
    #with DataReductionFile(dr_fname, 'r') as dr_file:
    #    header = dr_file.get_frame_header()
    #    ra_cent = header.get("RA")*(np.pi/12)          #convert from hr to radian (see line 64)
    #    dec_cent = header.get("Dec")*(np.pi/180.0)     # convert from degree to radian
    #    x_cent = header.get(ZNAXIS1)/2 # Length/2
    #    y_cent = header.get(ZNAXIS1)/2 # Width/2
    ra_cent = 10.2994169770979*(np.pi/12)
    dec_cent = 45.6471*(np.pi/180.0)
    x_cent = 3983/2.0
    y_cent = 2659/2.0
    
    return ra_cent, dec_cent, x_cent, y_cent


def projection(radec, ra_cent, dec_cent):  
    """
    docstring ...
    """
    degree_to_rad = np.pi / 180.0
    ra_diff = (radec[:,0]  * degree_to_rad - ra_cent)
    cos_ra_diff = np.cos(ra_diff)
    cos_source_dec = np.cos(radec[:,1] * degree_to_rad)
    cos_center_dec = np.cos(dec_cent)
    sin_source_dec = np.sin(radec[:,1] * degree_to_rad)
    sin_center_dec = np.sin(dec_cent)
    denominator = (sin_center_dec * sin_source_dec +
                   cos_center_dec * cos_source_dec * cos_ra_diff) * degree_to_rad

    projected = np.zeros(radec.shape, dtype=None)
    projected[:,0] = (cos_source_dec * np.sin(ra_diff)) / denominator
    projected[:,1] = (
                cos_center_dec * sin_source_dec
                -
                sin_center_dec * cos_source_dec * cos_ra_diff
                ) / denominator
    return projected


def inv_projection (zeta_eta, RA_cent, Dec_cent): 
    """
    Inverse projection from (zeta, eta) to (RA, DEC)
    Should be added to where we first used projection and use it there
    RA_cent and Dec_cent are in Radian 
    """
    unprojected = np.zeros(zeta_eta.shape, dtype=None)
        
    rho = np.sqrt(zeta_eta[:,0]**2 + zeta_eta[:,1]**2)
    c = np.arctan(rho)
    denominator = rho*np.cos(Dec_cent)*np.cos(c) - zeta_eta[:,1]*np.sin(Dec_cent)*np.sin(c)

    
    RA = RA_cent + np.arctan((zeta_eta[:,0]*np.sin(c))/denominator)
    Dec = np.arcsin(np.cos(c)*np.sin(Dec_cent)+(zeta_eta[:,1]*np.sin(c)*np.cos(Dec_cent))/rho)
    
    unprojected[:,0] = RA
    unprojected[:,1] = Dec
    
    return unprojected
    

def transformation_matrix(order, zeta, eta):
    """
    Constructing matrix M including zeta, eta
    """
    m = np.ones((eta.shape[0],1))
    for i in range(1, order+1):
        for j in range(i+1):
            m = np.block([m,zeta**(i-j)*eta**j])
    return m
    
    
def func(z, *data):
    """
    Constructing the two non linear function to be solved for new center (zeta, eta)
    """
    trans_x, trans_y, x_cent, y_cent, order = data
    
    zetac = z[0]
    etac = z[1]
   
    f = np.empty((2))
  #  print('f',f)
  #  f[0] = trans_x[0,0] + trans_x[1,0]*zetac + trans_x[2,0]*etac + trans_x[3,0]*zetac**2 +\
  #  trans_x[4,0]*zetac*etac + trans_x[5,0]*etac**2 - x_cent
   
  #  f[1] = trans_y[0,0] + trans_y[1,0]*zetac + trans_y[2,0]*etac + trans_y[3,0]*zetac**2 +\
 #   trans_y[4,0]*zetac*etac + trans_y[5,0]*etac**2 - y_cent



    f[0] = trans_x[0,0] - x_cent
    f[1] = trans_y[0,0] - y_cent
    #print('order',order)
    #for k in range(1,2*(order+1)):
    
    k = 1    #print('k',k)
    for i in range(1, order + 1):
        for j in range(i+1):
            #print("k, i, j ",k,i,j)
            f[0] = f[0] + trans_x[k, 0]*zetac**(i-j)*etac**(j)
            f[1] = f[1] + trans_y[k, 0]*zetac**(i-j)*etac**(j)
            k = k +1    
    #print("finish func")
    return f






def astrometry():
    """
    Main function: After we get the initial transformation, the rest is just iterating the steps to get new transformations 
    """
    
    xy_corr, radec_corr, xy_extracted, radec_catalog, order = read_inputs()
    ra_cent, dec_cent, x_cent, y_cent = read_dr_file()
    #print(ra_cent, dec_cent)
    #s =input()
    
    projected = projection(radec_corr, ra_cent, dec_cent)
    
    
    zeta = projected[:,0:1]
    eta = projected[:,1:2]
    
    mt = transformation_matrix(order, zeta, eta)   # Matrix (m) containing zeta, eta. Used to find Transformation(t)    
    trans_x, resid_x, rank_x, sigma_x = linalg.lstsq(mt, xy_corr[:,0:1])
    trans_y, resid_x, rank_y, sigma_y = linalg.lstsq(mt, xy_corr[:,1:2])
    
    print('\n Initial Transformation matrix for X components:')
    print(trans_x)
    print('\n Initial Transformation matrix for Y components:')
    print(trans_y)
    
    iteration(order, trans_x, trans_y)


# !!! we do not use resid, rank, sigma! delete if not necessary. !!! 


def iteration(order, trans_x, trans_y):
    """
    Iterate the process until we get a transformation that its difference from the previous one is less than a threshold
    """
    counter = 0
    
    xy_corr, radec_corr, xy_extracted, radec_catalog, order = read_inputs()
    ra_cent, dec_cent, x_cent, y_cent = read_dr_file()
    
    while True:
        counter = counter + 1
        if counter > 1:
            trans_x = trans_x_new
            trans_y = trans_y_new
            ra_cent = ra_cent_new
            dec_cent = dec_cent_new
        
        projected = projection(radec_catalog, ra_cent, dec_cent)

        zeta = projected[:,0:1]
        eta = projected[:,1:2]
   # ones = np.ones((projected.shape[0],1)) # col vector with len = # of our extracted sources from catalog
    
        mxy = transformation_matrix(order, zeta, eta) 
    # np.block([ones,zeta,eta,zeta**2,zeta*eta,eta**2])
        
        x_transformed = mxy @ trans_x 
        y_transformed = mxy @ trans_y 

        xy_transformed = np.block([x_transformed, y_transformed])
        #print("xy_transformed", xy_transformed.shape)



        kdtree = spatial.KDTree(xy_extracted)
        d, ix = kdtree.query(xy_transformed, distance_upper_bound = 1.5)
        n = 0

# Find the number of matched sources >>> I dont think we need this.
        for i in d:
            if not math.isinf(i):
                n = n + 1
        print("# of matched:",n)
        matched_sources = np.zeros((n,4),dtype=None)

# Put all matched sources together
# j is to find index in fistar
# k is to find index in ra dec
        j = 0
        k = -1
# =============================================================================
        for i in ix:
            k = k + 1
            if i != 5957:
                matched_sources[j, 0] = radec_catalog[k, 0]
                matched_sources[j, 1] = radec_catalog[k, 1]
                matched_sources[j, 2] = xy_extracted[i, 0]
                matched_sources[j, 3] = xy_extracted[i, 1]
                j = j + 1

############################# Isn't it better to use the projected of our header RA/Dec? #############################
        zGuess = np.array([np.mean(zeta), np.mean(eta)])
        data = (trans_x, trans_y, x_cent, y_cent, order)
        z = fsolve(func, zGuess, args = data)
        print("finish fsolve")
        zetaeta_cent = np.reshape(z,(1,2))    # Reshape from (2,) to (1,2)
        #print("zetaeta_cent", zetaeta_cent)
#This is in degree, since when we derived T, zeta and eta were in degrees.
        zetaeta_cent[0,0] = zetaeta_cent[0,0]*np.pi/180
        zetaeta_cent[0,1] = zetaeta_cent[0,1]*np.pi/180

# With those (zeta, eta) we use the inverse projection to find new (RAC, DECC)

        unprojected_cent = inv_projection(zetaeta_cent, ra_cent, dec_cent)


#print("RAC, DECC old: ", RA_cent, Dec_cent)
#print("RAC, DECC new: ", unprojected)

# {Replace RA_cent, Dec_cent with new values in unprojected}
# We found new RA_cent and Dec_cent
        ra_cent_new = unprojected_cent[0,0]
        dec_cent_new = unprojected_cent[0,1]

        projected_new = projection(matched_sources, ra_cent_new, dec_cent_new)

        x_matched_sources = matched_sources[:,2:3]
        y_matched_sources = matched_sources[:,3:4]
        zeta = projected_new[:,0:1]
        eta = projected_new[:,1:2]
        #ones = np.ones((n,1)) # col vector with len = # of our extracted sources
        mt = transformation_matrix(order, zeta, eta)
     #   np.block([ones,zeta,eta,zeta**2,zeta*eta,eta**2])


        trans_x_new, resid, rank, sigma = linalg.lstsq(mt, x_matched_sources)
        trans_y_new, resid, rank, sigma = linalg.lstsq(mt, y_matched_sources)

        print('\n new Transformation matrix for X components:')
        print(trans_x_new)
        print('\n new Transformation matrix for Y components:')
        print(trans_y_new)


        threshold = 0.1
    
        diff = np.block([trans_x_new - trans_x, trans_y_new - trans_y])
        if np.count_nonzero(diff > threshold) == 0:
            with open('reg.reg', 'w') as f_out:
                for x, y in xy_transformed:
                    print(
                        'box({xc!r},{yc!r},{w!r},{h!r},0) # color=green'.format(
                            xc=x,
                            yc=y,
                            w=8,
                            h=8
                      ), file = f_out
                    )  

            print('finish!, # of iteration =',counter)
            break



##### RUN!!!! ####

astrometry()














