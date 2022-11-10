# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 12:23:59 2021
@author: Ashkan

Goal: Calculating the transformation matrix as:
    (Ra,Dec) > projection(gnemonic) > (zeta,eta) > transformation (T) > (x, y)80
(Ra, Dec): on the sky which will be transformed to zeta(or xi) and eta
Trans_mat: Transformation Matrix which we are looking for
(x, y): coordinates on the frame

This will be done in 4 major steps:
    1. Get initial Transformation (T) using astrometry.net
    2. Building a KD tree
    3. Finding new T with much more matched sources
    4. Repeat the steps to get a better trnsformation
"""

import glob
import numpy as np
from astropy.io import fits as pyfits
from scipy import linalg
from scipy import spatial
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
from numpy.lib.recfunctions import structured_to_unstructured
#from superphot_pipeline import DataReductionFile

#from superphot_pipeline.astrometry.map_projections import gnomonic_projection
#from superphot_pipeline.astrometry.map_projections import gnomonic_projection_inv


def read_inputs():
    """
    Read all except DR file
    
    Args:
        z:
         
    Returns:
        ...
    """
        
    corr_file = "corr.csv"
    fistar_file = "10-465000_2_G1_250.fistar"
    catalog_file = "cat_sources.ucac4"
    order = 3
    
    xy_corr = \
    np.genfromtxt(corr_file, delimiter=",", usecols=(0,1), names=['x','y'])
    radec_corr = \
    np.genfromtxt(corr_file, delimiter=",", usecols=(6,7), names=['RA','Dec'])
    #radec_corr.reshape(len(radec_corr), 2)
    #radec_corrr = np.genfromtxt(corr_file, delimiter=",", usecols=(6, 7))
    #print('radec_corr',radec_corr.shape,radec_corr)
    #print('radec_corrr',radec_corrr.shape,radec_corr)
    
    xy_extracted = \
    np.genfromtxt(fistar_file, usecols=(1, 2), names = ['x', 'y'])   
    radec_catalog = np.genfromtxt(
             catalog_file, skip_header=1,usecols=(1,2), names=['RA','Dec'])
    
    inputs = {"xy_corr":xy_corr, "radec_corr":radec_corr, \
              "xy_extracted":xy_extracted, "radec_catalog":radec_catalog, \
              "order":order }
    
    return inputs
    

def read_dr_file():
    """
    Read some variable (existing in the header of the fits file)
    from the dr file
    
    Args:
        z:
         
    Returns:
        ...
    """
    #dr_fname = input("Enter the name of the DR file: ")
    
    #with DataReductionFile(dr_fname, 'r') as dr_file:
    #    header = dr_file.get_frame_header()
    #    ra_cent = header.get("RA")*(np.pi/12)          # hr to radian
    #    dec_cent = header.get("dec")*(np.pi/180.0)     # degree to radian
    #    x_cent = header.get(ZNAXIS1)/2 # Length/2
    #    y_cent = header.get(ZNAXIS1)/2 # Width/2
    ra_cent = 10.2994169770979
    dec_cent = 45.6471
    x_cent = 3983/2.0
    y_cent = 2659/2.0
    
    dr_inputs = \
    {"ra_cent":ra_cent, "dec_cent":dec_cent, "x_cent":x_cent, "y_cent":y_cent}
    
    return dr_inputs


def projection(sources, projected, **center):  
    """
    Project the given sky position to a tangent plane (gnomonic projection).
    
    Args:
        sources(structured array-like): The the sky position to project
            (should have "RA" and "Dec" keys coordinates in degrees.
        
        projected:    A numpy array with "xi" and "eta" fields to fill
            with the projected coordinates (in degrees).
        
        center(dict):    Should define the central `'RA'` and `'Dec'` around
            which to project.
    Returns:
        None
    """
    degree_to_rad = np.pi / 180.0
    center['RA'] *= np.pi / 12.0 # bc it is in hr, not degree
    center['Dec'] *= degree_to_rad
    ra_diff = (sources['RA']  * degree_to_rad - center['RA'])
    cos_ra_diff = np.cos(ra_diff)
    cos_source_dec = np.cos(sources['Dec'] * degree_to_rad)
    cos_center_dec = np.cos(center['Dec'])
    sin_source_dec = np.sin(sources['Dec'] * degree_to_rad)
    sin_center_dec = np.sin(center['Dec'])
    denominator = (sin_center_dec * sin_source_dec + \
              cos_center_dec * cos_source_dec * cos_ra_diff) * degree_to_rad
    #projected = np.zeros(radec.shape, dtype=None)
    projected['xi'] = (cos_source_dec * np.sin(ra_diff)) / denominator
    projected['eta'] = (
                cos_center_dec * sin_source_dec
                -
                sin_center_dec * cos_source_dec * cos_ra_diff
                ) / denominator
    return None





def inv_projection (sources, projected, **center):
    """
    Inverse projection from tangent plane (xi, eta)
        to the sky position (RA, DEC)
    
    Args:
        sources: An empty numpy array with "RA" and "Dec" fields to fill
                 with the inverse projected of tangent plane coordinates.
                 (in degrees)
        
        projected: numpy array with "xi" and "eta" fields
        
        center(dict): Should define the central "RA" and "Dec"
                      in hour and degree respectively, which will be converted
                      to radian
        
    Returns:
        None
    """
    center['RA'] *= np.pi / 12.0
    center['Dec'] *= np.pi / 180.0
    
    rho = np.sqrt(projected['xi']**2 + projected['eta']**2)
    c = np.arctan(rho)
    denominator = rho*np.cos(center['Dec'])*np.cos(c) - \
                  projected['eta']*np.sin(center['Dec'])*np.sin(c)

    
    sources['RA'] = center['RA'] + \
                    np.arctan((projected['xi']*np.sin(c))/denominator)
    sources['Dec'] = np.arcsin(np.cos(c)*np.sin(center['Dec']) + \
          (projected['eta']*np.sin(c)*np.cos(center['Dec']))/rho)
    
    return None

def inv_projection_old (zeta_eta, ra_cent, dec_cent): 
    """
    Inverse projection from tangent plane (zeta, eta)
        to the sky position (RA, DEC)
    
    Args:
        zeta_eta:
        
        ra_cent: In radian
        
        dec_cent: In radian
        
    Returns:
        ...
    """
    #unprojected = np.zeros(zeta_eta.shape, dtype=None)
        
    rho = np.sqrt(zeta_eta[:,0]**2 + zeta_eta[:,1]**2)
    c = np.arctan(rho)
    denominator = rho*np.cos(dec_cent)*np.cos(c) - \
                  zeta_eta[:,1]*np.sin(dec_cent)*np.sin(c)

    
    ra = ra_cent + np.arctan((zeta_eta[:,0]*np.sin(c))/denominator)
    dec = np.arcsin(np.cos(c)*np.sin(dec_cent) + \
          (zeta_eta[:,1]*np.sin(c)*np.cos(dec_cent))/rho)
    
    unprojected[:,0] = ra
    unprojected[:,1] = dec
    
    return unprojected
    

def transformation_matrix(order, zeta, eta):
    """
    Inverse projection from tangent plane (zeta, eta)
        to the sky position (RA, DEC)
    
    Args:
        order:
        
        zeta:
        
        eta:
        
    Returns:
        ...
    """
    # ones = col vector with len = # of our extracted sources from catalog
    m = np.ones((eta.shape[0],1))
    for i in range(1, order+1):
        for j in range(i+1):
            m = np.block([m,zeta**(i-j)*eta**j])
    return m
    
    
def func(z, trans_x, trans_y, x_cent, y_cent, order):
    """
    Constructing the two non linear function to be solved for
    new center (zeta, eta)
    
    Args:
        z:
         
    Returns:
        ...

    """
    
    zetac = z[0]
    etac = z[1]
   
    f = np.empty((2))

    f[0] = trans_x[0,0] - x_cent
    f[1] = trans_y[0,0] - y_cent
    
    k = 1
    for i in range(1, order + 1):
        for j in range(i+1):
            f[0] = f[0] + trans_x[k, 0]*zetac**(i-j)*etac**(j)
            f[1] = f[1] + trans_y[k, 0]*zetac**(i-j)*etac**(j)
            k = k + 1
    return f


def astrometry():
    """
    Main function: After we get the initial transformation,
    the rest is just iterating the steps to get new transformations 
    
    Args:
         
    Returns:
        ...
    """
    inputs = read_inputs()
    xy_corr = inputs["xy_corr"]
    radec_corr = inputs["radec_corr"]
    order = inputs["order"]
    
    dr_inputs = read_dr_file()
    ra_cent = dr_inputs["ra_cent"]
    dec_cent = dr_inputs["dec_cent"]
    
    
    projected = \
    np.empty(radec_corr.shape[0], dtype=[('xi', float), ('eta', float)])
    
    radec_cent = {"RA":ra_cent, "Dec":dec_cent}
    #print("--------------------")
    #print("radec_corr",radec_corr.shape)
    
    projection(radec_corr, projected, **radec_cent)
    #print("projected",projected.shape)
    #print("--------------------")
    
    
    
    #print('projected before', projected.shape,projected)    
    zeta = projected['xi']
    zeta = zeta[np.newaxis].T  # convert shape from (n,) to (n, 1)
    
    eta = projected['eta']
    eta = eta[np.newaxis].T  # convert shape from (n,) to (n, 1)
     
    # Matrix (m) containing zeta, eta. Used to find Transformation(t)
    mt = transformation_matrix(order, zeta, eta)
    #print("mt.shape",mt.shape)
    #print("xy_corr['x']",xy_corr['x'].shape)
    trans_x, resid_x, rank_x, sigma_x = linalg.lstsq(mt, xy_corr['x'])
    trans_y, resid_x, rank_y, sigma_y = linalg.lstsq(mt, xy_corr['y'])
    # We do not use resid, rank, sigma! Should be removed? 
    trans_x = trans_x[np.newaxis].T
    trans_y = trans_y[np.newaxis].T
    
    print('\n Initial Transformation matrix for X components:')
    print(trans_x)
    print("trans_x",trans_x.shape)
    print('\n Initial Transformation matrix for Y components:')
    print(trans_y)
    
    iteration(order, trans_x, trans_y)


def iteration(order, trans_x, trans_y):
    """
    Iterate the process until we get a transformation that
    its difference from the previous one is less than a threshold
    
    Args:
        order:
         
    Returns:
        ...
    """
    inputs = read_inputs()
    xy_extracted = inputs["xy_extracted"]
    xy_extracted = structured_to_unstructured(xy_extracted)
    
    
    #print(xy_extracted.shape)
    #print(xy_extracted)
    #print(xy_extracted['x'].reshape)
    #xy_extracted = np.block([xy_extracted['x'],xy_extracted['y']])
    #print(xy_extracted.shape)
    #print(xy_extracted)    
    radec_catalog = inputs["radec_catalog"]
    order = inputs["order"]
    
    dr_inputs = read_dr_file()
    ra_cent = dr_inputs["ra_cent"]
    dec_cent = dr_inputs["dec_cent"]
    x_cent = dr_inputs["x_cent"]
    y_cent = dr_inputs["y_cent"]
    
    counter = 0
    
    while True:
        counter = counter + 1
        if counter > 1:
            trans_x = trans_x_new
            trans_y = trans_y_new
            ra_cent = cent_new['RA']
            dec_cent = cent_new['Dec']
        print("shape2", radec_catalog.shape)
        
        radec_cent = {"RA":ra_cent,"Dec":dec_cent}
        projected = np.empty(
        radec_catalog.shape[0], dtype=[('xi', float), ('eta', float)]
        )
        projection(radec_catalog, projected, **radec_cent)

        zeta = projected['xi']
        zeta = zeta.reshape(len(zeta),1)  #convert shape from (n,) to (n, 1)    
        eta = projected['eta']
        eta = eta.reshape(len(eta),1)   #convert shape from (n,) to (n, 1)
    
        mxy = transformation_matrix(order, zeta, eta)
        
        x_transformed = mxy @ trans_x 
        y_transformed = mxy @ trans_y 
        #print("x_transformed",x_transformed)
        xy_transformed = np.block([x_transformed, y_transformed])

        kdtree = spatial.KDTree(xy_extracted)
        
        print("xy_extracted", xy_extracted.shape)
        print("xy_transformed", xy_transformed.shape)
        d, ix = kdtree.query(xy_transformed, distance_upper_bound = 1.5)
        n = 0

        # Find the number of matched sources. (If it is needed.)
        for i in d:
            if not np.isinf(i):
                n = n + 1
        print("# of matched:",n)
        matched_sources = \
        np.empty(
        n, dtype=[('RA', float), ('Dec', float), ('x', float), ('y', float)]
        )

        # Put all matched sources together
        j = 0
        k = -1
        for i in range(ix.size):
            k = k + 1
            if not np.isinf(d[i]):
                matched_sources['RA'][j] = radec_catalog['RA'][k]
                matched_sources['Dec'][j] = radec_catalog['Dec'][k]
                matched_sources['x'][j] = xy_extracted[ix[i], 0]
                matched_sources['y'][j] = xy_extracted[ix[i], 1]
                j = j + 1

        zGuess = np.array([np.mean(zeta), np.mean(eta)])
        z = fsolve(
        func, zGuess, args = (trans_x, trans_y, x_cent, y_cent, order)
        )
        #zetaeta_cent = np.reshape(z,(1,2))    # Reshape from (2,) to (1,2)
        
        # since when we derived T, zeta and eta were in degrees.
        zetaeta_cent = np.empty(1, dtype=[('xi', float), ('eta', float)])
        zetaeta_cent['xi'] = z[0]*np.pi/180
        zetaeta_cent['eta'] = z[1]*np.pi/180

        # With those (zeta, eta) we use the inverse projection to
        # find new (RAC, DECC)
        #radec_cent = {"RA":ra_cent, "Dec":dec_cent} < we already have this
        source = np.empty(1, dtype=[('RA', float), ('Dec', float)])
        inv_projection(source, zetaeta_cent, **radec_cent)
        
        # Replace ra_cent, dec_cent with new values in unprojected
        #ra_cent_new = radec_cent['RA']
        #dec_cent_new = radec_cent['Dec']
        #print('source',source.shape,source)
        #print(source['RA'])
        cent_new = {'RA':source['RA'][0]*(12.0/np.pi), 'Dec':source['Dec'][0]*(180.0/np.pi)}
        # This is to convert them to degree, as in projection func we convert them back!
        projected_new = np.empty(matched_sources.shape[0], dtype=[('xi', float), ('eta', float)])
        #print('matched_sources',matched_sources.shape)
        #print('cent_new',cent_new)
        #print("--------------------")    
        
        structured_to_unstructured
        #print("matched_sources",matched_sources.shape)
        
        projection(matched_sources, projected_new, **cent_new)
        
        #print("projected_new",projected_new.shape)
        #print("--------------------")    
        matched_sources = structured_to_unstructured(matched_sources)
        projected_new = structured_to_unstructured(projected_new)
        #print('matched_sources',matched_sources)
        #print('projected_new',projected_new)
        x_matched_sources = matched_sources[:,2:3]
        y_matched_sources = matched_sources[:,3:4]
        zeta = projected_new[:,0:1]
        eta = projected_new[:,1:2]
        #print('zeta',zeta)
        mt = transformation_matrix(order, zeta, eta)
        #print('mt', mt)
        trans_x_new, resid, rank, sigma = linalg.lstsq(mt, x_matched_sources)
        trans_y_new, resid, rank, sigma = linalg.lstsq(mt, y_matched_sources)

        print('\n new Transformation matrix for X components:')
        print(trans_x_new)
        print('\n new Transformation matrix for Y components:')
        print(trans_y_new)

        # What is the better way to finish the iteration?
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


if __name__ == '__main__':
    astrometry()














