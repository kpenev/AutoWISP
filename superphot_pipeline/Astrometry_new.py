# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 12:23:59 2021
@author: Ashkan

Goal: Calculating the transformation matrix as:
(Ra,Dec) > projection(gnemonic) > (zeta,eta) > transformation (T) > (x, y)
(Ra, Dec): on the sky which will be transformed to zeta(or xi) and eta
Trans_mat: Transformation Matrix which we are looking for
(x, y): coordinates on the frame

This will be done in 4 major steps:
    1. Get initial Transformation (T) using astrometry.net
    2. Building a KD tree
    3. Finding new T with much more matched sources
    4. Repeat the steps to get a better trnsformation
"""
import time
import glob
import numpy as np
import logging
import math
import sys
import argparse
from astropy.io import fits as pyfits
from scipy import linalg
from scipy import spatial
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
from numpy.lib.recfunctions import structured_to_unstructured
from superphot_pipeline.processing_steps.manual_util import read_catalogue
from superphot_pipeline import source_finder_util
from superphot_pipeline.astrometry.map_projections import gnomonic_projection

################## see line 116 ###############################
#from superphot_pipeline.astrometry.map_projections import \
#gnomonic_projection_inv

logging.basicConfig(filename='ast_result.log',level = logging.DEBUG)

def parse_command_line():
    """
    Return the parsed command line arguments.
    """

    parser = argparse.ArgumentParser(
        description='Inputs (name of the files, order')

    parser.add_argument(
        'order', type=int, help = 'order of astrometry')

    parser.add_argument(
        'matching_max_distance',type=float,help='kd3 upper bound distance')    

    parser.add_argument(
        'trans_threshold',type=float,help ='threshold for the difference of\
                                            two consecutive transformation')
    parser.add_argument(
        'ra_cent', type=float, help = 'RA of the center of the frame')

    parser.add_argument(
        'dec_cent', type=float, help = 'Dec of the center of the frame')

    parser.add_argument(
        'x_frame', type=float, help = 'Length (x) of the frame')

    parser.add_argument(
        'y_frame', type=float, help = 'Width (y) of the frame')

    parser.add_argument(
        'corr_file',type=str,help='Initial correlation file name from \
                                                     Astrometry.net')
    
    parser.add_argument(
        'fistar_file', type=str, help = 'fistar file name')
    
    parser.add_argument(
        'catalog_file', type=str, help = 'catalog file name')

    return parser.parse_args()

#############################################################################
### 1st edit map_projection (2nd line), then replace this func with: ########
### PhotometryPipeline/superphot_pipeline/astrometry/map_projections.py #####
#############################################################################
def projection(sources, projected, **center):
    """
    Project the given sky position to a tangent plane (gnomonic projection).
    
    Args:
        sources(structured array-like): The the sky position to project
            (should have "RA" and "Dec" keys coordinates in degrees.
        
        projected: A numpy array with "xi" and "eta" fields to fill
            with the projected coordinates (in degrees).
        
        center(dict): Should define the central `'RA'` and `'Dec'` around
            which to project. (It is supposed that RA is in hr and Dec is
            in degree)
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
    
    projected['xi'] = (cos_source_dec * np.sin(ra_diff)) / denominator
    projected['eta'] = (
                cos_center_dec * sin_source_dec
                -
                sin_center_dec * cos_source_dec * cos_ra_diff
                ) / denominator
    return None

##########################################################
##### Add this beside map_projection in: #################
#### PhotometryPipeline/superphot_pipeline/astrometry/ ###
##########################################################
def inv_projection (sources, projected, **center):
    """
    Inverse projection from tangent plane (xi, eta)
        to the sky position (RA, DEC)
    
    Args:
        sources: An empty numpy array with "RA" and "Dec" fields to fill
                 with the inverse projected of tangent plane coordinates.
                 (in degrees)
        
        projected: numpy array with "xi" and "eta" fields (in degrees)
        
        center(dict): Should define the central "RA" and "Dec"
                      in hour and degree respectively, which will be
                      converted to radian
        
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
    
    sources['RA'] *= 12.0/np.pi    # convert from radian to degree
    sources['Dec'] *= 180.0/np.pi  # convert from radian to degree
    
    return None


def transformation_matrix(order, zeta, eta):
    """
    Setting up the transformation matrix which includes the related orders
    of (xi, eta)
    ex for order 2: 1, xi, eta, xi^2, xi*eta, eta^2
    
    Args:
        order: order of astrometry
        
        zeta: from projected coordinates
        
        eta: from projected coordinates
        
    Returns:
        m: transformation matrix
    """
    # ones = col vector with len = # of our extracted sources from catalog
    m = np.ones((eta.shape[0],1))
    for i in range(1, order+1):
        for j in range(i+1):
            m = np.block([m,zeta**(i-j)*eta**j])
    return m
    
    
def new_center_func(z, trans_x, trans_y, x_cent, y_cent, order):
    """
    Constructing the two non linear function to be solved for
    new center (zeta, eta)
    
    Args:
        z:
        
        trans_x: transformation matrix for x
         
        trans_y: transformation matrix for y
        
        x_cent: x of the center of the frame
        
        y_cent: y of the center of the frame
         
    Returns:
        f: the related function

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
    Main function: To get the initial transformation,
    the rest is calling iteration function to iterate the steps to get
    new transformations 
    
    Args:
        None
         
    Returns:
        See the returns of func. iteration
                               
    """
    
    cmdline_config = parse_command_line()
    
    order = cmdline_config.order
    matching_max_distance = cmdline_config.matching_max_distance
    trans_threshold = cmdline_config.trans_threshold
    ra_cent = cmdline_config.ra_cent
    dec_cent = cmdline_config.dec_cent
    x_frame = cmdline_config.x_frame
    y_frame = cmdline_config.y_frame
    corr_file = cmdline_config.corr_file 
    fistar_file = cmdline_config.fistar_file
    catalog_file = cmdline_config.catalog_file

    
    
    xy_corr = \
    np.genfromtxt(corr_file, delimiter=",", usecols=(0,1), names=['x','y'])
    
    radec_corr = \
    np.genfromtxt(corr_file, delimiter=",", usecols=(6,7), names=['RA','Dec'])
    

    xy_extracted = np.genfromtxt(
         fistar_file,names=source_finder_util.get_srcextract_columns('fistar'),
         dtype=None, deletechars=''
    )

#############################################################################
##### use:
#from superphot_pipeline.processing_steps.manual_util import read_catalogue #
#############################################################################

    radec_catalog = np.genfromtxt(
             catalog_file, skip_header=1,usecols=(1,2), names=['RA','Dec'])
    
    projected = \
    np.empty(radec_corr.shape[0], dtype=[('xi', float), ('eta', float)])
    
    radec_cent = {"RA":ra_cent, "Dec":dec_cent}
    
    projection(radec_corr, projected, **radec_cent)

    zeta = projected['xi']
    zeta = zeta[np.newaxis].T  # convert shape from (n,) to (n, 1)
    
    eta = projected['eta']
    eta = eta[np.newaxis].T  # convert shape from (n,) to (n, 1)
     
    # Matrix (m) containing zeta, eta. Used to find Transformation(t)
    mt = transformation_matrix(order, zeta, eta)


    trans_x, resid_x, rank_x, sigma_x = linalg.lstsq(mt, xy_corr['x'])
    trans_y, resid_x, rank_y, sigma_y = linalg.lstsq(mt, xy_corr['y'])
    #### We do not use resid, rank, sigma! Should be removed? ### 
    trans_x = trans_x[np.newaxis].T
    trans_y = trans_y[np.newaxis].T
    
    logging.debug('\n Initial Transformation matrix for X components:')
    logging.debug(trans_x)
    logging.debug('\n Initial Transformation matrix for Y components:')
    logging.debug(trans_y)
    

    parameters = {'order':order,'matching_max_distance':matching_max_distance,
                  'trans_threshold':trans_threshold,'trans_x':trans_x,
                  'trans_y':trans_y, 'ra_cent':ra_cent, 'dec_cent':dec_cent,
                  'x_frame':x_frame, 'y_frame':y_frame, 'xy_corr':xy_corr,
                  'radec_corr':radec_corr, 'xy_extracted':xy_extracted,
                  'radec_catalog':radec_catalog}
    
    return(iteration(parameters))


def iteration(params):
    """
    Iterate the process until we get a transformation that
    its difference from the previous one is less than a threshold
    
    Args:
        params: A dictionary containing:

            order: order of astrometry
            matching_max_distance: upper bound distance in kd3
            trans_threshold: threshold for the difference of two consecutive
                             transformation
            trans_x: transformation matrix for x
            trans_y: transformation matrix for y
            ra_cent: RA of the center of the frame
            dec_cent: Dec of the center of the frame
            x_frame: length of the frame in pixel
            y_frame: width of the frame in pixel
            xy_corr: x and y values of the initial correlation
            radec_corr: RA and Dec values of the initial correlation
            xy_extracted: x and y of the extracted sources of the frame
            radec_catalog: RA and Dec of the catalog sources
            
    Returns:
        n_matched: # of matched sources
   
        n_extracted: # of matched sources
     
        n_catalog: # of matched sources
    
        res_sum: sum of residuals
    """
    order = params['order']
    matching_max_distance = params['matching_max_distance']
    trans_threshold = params['trans_threshold']
    trans_x = params['trans_x']
    trans_y = params['trans_y']
    ra_cent = params['ra_cent']
    dec_cent = params['dec_cent']
    x_frame = params['x_frame']
    y_frame = params['y_frame']
    xy_corr = params['xy_corr']
    radec_corr=params['radec_corr']
    xy_extracted = params['xy_extracted']
    radec_catalog = params['radec_catalog']

    x_cent = x_frame/2.0
    y_cent = y_frame/2.0    

    xy_extracted = structured_to_unstructured(xy_extracted)
    xy_extracted = xy_extracted[:,1:3] 

    n_extracted = len(xy_extracted)
    n_catalog = len(radec_catalog)

    counter = 0
    while True:
        
        counter = counter + 1
        if counter > 1:
            trans_x = trans_x_new
            trans_y = trans_y_new
            ra_cent = cent_new['RA']
            dec_cent = cent_new['Dec']

        
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
        # we should use: np.flatnonzero(x>2)

        xy_transformed = np.block([x_transformed, y_transformed])

        kdtree = spatial.KDTree(xy_extracted)
        d, ix = kdtree.query(
            xy_transformed, distance_upper_bound = matching_max_distance
        )
        #####################################################################
        ### Should we let the user choose upper bound distance? #############
        #####################################################################
        result, count = np.unique(ix, return_counts = True)
        
        # Here we find those catalog sources that are matched with a common
        # frame source. We keep only the nearest source:
        for multi_match_i in result[count > 1][:-1]:
            bad_match = (ix == multi_match_i)
            # This is the desired index that should be kept:
            found_index = np.argwhere(bad_match)[np.argmin(d[bad_match])][0]
            bad_match[found_index] = False
            good_match = np.invert(bad_match)
            d[bad_match] = np.inf
            ix[bad_match] = result[-1]
        ######################################################################
        ### IMP: We should try to see if there is a better way to do this ####
        ### removing of multiple matched sources. Or not, as this part of ####
        ### the astromy process is the least time consuming part #############
        ######################################################################


        n_matched = 0
        res_sum = 0

        for i in d:
            if not np.isinf(i):
                n_matched = n_matched + 1
                res_sum = res_sum + i**2

        logging.debug("# of matched: {}".format(n_matched))
        matched_sources = \
        np.empty(
        n_matched,dtype=[('RA',float), ('Dec',float), ('x',float), ('y',float)]
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
        new_center_func, zGuess, args=(trans_x, trans_y, x_cent, y_cent, order)
        )
        
        # since when we derived T, zeta and eta were in degrees.
        zetaeta_cent = np.empty(1, dtype=[('xi', float), ('eta', float)])
        zetaeta_cent['xi'] = z[0]*np.pi/180
        zetaeta_cent['eta'] = z[1]*np.pi/180

        # With those (zeta, eta) we use the inverse projection to
        # find new (RAC, DECC)
        # radec_cent = {"RA":ra_cent, "Dec":dec_cent} < we already have this
        source = np.empty(1, dtype=[('RA', float), ('Dec', float)])
        inv_projection(source, zetaeta_cent, **radec_cent)
        # Replace ra_cent, dec_cent with new values in unprojected
        cent_new = {
        'RA':source['RA'][0], 'Dec':source['Dec'][0]
        }

        projected_new = np.empty(
        matched_sources.shape[0], dtype=[('xi', float), ('eta', float)]
        )

        projection(matched_sources, projected_new, **cent_new)
        
        matched_sources = structured_to_unstructured(matched_sources)
        projected_new = structured_to_unstructured(projected_new)
        x_matched_sources = matched_sources[:,2:3]
        y_matched_sources = matched_sources[:,3:4]
        zeta = projected_new[:,0:1]
        eta = projected_new[:,1:2]

        mt = transformation_matrix(order, zeta, eta)

        trans_x_new, resid, rank, sigma = linalg.lstsq(mt, x_matched_sources)
        trans_y_new, resid, rank, sigma = linalg.lstsq(mt, y_matched_sources)

        logging.debug('\n new Transformation matrix for X components:')
        logging.debug(trans_x_new)
        logging.debug('\n new Transformation matrix for Y components:')
        logging.debug(trans_y_new)

        ##########################################################
        ### 1. What is the better way to finish the iteration? ###
        ### 2. Should we let the user to choose this? ############
        ##########################################################

        diff = np.block([trans_x_new - trans_x, trans_y_new - trans_y])

        if np.count_nonzero(diff > trans_threshold) == 0:
            # Exclude the sources that are not within the frame:
            in_frame = np.logical_and(
                   np.logical_and(xy_transformed[:,0]>-3, \
                                  xy_transformed[:,0]< (x_frame+3)), \
                   np.logical_and(xy_transformed[:,1]>-3, \
                                  xy_transformed[:,1]<(y_frame+3)))
            
            in_frame = in_frame[np.newaxis].T
            in_frame = np.append(in_frame, in_frame, axis = 1)            
            xy_transformed = xy_transformed[in_frame].reshape(-1,2)
            radec_catalog = structured_to_unstructured(radec_catalog)
            radec_catalog = radec_catalog[in_frame].reshape(-1,2)

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

            print('finished!, # of iteration =', counter)
            # outputs={} put all here
            return n_matched, n_extracted, n_catalog, res_sum


if __name__ == '__main__':

    n_matched, n_extracted, n_catalog, res_sum = astrometry()
    
    logging.debug("# of catalog sources: {}".format(n_catalog))
    logging.debug("# of extracted: {}".format(n_extracted))
    logging.debug("# of matched sources: {}".format(n_matched))
    logging.debug("fraction of matched/extracted: {}".format(
        round(n_matched/n_extracted,3))
    )
    logging.debug("Residual: {}".format(round(math.sqrt(res_sum/n_matched),4)))

