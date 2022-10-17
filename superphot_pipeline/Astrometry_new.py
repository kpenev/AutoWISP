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


def projection(RA_Dec, RA_cent, Dec_cent):
    
    degree_to_rad = np.pi / 180.0
    ra_diff = (RA_Dec[:,0]  * degree_to_rad - RA_cent)
    cos_ra_diff = np.cos(ra_diff)
    cos_source_dec = np.cos(RA_Dec[:,1] * degree_to_rad)
    cos_center_dec = np.cos(Dec_cent)
    sin_source_dec = np.sin(RA_Dec[:,1] * degree_to_rad)
    sin_center_dec = np.sin(Dec_cent)
    denominator = (sin_center_dec * sin_source_dec +
                   cos_center_dec * cos_source_dec * cos_ra_diff) * degree_to_rad

    projected = np.zeros(RA_Dec.shape, dtype=None)
    projected[:,0] = (cos_source_dec * np.sin(ra_diff)) / denominator
    projected[:,1] = (
                cos_center_dec * sin_source_dec
                -
                sin_center_dec * cos_source_dec * cos_ra_diff
                ) / denominator
    return projected


def solve_eq(z):
   zetac = z[0]
   etac = z[1]
   
   F = np.empty((2))
   
   F[0] = T_x[0,0] + T_x[1,0]*zetac + T_x[2,0]*etac + T_x[3,0]*zetac**2 +\
   T_x[4,0]*zetac*etac + T_x[5,0]*etac**2 - cent_x
   
   F[1] = T_y[0,0] + T_y[1,0]*zetac + T_y[2,0]*etac + T_y[3,0]*zetac**2 +\
   T_y[4,0]*zetac*etac + T_y[5,0]*etac**2 - cent_y
   
   return F


def inv_projection (zeta_eta, RA_cent, Dec_cent):
    """
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
    
    
    

def Trans_Mat(order, zeta, eta):
    M = np.ones((eta.shape[0],1))
    for i in range(1,order+1):
        for j in range(i+1):
            M = np.block([M,zeta**(i-j)*eta**j])
    return M

# =============================================================================
# corr.csv from astrometry.net: x field / y field / ra field / dec field/ ...
# =============================================================================
####################  How to do this part using astrometry.net package? ####################

order = 5

X_Y = np.genfromtxt('corr.csv', delimiter=",", usecols=(0, 1))
RA_Dec = np.genfromtxt('corr.csv',delimiter=",", usecols=(6, 7))
X = X_Y[:,0:1]
Y = X_Y[:,1:2]

a,b = X_Y.shape
c,d = RA_Dec.shape


# =============================================================================
#  Values from frame header
# =============================================================================

hdu = 1             # It is compressed, so hdu = 1
dir = "./"          # Should we give it the direction in config file?
for fitsName in glob.glob(dir+'*.fits.fz'):        # Not necessary here, but maybe good for later!  
    header = pyfits.getheader(fitsName, hdu)
    RA_cent = header.get("RA")*(np.pi/12)          #convert from hr to radian (see line 64)
    Dec_cent = header.get("Dec")*(np.pi/180.0)     # convert from degree to radian

# x, y of the center of frame: #should be read from header
cent_x = 3983/2.0
cent_y = 2659/2.0

# -------------------------------------------------------------------------------------

# Running the projection function
projected = projection(RA_Dec, RA_cent, Dec_cent)

# =============================================================================
# Now that we have projected array, we should find our transformation to go from
# projected (xi, eta) to frame (x, y) 
# Finally generating M or matrix of multiplication 
# =============================================================================

zeta = projected[:,0:1]
eta = projected[:,1:2]

####################  When forming the loop, at each step increase the order of the transformation, only M should be changed ####################
MT = Trans_Mat(order, zeta, eta)

# Now solving for transformation matrix:
# (Better to add weight of each star > its brightness in future)

# !!! we do not use resid, rank, sigma! delete if not necessary. !!! 
T_x, resid_x, rank_x, sigma_x = linalg.lstsq(MT,X)
T_y, resid_x, rank_y, sigma_y = linalg.lstsq(MT,Y)

print('\n Initial Transformation matrix for X components:')
print(T_x)
print('\n Initial Transformation matrix for Y components:')
print(T_y)

# =============================================================================
# Now we have the transformation from step 1, (We do not need the others)
# That means:
# FINISH STEP 1.
# =============================================================================
# STEP 2:
# =============================================================================
# 2.1) and 2.2)
# Building a KD tree from fistar file.
# We have it as:
# X_Y (line 25)
# Using ucac4read to get a full list of (RA, Dec, flux, ...) which we call it C
# that completely covers our frame area
# We have it as: complete_list_of_sources_10299_45647.ucac4
# =============================================================================


X_Y_f = np.genfromtxt('10-465000_2_G1_250.fistar', usecols=(1, 2))
RA_Dec_C = np.genfromtxt('cat_sources.ucac4', skip_header = 1, usecols=(1, 2))

# =============================================================================
# 2.3) Apply projection to C(RA_Dec_C)
# =============================================================================

iteration = 0

while True:
    
    iteration = iteration + 1
    if iteration > 1:
        T_x = T_x_new
        T_y = T_y_new
        RA_cent = RA_cent_new
        Dec_cent = Dec_cent_new
        
    projected = projection(RA_Dec_C, RA_cent, Dec_cent )

# =============================================================================
# FINISH STEP 2.
# =============================================================================
# STEP 3:
# =============================================================================
# =============================================================================
# 3.1)
# Apply T to projected or zeta, eta  
# =============================================================================

    zeta = projected[:,0:1]
    eta = projected[:,1:2]
   # ones = np.ones((projected.shape[0],1)) # col vector with len = # of our extracted sources from catalog
    
    MXY = Trans_Mat(order, zeta, eta) 
    # np.block([ones,zeta,eta,zeta**2,zeta*eta,eta**2])

    zGuess = np.array([np.mean(zeta), np.mean(eta)])
    
        
    X_all_p = MXY @ T_x 
    Y_all_p = MXY @ T_y 

    X_Y_C = np.block([X_all_p, Y_all_p])
    print("X_Y_C shape:", X_Y_C.shape)

# =============================================================================
# 3.2)
# Apply KDTree: tree = X_Y_f, query = X_Y_C
# Matched sources in X_Y_f >> X_Y_e
# =============================================================================

    kdtree = spatial.KDTree(X_Y_f)
    d, ix = kdtree.query(X_Y_C, distance_upper_bound = 1.5)
    n = 0


# Find the number of matched sources >>> I dont think we need this.
    for i in d:
        if not math.isinf(i):
            n = n + 1
    print("# of matched:",n)
    X_Y_e = np.zeros((n,4),dtype=None)

# Put all matched sources together
# j is to find index in fistar
# k is to find index in ra dec
    j = 0
    k = -1
# =============================================================================
    for i in ix:
        k = k + 1
        if i != 5957:
            X_Y_e[j, 0] = RA_Dec_C[k, 0]
            X_Y_e[j, 1] = RA_Dec_C[k, 1]
            X_Y_e[j, 2] = X_Y_f[i, 0]
            X_Y_e[j, 3] = X_Y_f[i, 1]
            j = j +1


# =============================================================================
# 3.3)
# Use numerical solver to find (RA, Dec) that when we apply gnomonic 
# projection on (RAC, DEC) and then T, goes to exactly center of frame
# we replace (RAC, DecC) this new (RAC, DecC) 
# =============================================================================



# First, using cent_x, cent_y and solving two nonlinear eq:
# M(zeta, eta)*T_x = x, M(zeta, eta)*T_y = y
# to get the corresponding (zeta, eta)



# We choose our initial guess = avg of (zeta, eta) that we have
############################# Isn't it better to use the projected of our header RA/Dec? #############################

    z = fsolve(solve_eq, zGuess)

    zeta_eta = np.reshape(z,(1,2))    # Reshape from (2,) to (1,2)


#This is in degree, since when we derived T, zeta and eta were in degrees.
    zeta_eta[0,0] = zeta_eta[0,0]*np.pi/180
    zeta_eta[0,1] = zeta_eta[0,1]*np.pi/180


# With those (zeta, eta) we use the inverse projection to find new (RAC, DECC)


    unprojected_cent = inv_projection(zeta_eta, RA_cent, Dec_cent)


#print("RAC, DECC old: ", RA_cent, Dec_cent)
#print("RAC, DECC new: ", unprojected)

# {Replace RA_cent, Dec_cent with new values in unprojected}
# We found new RA_cent and Dec_cent
    RA_cent_new = unprojected_cent[0,0]
    Dec_cent_new = unprojected_cent[0,1]

# =============================================================================
# 3.4)
# Apply gnemonic projection with updated RAC, DecC to C 
# =============================================================================

# RA, Dec exist in X_Y_e first and second column, we cnan feed it with RA_Dec_C too.

    projected_new = projection(X_Y_e, RA_cent_new, Dec_cent_new)

# =============================================================================
# 3.5) Find new transformation
# =============================================================================

    X = X_Y_e[:,2:3]
    Y = X_Y_e[:,3:4]
    zeta = projected_new[:,0:1]
    eta = projected_new[:,1:2]
    #ones = np.ones((n,1)) # col vector with len = # of our extracted sources
    MT = Trans_Mat(order, zeta,eta)
 #   np.block([ones,zeta,eta,zeta**2,zeta*eta,eta**2])

# Now solving for transformation matrix:
# (Better to add weight of each star > its brightness in future)
    T_x_new, resid, rank, sigma = linalg.lstsq(MT,X)
    T_y_new, resid, rank, sigma = linalg.lstsq(MT,Y)

    print('\n 2nd Transformation matrix for X components:')
    print(T_x_new)
    print('\n 2nd Transformation matrix for Y components:')
    print(T_y_new)


# ********************************************************


    threshold = 0.1
    
    diff = np.block([T_x_new - T_x, T_y_new - T_y])
    if np.count_nonzero(diff > threshold) == 0:
        break

            

# I add this part to export X_Y_C in a reg file to finally plot it on ds9


with open('reg.reg', 'w') as f_out:
        for x, y in X_Y_C:
            print(
                'box({xc!r},{yc!r},{w!r},{h!r},0) # color=green'.format(
                    xc=x,
                    yc=y,
                    w=8,
                    h=8
                ), file = f_out
            )
    
    

print('finish!, # of iteration =',iteration)












