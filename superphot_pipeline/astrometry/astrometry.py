#!/usr/bin/env python3

"""Fit for a transformation between sky and image coordinates."""
import logging

import numpy
from numpy.lib.recfunctions import structured_to_unstructured

from scipy import linalg
from scipy import spatial
from scipy.optimize import fsolve

from superphot_pipeline.astrometry.map_projections import\
    gnomonic_projection,\
    inv_projection

#pylint:disable=R0913
#pylint:disable=R0914
#pylint:disable=R0915
#pylint:disable=C0103

def transformation_matrix(astrometry_order, xi, eta):
    """
    Setting up the transformation matrix which includes the
    related astrometry_orders of (xi, eta)

    Args:
        astrometry_order(int): The order of the transformation to fit

        xi(numpy array): from projected coordinates

        eta(numpy array): from projected coordinates

    Returns:
        trans_matrix(numpy array): transformation matrix

    Notes:
        Ex: for astrometry_order 2: 1, xi, eta, xi^2, xi*eta, eta^2
    """

    #TODO: alocate matrix first with right size and then fill, instead of doing
    #it the slow way below

    trans_matrix = numpy.ones((eta.shape[0], 1))

    for i in range(1, astrometry_order + 1):
        for j in range(i + 1):
            trans_matrix = numpy.block([trans_matrix, xi ** (i - j) * eta ** j])

    return trans_matrix

def new_xieta_cent_function(xieta_cent,
                            trans_x,
                            trans_y,
                            x_cent,
                            y_cent,
                            astrometry_order):
    """
    Constructing the two non-linear function to be solved for
    new center (xi, eta)

    Args:
        xieta_cent(numpy array): the center of xi and eta

        trans_x(numpy array): transformation matrix for x

        trans_y(numpy array): transformation matrix for y

        x_cent(float): x of the center of the frame

        y_cent(float): y of the center of the frame

        astrometry_order(int): The order of the transformation to fit

    Returns:
        new_xieta_cent(numpy array): the new center function for (xi, eta)

    """
    xi = xieta_cent[0]
    eta = xieta_cent[1]

    new_xieta_cent = numpy.empty(2)

    new_xieta_cent[0] = trans_x[0, 0] - x_cent
    new_xieta_cent[1] = trans_y[0, 0] - y_cent

    k = 1
    for i in range(1, astrometry_order + 1):
        for j in range(i + 1):
            new_xieta_cent[0] = new_xieta_cent[0] + \
                                trans_x[k, 0] * xi ** (i - j) * eta ** j
            new_xieta_cent[1] = new_xieta_cent[1] + \
                                trans_y[k, 0] * xi ** (i - j) * eta ** j
            k = k + 1
    return new_xieta_cent

def estimate_transformation(initial_corr,
                            ra_cent,
                            dec_cent,
                            tweak_order,
                            astrometry_order):
    """
    Estimate the transformation from astrometry.net correspondence file.

    Args:
        initial_corr(structured numpy array):    The correspondence file
            containing field_x, field_y, index_ra, and index_dec

        ra_cent(float):    Estimate of the RA of the center of the frame

        dec_cent(float):    Estimate of the Dec of the center of the frame

        tweak_order(int):    The order of the astrometry.net transformation

        astrometry_order(int):    The order of the transformation that will be
            fit. Fills terms of higher order than `tweak_order` with zeros.

    Returns:
        matrix:
            Estimate of the transformation x(xi, eta)

        matrix:
            Estimate of the transformation y(xi, eta)



    """

    projected = numpy.empty(initial_corr.shape[0],
                            dtype=[('xi', float), ('eta', float)])

    radec_center = dict(RA=ra_cent, Dec=dec_cent)

    gnomonic_projection(initial_corr,
                        projected,
                        **radec_center
                        )

    xi = projected['xi'][numpy.newaxis].T

    eta = projected['eta'][numpy.newaxis].T

    trans_matrix = transformation_matrix(tweak_order,
                                         xi,
                                         eta)
    num_trans_terms = ((astrometry_order + 1) * (astrometry_order + 2)) // 2
    num_tweak_terms = ((tweak_order + 1) * (tweak_order + 2)) // 2

    trans_x = numpy.zeros(num_trans_terms)
    trans_y = numpy.zeros(num_trans_terms)

    trans_x[:num_tweak_terms] = linalg.lstsq(trans_matrix, initial_corr['x'])[0]
    trans_y[:num_tweak_terms] = linalg.lstsq(trans_matrix, initial_corr['y'])[0]

    return trans_x[numpy.newaxis].T, trans_y[numpy.newaxis].T


def refine_transformation(*,
                          astrometry_order,
                          max_srcmatch_distance,
                          max_iterations,
                          trans_threshold,
                          trans_x,
                          trans_y,
                          ra_cent,
                          dec_cent,
                          x_frame,
                          y_frame,
                          xy_extracted,
                          catalogue):
    """
    Iterate the process until we get a transformation that
    its difference from the previous one is less than a threshold

    Args:
        astrometry_order(int): The order of the transformation to fit

        max_srcmatch_distance(float): The upper bound distance in the
            KD tree

        trans_threshold(float): The threshold for the difference of two
            consecutive transformations

        trans_x(numpy array): Initial estimate for the x transformation matrix

        trans_y(numpy array): Initial estimate for the x transformation matrix

        ra_cent(float): Initial estimate for the RA of the center of the frame

        dec_cent(float): Initial estimate for the Dec of the center of the frame

        x_frame(float): length of the frame in pixels

        y_frame(float): width of the frame in pixels

        xy_extracted(structured numpy array): x and y of the extracted
            sources of the frame

        catalogue: The catalogue of sources to match to

    Returns:
        trans_x(2D numpy array):
            the coefficients of the x(xi, eta) transformation

        trans_y(2D numpy array):
            the coefficients of the y(xi, eta) transformation

        cat_extracted_corr(structured numpy array):
            the catalogues extracted correspondence indexes

        res_rms(float): the residual

        ratio(float): the ratio of matched to unmatched

        ra_cent(float): the new RA center array

        dec_cent(float): the new Dec center array
    """

    x_cent = x_frame / 2.0
    y_cent = y_frame / 2.0
    xy_extracted = structured_to_unstructured(xy_extracted)
    xy_extracted = xy_extracted[:, 0:2]
    counter = 0
    x_transformed = numpy.inf
    y_transformed = numpy.inf

    kdtree = spatial.KDTree(xy_extracted)

    while True:

        counter = counter + 1
        if counter > 1:
            #TODO: fix pylint disables here
            #pylint:disable=used-before-assignment
            ra_cent = cent_new['RA']
            dec_cent = cent_new['Dec']
            # pylint:enable=used-before-assignment
        radec_cent = {"RA": ra_cent,
                      "Dec": dec_cent}

        projected = numpy.empty(
            catalogue.shape[0],
            dtype=[('xi', float), ('eta', float)]
        )
        gnomonic_projection(catalogue, projected, **radec_cent)

        xi = projected['xi']
        xi = xi.reshape(len(xi), 1)  # convert shape from (n,) to (n, 1)
        eta = projected['eta']
        eta = eta.reshape(len(eta), 1)  # convert shape from (n,) to (n, 1)

        trans_matrix_xy = transformation_matrix(astrometry_order,
                                                xi,
                                                eta)

        old_x_transformed = x_transformed
        old_y_transformed = y_transformed
        x_transformed = trans_matrix_xy @ trans_x
        y_transformed = trans_matrix_xy @ trans_y

        in_frame = numpy.logical_and(
            numpy.logical_and(x_transformed > -3,
                              x_transformed < (x_frame + 3)),
            numpy.logical_and(y_transformed > -3,
                              y_transformed < (y_frame + 3))
        ).flatten()

        diff = numpy.sqrt(
            (old_x_transformed - x_transformed)**2 +
            (old_y_transformed - y_transformed)**2
        ).flatten()[in_frame]

        print('diff:'+repr(diff.max()))
        if not (diff > trans_threshold).any() or counter > max_iterations:
            # pylint:disable=used-before-assignment
            cat_extracted_corr = numpy.empty((n_matched, 2),
                                             dtype=int)
            cat_extracted_corr[:, 0] = numpy.arange(catalogue.shape[0])[matched]
            cat_extracted_corr[:, 1] = ix[matched]
            # Exclude the sources that are not within the frame:

            return trans_x, \
                trans_y,\
                cat_extracted_corr, \
                res_rms, \
                ratio, \
                ra_cent, \
                dec_cent
            # pylint:enable=used-before-assignment
        xy_transformed = numpy.block([x_transformed, y_transformed])
        d, ix = kdtree.query(
            xy_transformed,
            distance_upper_bound=max_srcmatch_distance
        )

        result, count = numpy.unique(ix, return_counts=True)

        for multi_match_i in result[count > 1][:-1]:
            bad_match = (ix == multi_match_i)
            d[bad_match] = numpy.inf
            ix[bad_match] = result[-1]

        matched = numpy.isfinite(d)
        n_matched = matched.sum()
        n_extracted = len(xy_extracted)
        #TODO: add weights to residual and to the fit eventually
        res_rms = numpy.sqrt(numpy.square(d[matched]).mean())
        ratio = round(n_matched / n_extracted, 3)

        logging.debug("# of matched: %s", n_matched)
        matched_sources = \
            numpy.empty(
                n_matched,
                dtype=[('RA', float),
                       ('Dec', float),
                       ('x', float),
                       ('y', float)]
            )

        j = 0
        k = -1

        for i in range(ix.size):
            k += 1
            if not numpy.isinf(d[i]):
                matched_sources['RA'][j] = catalogue['RA'][k]
                matched_sources['Dec'][j] = catalogue['Dec'][k]
                matched_sources['x'][j] = xy_extracted[ix[i], 0]
                matched_sources['y'][j] = xy_extracted[ix[i], 1]
                j = j + 1

        xieta_guess = numpy.array(
            [numpy.mean(xi),
             numpy.mean(eta)]
        )
        new_xieta_cent = fsolve(
            new_xieta_cent_function,
            xieta_guess,
            args=(trans_x,
                  trans_y,
                  x_cent,
                  y_cent,
                  astrometry_order)
        )

        xieta_cent = numpy.empty(1, dtype=[('xi', float), ('eta', float)])
        xieta_cent['xi'] = new_xieta_cent[0] * numpy.pi / 180
        xieta_cent['eta'] = new_xieta_cent[1] * numpy.pi / 180

        source = numpy.empty(
            1,
            dtype=[('RA', float), ('Dec', float)]
        )
        inv_projection(source, xieta_cent, **radec_cent)

        cent_new = {
            'RA': source['RA'][0], 'Dec': source['Dec'][0]
        }

        projected_new = numpy.empty(
            matched_sources.shape[0], dtype=[('xi', float), ('eta', float)]
        )

        gnomonic_projection(matched_sources, projected_new, **cent_new)

        trans_matrix = transformation_matrix(
            astrometry_order,
            projected_new['xi'].reshape(projected_new['xi'].size, 1),
            projected_new['eta'].reshape(projected_new['eta'].size, 1)
        )
        print(trans_matrix.shape)

        if trans_matrix.shape[0] <= 5 * trans_matrix.shape[1]:
            raise ValueError('The number of equations is '
                             'insufficient to solve transformation '
                             'coefficients')

        trans_x = linalg.lstsq(
            trans_matrix,
            matched_sources['x'].reshape(matched_sources['x'].size, 1)
        )[0]
        trans_y = linalg.lstsq(
            trans_matrix,
            matched_sources['y'].reshape(matched_sources['x'].size, 1)
        )[0]

        # x_extracted = xy_extracted[:, 0]
        # y_extracted = xy_extracted[:, 1]
        #
        # x_projected = x_transformed
        # y_projected = y_transformed
        #
        # x_matched = matched_sources['x']
        # y_matched = matched_sources['y']
        # if counter == 22:
        #
        #     with open (os.path.join('/home/aer140130/python_work/SonyAlphaPhotometry/scripts/','regions_ds9_extracted(r)_projected(g)_counter_'+str(counter)+'.reg'),'w+') as reg_file:
        #         for x, y in zip(x_extracted, y_extracted):
        #             reg_file.write(
        #                 'box({xc!r},{yc!r},{w!r},{h!r},0) # color=red width=4 \n'.format(
        #                     xc=x + 0.5,
        #                     yc=y + 0.5,
        #                     w=5,
        #                     h=5
        #                 )
        #             )
        #         for x, y in zip(x_projected, y_projected):
        #             x = numpy.float64(x)
        #             y = numpy.float64(y)
        #             reg_file.write(
        #                 'box({xc!r},{yc!r},{w!r},{h!r},0) # color=green width=4 \n'.format(
        #                     xc=x + 0.5,
        #                     yc=y + 0.5,
        #                     w=8,
        #                     h=8
        #                 )
        #             )
        #         for x, y in zip(x_matched, y_matched):
        #             reg_file.write(
        #                 'box({xc!r},{yc!r},{w!r},{h!r},0) # color=blue width=4 \n'.format(
        #                     xc=x + 0.5,
        #                     yc=y + 0.5,
        #                     w=3,
        #                     h=3
        #                 )
        #             )
        # if counter == 23:
        #
        #     with open (os.path.join('/home/aer140130/python_work/SonyAlphaPhotometry/scripts/','regions_ds9_projected(b)_counter_'+str(counter)+'.reg'),'w+') as reg_file:
        #
        #         for x, y in zip(x_projected, y_projected):
        #             x = numpy.float64(x)
        #             y = numpy.float64(y)
        #             reg_file.write(
        #                 'box({xc!r},{yc!r},{w!r},{h!r},0) # color=blue width=1 \n'.format(
        #                     xc=x + 0.5,
        #                     yc=y + 0.5,
        #                     w=8,
        #                     h=8
        #                 )
        #             )
        # if counter > 23:
        #     break
