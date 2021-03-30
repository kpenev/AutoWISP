"""Define a function doing iterative re-fitting of terms from an expression."""

import logging
import scipy
import scipy.linalg

#pylint: disable=too-many-locals
def iterative_fit_qr(weighted_predictors,
                     weighted_qrp,
                     weighted_target,
                     *,
                     weights=None,
                     max_downdates=0.1,
                     error_avg,
                     rej_level,
                     max_rej_iter,
                     fit_identifier,
                     pre_reject=False):
    """
    Same as iterative_fit() but using the QR decomposition of predictors.

    Args:
        weighted_predictors(2-D array):    The matrix of predictors, with
            weights already applied (used for calculating residuals and if too
            many poinst get rejected, see `max_downdates`).

        weighted_qrp(2-D array):    The QR decomposition of
            `weighted_predictors` with pivoting. Should be exactly the output of
            `scipy.linalg.qr(weighted_predictors, pivoting=True)`, possibly with
            `mode='economic'`.

        weighted_target(1-D array):    The vector of values to reproduce as a
            linear combination of predictors. Should also already be weighted.

        max_downdates:    See iterative_fit().

        weights:    See iterative_fit(). Only used for outlier detection.

        error_avg:    See iterative_fit().

        rej_level:    See iterative_fit().

        max_rej_iter:    See iterative_fit().

        fit_identifier:    See iterative_fit().

    Returns:
        See iterative_fit_qr()
    """

    def rejected_indices(weighted_fit_diff, weights):
        """Return indices of outlier sources and squared fit residual."""

        logger.debug('Weigthed difference: %s', repr(weighted_fit_diff))
        logger.debug('Weigths: %s', repr(weights))

        assert weights is None or (weights > 0).all()
        assert scipy.isfinite(weighted_fit_diff).all()

        fit_diff2 = pow(
            weighted_fit_diff/(1.0 if weights is None else weights),
            2
        )
        logger.debug('Square difference: %s', repr(fit_diff2))
        if error_avg == 'weightedmean':
            res2 = scipy.mean(pow(weighted_fit_diff, 2))
            if weights is not None:
                res2 /= scipy.mean(pow(weights, 2))
        else:
            res2 = getattr(scipy, error_avg)(fit_diff2)
        max_diff2 = rej_level**2*res2
        logger.debug('max square difference: %s', repr(max_diff2))
        if res2 < 0:
            logger.debug(
                '%s',
                '\n'.join([
                    repr(fit_identifier),
                    '\tNegative square residual: ' + repr(res2),
                    (
                        '\tWeights (min, max): ('
                        +
                        repr(1.0 if weights is None else min(weights))
                        +
                        ', '
                        +
                        repr(1.0 if weights is None else max(weights))
                        +
                        ')'
                    ),
                    (
                        '\tweighted_fit_diff (min, max): ('
                        +
                        repr(min(weighted_fit_diff))
                        +
                        ', '
                        +
                        repr(max(weighted_fit_diff))
                        +
                        ')'
                    )
                ])
            )
        return (fit_diff2 > max_diff2).nonzero()[0], res2

    logger = logging.getLogger(__name__)
    num_free_coef = len(weighted_predictors)

    if 0.0 <= max_downdates < 1:
        max_downdates = int(scipy.around(max_downdates * weighted_target.size))

    bad_ind = scipy.logical_not(scipy.isfinite(weighted_target))
    if weights is not None:
        bad_ind = scipy.logical_or(bad_ind, weights <= 0)
    bad_ind = bad_ind.nonzero()[0]

    permutation = scipy.argsort(weighted_qrp[2])

    for rej_iter in range(-1 if pre_reject else 0, max_rej_iter + 1):
        weighted_target = scipy.delete(weighted_target, bad_ind)
        if len(weighted_target) < num_free_coef:
            return None, None, 0

        weighted_predictors = scipy.delete(weighted_predictors, bad_ind, 1)
        if weights is not None:
            weights = scipy.delete(weights, bad_ind)
        logger.debug('Iteration %d, %d sources, %d coefficients\n',
                     rej_iter,
                     len(weighted_target),
                     num_free_coef)
        if bad_ind.size > max_downdates:
            #False positive
            #pylint: disable=unexpected-keyword-arg
            weighted_qrp = scipy.linalg.qr(weighted_predictors.T,
                                           mode='economic',
                                           pivoting=True)
            #pylint: enable=unexpected-keyword-arg
        else:
            for i in scipy.flip(bad_ind):
                weighted_qrp = (
                    #False positive
                    #pylint: disable=no-member
                    *scipy.linalg.qr_delete(*weighted_qrp[:2], i),
                    #pylint: enable=no-member
                    weighted_qrp[2]
                )

        if rej_iter < 0:
            best_fit_coef = scipy.zeros(num_free_coef)
        else:
            try:
                #False positive
                #pylint: disable=no-member
                best_fit_coef = scipy.linalg.solve_triangular(
                    weighted_qrp[1],
                    scipy.dot(weighted_qrp[0].T, weighted_target)
                )[permutation]
                #pylint: enable=no-member
            except scipy.linalg.LinAlgError:
                return None, None, 0

        bad_ind, fit_res2 = rejected_indices(
            scipy.dot(best_fit_coef, weighted_predictors) - weighted_target,
            weights
        )
        logger.debug('Fit: coef = %s, square residual = %s, %d rejected',
                     repr(best_fit_coef), repr(fit_res2), bad_ind.size)

        if bad_ind.size == 0:
            return best_fit_coef, fit_res2, len(weighted_target)

    return best_fit_coef, fit_res2, len(weighted_target)
#pylint: enable=too-many-locals

def iterative_fit(predictors,
                  target_values,
                  *,
                  max_downdates=0.1,
                  weights=None,
                  error_avg,
                  rej_level,
                  max_rej_iter,
                  fit_identifier,
                  pre_reject=False):
    """
    Find least squares coefficients reproducing target_values using predictors.

    Args:
        predictors:    The derivatives w.r.t. to the fit coefficients of
            the predicted values (i.e. the matrix defining the fitting
            problem, apart from weighting).

        target_values:    The values wey are trying to reproduce.

        max_downdates(float or int):    The maximum number of deletions to
            handle by downdating the QR decomposition, either as a fraction of
            the size of `weighted_target` (if `max_downdates` is a floating
            point between 0 and 1), or as an absolute value (if `max_downdates`
            is a positive integer). If an operation results in a larger number
            of deletions, the original matrix is re-constructed, the
            corresponding rows are deleted and a new QR decomposition is derived
            from scratch.

        weights:    The weight to give to each entry in `target_values`. If
            None, no weighting is done.

        phot_ind:    The index of the photometry being fit (only used for
            reporting errors).

        error_avg(str):    How to average fitting residuals for outlier
            rejection. Should be a scipy/numpy top-level function (e.g. mean,
            nanmedian etc).

        rej_level(float):    How far away from the fit should a point be before
            it is rejected in units of error_avg.

        max_rej_iter:    The maximum number of rejection/re-fitting iterations
            to perform. If the fit has not converged by then, the latest
            iteration is accepted.

        fit_identifier:    Identifier of what is being fit. Only used in logging
            messages.

        pre_reject:    Should a rejection iteration be performed before even the
            first fit is attempted (i.e. discard outliers even from the first
            fit).

    Returns:
        scipy.array:
            The best fit coefficients.

        float:
            The average square residual of the best fit.

        int:
            The number of non-rejected points used in the last fit iteration.
    """

    if weights is not None:
        predictors = scipy.multiply(predictors, weights)
        target_values = scipy.multiply(target_values, weights)

    #False positive
    #pylint: disable=unexpected-keyword-arg
    qrp = scipy.linalg.qr(predictors.T, mode='economic', pivoting=True)
    #pylint: enable=unexpected-keyword-arg

    return iterative_fit_qr(predictors,
                            qrp,
                            target_values,
                            weights=weights,
                            max_downdates=max_downdates,
                            error_avg=error_avg,
                            rej_level=rej_level,
                            max_rej_iter=max_rej_iter,
                            fit_identifier=fit_identifier,
                            pre_reject=pre_reject)
