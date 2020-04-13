"""Define a function doing iterative re-fitting of terms from an expression."""

import logging
import scipy
import scipy.linalg


#TODO: switch to QR and use downdating to remove outliers.
#TODO: accept QR decomposition directly to accomodate TFA
#Using QR, the algorithm for solving without weights is:
#```
#    q, r= scipy.linalg.qr(a, mode='economic')
#    q_rhs = scipy.dot(q.T, rhs)
#    solution = scipy.linalg.solve_triangular(r, q_rhs)
#```
#To accomodate weights one can scale q.
def iterative_fit(predictors,
                  target_values,
                  *,
                  weights=None,
                  error_avg,
                  rej_level,
                  max_rej_iter,
                  fit_identifier):
    """
    Fit for the coefficients of a single fitting group.

    Args:
        predictors:    The derivatives w.r.t. to the fit coefficients of
            the predicted values (i.e. the matrix defining the fitting
            problem, apart from weighting).

        target_values:    The values wey are trying to reproduce.

        weights:    The weight to give to each entry in `target_values`. If
            None, no weighting is done.

        phot_ind:    The index of the photometry being fit (only used for
            reporting errors).

        error_avg:    How to average fitting residuals for outlier rejection.

        rej_level:    How far away from the fit should a point be before it is
            rejected in utins of error_avg.

        max_rej_iter:    The maximum number of rejection/re-fitting iterations
            to perform. If the fit has not converged by then, the latest
            iteration is accepted.

        fit_identifier:    Identifier of what is being fit. Only used in logging
            messages.

    Returns:
        scipy.array:
            The best fit coefficients.

        float:
            The average square residual of the best fit.

        int:
            The number of non-rejected points used in the last fit iteration.
    """

    logger = logging.getLogger(__name__)

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

    if weights is not None:
        predictors = scipy.multiply(predictors, weights)
        target_values = scipy.multiply(target_values, weights)
    num_free_coef = len(predictors)

    bad_ind = scipy.logical_not(scipy.isfinite(target_values))
    if weights is not None:
        bad_ind = scipy.logical_or(bad_ind, weights <= 0)
    bad_ind = bad_ind.nonzero()[0]

    #Intended just to limit the number of iterations
    #pylint: disable=unused-variable
    for rej_iter in range(max_rej_iter + 1):
    #pylint: enable=unused-variable
        predictors = scipy.delete(predictors, bad_ind, 1)
        target_values = scipy.delete(target_values, bad_ind)
        if weights is not None:
            weights = scipy.delete(weights, bad_ind)
        logger.debug('Iteration %d, %d sources, %d coefficients\n',
                     rej_iter,
                     len(target_values),
                     num_free_coef)
        if len(target_values) < num_free_coef:
            return None, None, 0
        best_fit_coef, residues, rank, sv = scipy.linalg.lstsq(
            predictors.T,
            target_values,
            None,
            lapack_driver='gelsy'
        )
        logger.debug('Fit: coef = %s, residues = %s, rank = %s, sv = %s',
                     repr(best_fit_coef), repr(residues), rank, sv)
        bad_ind, fit_res2 = rejected_indices(
            scipy.dot(best_fit_coef, predictors) - target_values,
            weights
        )
        if bad_ind.size == 0:
            return best_fit_coef, fit_res2, len(target_values)

    return best_fit_coef, fit_res2, len(target_values)
