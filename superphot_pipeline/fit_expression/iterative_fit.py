"""Define a function doing iterative re-fitting of terms from an expression."""

import logging

import scipy
import scipy.optimize

#TODO: switch to linear least squares
#Could not come up with a reasonable way to simplify
#pylint: disable=too-many-locals
def iterative_fit(derivatives,
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
        derivatives:    The derivatives w.r.t. to the fit coefficients of
            the predicted values (i.e. the matrix defining the fitting
            problem).

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
        coefficients:     The best fit coefficients.

        fit_res2:    The square residuals of the best fit.
    """

    logger = logging.getLogger(__name__)

    def rejected_indices(weighted_fit_diff, weights):
        """Return indices of outlier sources and squared fit residual."""

        logger.debug('Weigthed difference: %s', repr(weighted_fit_diff))
        logger.debug('Weigths: %s', repr(weights))
        fit_diff2 = pow(weighted_fit_diff/(weights or 1.0), 2)
        logger.debug('Square difference: %s', repr(fit_diff2))
        if error_avg == 'weightedmean':
            res2 = (scipy.mean(pow(weighted_fit_diff, 2))
                    /
                    (1.0 if weights is None else scipy.mean(pow(weights, 2))))
        else:
            avg = getattr(scipy, error_avg)
            res2 = avg(fit_diff2)
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
        return (fit_diff2 > max_diff2).nonzero(), res2

    num_free_coef = len(derivatives)
    error_func = lambda coef: scipy.dot(coef, derivatives) - target_values
    deriv_func = lambda coef: derivatives
    initial_guess = scipy.zeros(num_free_coef)
    rej_iter = 0
    while True:
        logger.debug('%d sources, %d coefficients\n',
                     len(target_values),
                     num_free_coef)
        if len(target_values) < num_free_coef:
            return None, None, 0
        fit_results = scipy.optimize.leastsq(
            error_func,
            initial_guess,
            Dfun=deriv_func,
            col_deriv=1,
            full_output=1
        )
        if fit_results[4] not in [1, 2, 3, 4]:
            raise RuntimeError(
                "Linear least squares fitting failed for '%s': %s"
                %
                (fit_identifier, fit_results[3])
            )
        bad_ind, fit_res2 = rejected_indices(fit_results[2]['fvec'],
                                             weights)
        rej_iter += 1
        if not bad_ind or rej_iter == max_rej_iter:
            return fit_results[0], fit_res2, len(target_values)
        derivatives = scipy.delete(derivatives, bad_ind, 1)
        target_values = scipy.delete(target_values, bad_ind)
        if weights is not None:
            weights = scipy.delete(weights, bad_ind)
#pylint: enable=too-many-locals
