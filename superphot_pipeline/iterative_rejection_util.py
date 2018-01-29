"""A collection of general purpose statistical manipulations of scipy arrays."""

import scipy
import scipy.linalg

from superphot_pipeline.pipeline_exceptions import ConvergenceError

git_id = '$Id$'

#Too many arguments indeed, but most would never be needed.
#Breaking up into smaller pieces will decrease readability
#pylint: disable=too-many-arguments
#pylint: disable=too-many-locals
def iterative_rejection_average(array,
                                outlier_threshold,
                                average_func=scipy.nanmedian,
                                max_iter=scipy.inf,
                                axis=0,
                                require_convergence=False,
                                mangle_input=False,
                                keepdims=False):
    """
    Avarage with iterative rejection of outliers along an axis.

    Notes:
        A more efficient implementation is possible for median.

    Args:
        array:    The array to compute the average of.

        outlier_threshold:    Outliers are defined as outlier_threshold * (root
            maen square deviation around the average). Non-finite values are
            always outliers. This value could also be a 2-tuple with one
            positive and one negative entry, specifying the thresholds in the
            positive and negative directions separately.

        average_func:    A function which returns the average to compute (e.g.
            scipy.nanmean or scipy.nanmedian), must ignore nan values.

        axis:    The axis along which to compute the average.

        max_iter:    The maximum number of rejection - re-fitting iterations
            to perform.

        require_convergence:    If the maximum number of iterations is reached
            and still there are entries that should be rejected this argument
            determines what happens. If True, an exception is raised, if False,
            the last result is returned as final.

        mangle_input:    Is this function allowed to mangle the input array.

        keepdims:    See the keepdims argument of scipy.mean

    Returns:
        average:    An array with all axes of a other than axis being the same
            and the dimension along the axis-th axis being 1. Each entry if of
            average is independently computed from all other entries.

        stdev:    An empirical estimate of the standard deviation around the
            returned `average` for each pixel. Calculated as RMS of the
            difference between individual values and the average divided by one
            less than the number of pixels contributing to that particular
            pixel's average. Has the same shape as `average`.

        num_averaged:    The number of non-rejected non-NaN values included
            in the average of each pixel. Same shape as `average`.
    """

    working_array = (array if mangle_input else scipy.copy(array))

    if isinstance(outlier_threshold, (float, int)):
        threshold_plus = outlier_threshold
        threshold_minus = -outlier_threshold
    else:
        assert len(outlier_threshold) == 2
        assert outlier_threshold[0] * outlier_threshold[1] < 0
        if outlier_threshold[0] > 0:
            threshold_plus, threshold_minus = outlier_threshold
        else:
            threshold_minus, threshold_plus = outlier_threshold

    iteration = 0
    found_outliers = True
    while found_outliers and iteration < max_iter:
        average = average_func(working_array, axis=axis, keepdims=True)
        difference = working_array - average
        rms = scipy.sqrt(
            scipy.mean(
                scipy.square(difference),
                axis=axis,
                keepdims=True
            )
        )
        outliers = scipy.logical_or(difference < threshold_minus * rms,
                                    difference > threshold_plus * rms)

        found_outliers = scipy.any(outliers)

        if found_outliers:
            working_array[outliers] = scipy.nan

    if found_outliers and require_convergence:
        raise ConvergenceError(
            'Computing '
            +
            average_func.__name__
            +
            ' with iterative rejection did not converge after '
            +
            str(iteration)
            +
            ' iterations!'
        )

    num_averaged = scipy.sum(scipy.logical_not(scipy.isnan(working_array)),
                             axis=axis,
                             keepdims=keepdims)

    stdev = (
        scipy.sqrt(
            scipy.nanmean(scipy.square(working_array - average),
                          axis=axis,
                          keepdims=keepdims)
            /
            (num_averaged - 1)
        )
    )

    if not keepdims:
        average = scipy.squeeze(average, axis)

    return average, stdev, num_averaged
#pylint: enable=too-many-arguments
#pylint: enable=too-many-locals

def iterative_rej_linear_leastsq(matrix,
                                 rhs,
                                 outlier_threshold,
                                 max_iterations=scipy.inf,
                                 return_predicted=False):
    """
    Perform linear leasts squares fit iteratively rejecting outliers.

    The returned function finds vector x that minimizes the square difference
    between matrix.dot(x) and rhs, iterating between fitting and  rejecting RHS
    entries which are too far from the fit.

    Args:
        matrix:    The matrix defining the linear least squares problem.

        rhs:    The RHS of the least squares problem.

        outlier_threshold:    The RHS entries are considered outliers if they
            devite from the fit by more than this values times the root mean
            square of the fit residuals.

        max_iterations:    The maximum number of rejection/re-fitting iterations
            allowed. Zero for simple fit with no rejections.

        return_predicted:    Should the best-fit values for the RHS be returned?

    Returns:
        solution:    The best fit coefficients.

        residual:    The root mean square residual of the latest fit iteration.

        predicted:    The predicted values for the RHS. Only available if
            `return_predicted==True`.
    """

    num_surviving = rhs.size
    iteration = 0
    fit_rhs = scipy.copy(rhs)
    fit_matrix = scipy.copy(matrix)
    while True:
        fit_coef, residual = scipy.linalg.lstsq(fit_matrix, fit_rhs)[:2]
        residual /= num_surviving
        if iteration == max_iterations:
            break
        outliers = (scipy.square(fit_rhs - fit_matrix.dot(fit_coef))
                    >
                    outlier_threshold**2 * residual)
        num_surviving -= outliers.sum()
        fit_rhs[outliers] = 0
        fit_matrix[outliers, :] = 0
        if not outliers.any():
            break
        iteration += 1
    if return_predicted:
        return fit_coef, scipy.sqrt(residual), matrix.dot(fit_coef)
    return fit_coef, scipy.sqrt(residual)

#x and y are perfectly readable arguments for a fitting function.
#pylint: disable=invalid-name
def iterative_rej_polynomial_fit(x,
                                 y,
                                 order,
                                 *leastsq_args,
                                 **leastsq_kwargs):
    """
    Fit for c_i in y = sum(c_i * x^i), iteratively rejecting outliers.

    Args:
        x:    The x (independent variable) in the polynomial.

        y:    The value predicted by the polynomial (y).

        order:    The maximum power of x term to include in the polynomial
            expansion.

        leastsq_args:    Passed directly to iterative_rej_linear_leastsq().
        leastsq_kwargs:    Passed directly to iterative_rej_linear_leastsq().

    Returns:
        solution:    See iterative_rej_linear_leastsq()

        residual:    See iterative_rej_linear_leastsq()
    """

    matrix = scipy.empty((x.size, order+1))
    matrix[:, 0] = 1.0
    for column in range(1, order + 1):
        matrix[:, column] = matrix[:, column - 1] * x

    return iterative_rej_linear_leastsq(matrix,
                                        y,
                                        *leastsq_args,
                                        **leastsq_kwargs)
#pylint: enable=invalid-name
