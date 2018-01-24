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
            always outliers.

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

    threshold2 = outlier_threshold**2

    iteration = 0
    found_outliers = True
    while found_outliers and iteration < max_iter:
        average = average_func(working_array, axis=axis, keepdims=True)

        square_difference = scipy.square(working_array - average)
        outliers = (
            square_difference
            >
            threshold2 * scipy.mean(square_difference, axis=axis, keepdims=True)
        )

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
                                 max_iterations=scipy.inf):
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

    Returns:
        solution:    The best fit coefficients.

        residual:    The root mean square residual of the latest fit iteration.
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
    return fit_coef, scipy.sqrt(residual)
