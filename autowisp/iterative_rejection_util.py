"""A collection of general purpose statistical manipulations of scipy arrays."""

import logging

import numpy
import scipy.linalg
from scipy.interpolate import UnivariateSpline

from autowisp.pipeline_exceptions import ConvergenceError

git_id = "$Id$"


# Too many arguments indeed, but most would never be needed.
# Breaking up into smaller pieces will decrease readability
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
def iterative_rejection_average(
    array,
    outlier_threshold,
    *,
    average_func=numpy.nanmedian,
    deviation_average=numpy.nanmean,
    max_iter=numpy.inf,
    axis=0,
    require_convergence=False,
    mangle_input=False,
    keepdims=False
):
    r"""
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
            :func:`numpy.nanmean` or :func:`numpy.nanmedian`\ ), must ignore nan
            values.

        deviation_average:    A function or iterable of functions to use for
            computing the average deviation. If multiple, the first is used for
            outlier rejection.

        max_iter:    The maximum number of rejection - re-fitting iterations
            to perform.

        axis:    The axis along which to compute the average.

        require_convergence:    If the maximum number of iterations is reached
            and still there are entries that should be rejected this argument
            determines what happens. If True, an exception is raised, if False,
            the last result is returned as final.

        mangle_input:    Is this function allowed to mangle the input array.

        keepdims:    See the keepdims argument of :func:`numpy.mean`\ .

    Returns:
        (tuple):
            array(float):
                An array with all axes of a other than axis being the same and
                the dimension along the axis-th axis being 1. Each entry if of
                average is independently computed from all other entries.

            array(float):
                An empirical estimate of the standard deviation around the
                returned `average` for each pixel. Calculated as RMS of the
                difference between individual values and the average divided by
                one less than the number of pixels contributing to that
                particular pixel's average. Has the same shape as above.

            array(int):
                The number of non-rejected non-NaN values included in the
                average of each pixel. Same shape as above.

    """

    logger = logging.getLogger(__name__)
    working_array = array if mangle_input else numpy.copy(array)
    logger.debug(
        "Calculating iterative rejection average along axis %d of array with "
        "shape %s:\n%s",
        axis,
        repr(working_array.shape),
        repr(working_array)
    )

    if isinstance(outlier_threshold, (float, int)):
        threshold_plus = outlier_threshold
        threshold_minus = -outlier_threshold
    else:
        if len(outlier_threshold) == 1:
            assert outlier_threshold[0] > 0
            threshold_plus = outlier_threshold[0]
            threshold_minus = -outlier_threshold[0]
        else:
            assert len(outlier_threshold) == 2
            assert outlier_threshold[0] * outlier_threshold[1] < 0
            if outlier_threshold[0] > 0:
                threshold_plus, threshold_minus = outlier_threshold
            else:
                threshold_minus, threshold_plus = outlier_threshold

    if not hasattr(deviation_average, "__getitem__"):
        deviation_average = (deviation_average,)

    iteration = 0
    found_outliers = True
    while found_outliers and iteration < max_iter:
        logger.debug("Average iteration: %s", repr(iteration))
        average = average_func(working_array, axis=axis, keepdims=True)
        difference = working_array - average
        rms = numpy.sqrt(
            deviation_average[0](
                numpy.square(difference), axis=axis, keepdims=True
            )
        )
        outliers = numpy.logical_or(
            difference < threshold_minus * rms,
            difference > threshold_plus * rms,
        )

        found_outliers = numpy.any(outliers)

        if found_outliers:
            working_array[outliers] = numpy.nan
        logger.debug("Found %d outliers.", found_outliers.sum())
        iteration = iteration + 1
    logger.debug("Exited found_outliers while loop")
    if found_outliers and require_convergence:
        raise ConvergenceError(
            "Computing "
            + average_func.__name__
            + " with iterative rejection did not converge after "
            + str(iteration)
            + " iterations!"
        )

    num_averaged = numpy.sum(
        numpy.logical_not(numpy.isnan(working_array)),
        axis=axis,
        keepdims=keepdims,
    )
    logger.debug("num_averaged computed: %s", num_averaged)
    average_dev = tuple(
        numpy.sqrt(
            avg_func(
                numpy.square(working_array - average),
                axis=axis,
                keepdims=keepdims,
            )
            / (num_averaged - 1)
        )
        for avg_func in deviation_average
    )
    if len(average_dev) == 1:
        average_dev = average_dev[0]

    logger.debug("Average deviation computed")

    if not keepdims:
        average = numpy.squeeze(average, axis)

    logger.debug("Finished average function!!!!")
    return average, average_dev, num_averaged


# pylint: enable=too-many-arguments
# pylint: enable=too-many-locals


def iterative_rej_linear_leastsq(
    matrix,
    rhs,
    outlier_threshold,
    max_iterations=numpy.inf,
    return_predicted=False,
):
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
        (tuple):
            array:
                The best fit coefficients.

            float:
                The root mean square residual of the latest fit iteration.

            array:
                The predicted values for the RHS. Only available if
                **return_predicted** == ``True``.
    """

    num_surviving = rhs.size
    iteration = 0
    fit_rhs = numpy.copy(rhs)
    fit_matrix = numpy.copy(matrix)
    while True:
        fit_coef, residual = scipy.linalg.lstsq(fit_matrix, fit_rhs)[:2]
        residual /= num_surviving
        if iteration == max_iterations:
            break
        outliers = (
            numpy.square(fit_rhs - fit_matrix.dot(fit_coef))
            > outlier_threshold**2 * residual
        )
        num_surviving -= outliers.sum()
        fit_rhs[outliers] = 0
        fit_matrix[outliers, :] = 0
        if not outliers.any():
            break
        iteration += 1
    if return_predicted:
        return fit_coef, numpy.sqrt(residual), matrix.dot(fit_coef)
    return fit_coef, numpy.sqrt(residual)


# x and y are perfectly readable arguments for a fitting function.
# pylint: disable=invalid-name
def iterative_rej_polynomial_fit(x, y, order, *leastsq_args, **leastsq_kwargs):
    r"""
    Fit for c_i in y = sum(c_i * x^i), iteratively rejecting outliers.

    Args:
        x:    The x (independent variable) in the polynomial.

        y:    The value predicted by the polynomial (y).

        order:    The maximum power of x term to include in the
            polynomial expansion.

        leastsq_args:    Passed directly to
            :func:`iterative_rej_linear_leastsq`\ .

        leastsq_kwargs:    Passed directly to
            :func:`iterative_rej_linear_leastsq`\ .

    Returns:
        See :func:`iterative_rej_linear_leastsq`\ .
    """

    matrix = numpy.empty((x.size, order + 1))
    matrix[:, 0] = 1.0
    for column in range(1, order + 1):
        matrix[:, column] = matrix[:, column - 1] * x

    return iterative_rej_linear_leastsq(
        matrix, y, *leastsq_args, **leastsq_kwargs
    )


def iterative_rejection_smoothing_spline(
    x, y, outlier_threshold, max_iterations=numpy.inf, **spline_args
):
    r"""
    Use scipy's UnivariateSpline for iterative rejection smoothing.

    Args:
        x:    The x (independent variable) in the dependence.

        y:    The y (dependenc variable) in the dependence.

        outlier_threshold:    See same name argument of
            :func:`iterative_rej_linear_leastsq`\ .

        max_iterations:    See same name argument of
            :func:`iterative_rej_linear_leastsq`\ .

        spline_args:    Keyword arguments passed directly to
            scipy.interpolate.UnivariateSpline.

    Returns:
        scipy.interpolate.UnivariateSpline:
            The latest iteration of the smoothing spline fit, after either the
            outlier rejection/refitting iterations have converged or
            max_iterations was reached.
    """

    logger = logging.getLogger(__name__)

    found_outliers = True
    iteration = 0
    fit_points = numpy.logical_and(numpy.isfinite(x), numpy.isfinite(y))
    fit_x = x[fit_points]
    fit_y = y[fit_points]
    if "w" in spline_args:
        fit_w = spline_args["w"][fit_points]
        del spline_args["w"]
    else:
        fit_w = None
    while found_outliers and iteration < max_iterations:
        smooth_func = UnivariateSpline(fit_x, fit_y, w=fit_w, **spline_args)

        square_residuals = numpy.square(smooth_func(fit_x) - fit_y)
        logger.debug("Square res:\n%s", repr(square_residuals))
        non_outliers = square_residuals < outlier_threshold**2 * numpy.mean(
            square_residuals
        )
        logger.debug("Non outliers: %s", repr(non_outliers))
        logger.debug(
            "Rejecting %d / %d points.",
            len(non_outliers) - non_outliers.sum(),
            len(non_outliers),
        )

        fit_x = fit_x[non_outliers]
        fit_y = fit_y[non_outliers]
        if fit_w is not None:
            fit_w = fit_w[non_outliers]

        found_outliers = not numpy.all(non_outliers)

    return smooth_func


# pylint: enable=invalid-name
