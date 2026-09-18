"""A collection of general purpose statistical manipulations of scipy arrays."""

import logging

import numpy
import scipy.linalg
from scipy.interpolate import splrep, BSpline

from autowisp.exceptions import ConvergenceError

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
    keepdims=False,
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
        repr(working_array),
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


def flag_outliers(residuals, threshold):
    """Flag outlier residuals (see :func:`iterative_rej_linear_leastsq`)."""

    try:
        if len(threshold) == 1:
            upper_threshold = lower_threshold = threshold[0]
        else:
            upper_threshold, lower_threshold = float(threshold[0]), float(
                threshold[1]
            )
    except TypeError:
        upper_threshold = lower_threshold = float(threshold)

    rms = numpy.sqrt(numpy.mean(residuals**2))
    return numpy.logical_or(
        residuals > upper_threshold * rms, residuals < -lower_threshold * rms
    )


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

        rhs:    The RHS of the least squares problem. Non-finite entries are
            always treated as outliers.

        outlier_threshold:    The RHS entries are considered outliers if they
            devite from the fit by more than this values times the root mean
            square of the fit residuals.

        max_iterations:    The maximum number of rejection/re-fitting iterations
            allowed. Zero for simple fit with no rejections (other than of
            non-finite RHS entries).

        return_predicted:    Should the best-fit values for the RHS be returned?

    Returns:
        (tuple):
            array:
                The best fit coefficients. All NaN, as are the other returned
                values, if fewer RHS entries survive than there are
                coefficients to fit.

            float:
                The root mean square residual of the latest fit iteration.

            array:
                The predicted values for the RHS. Only available if
                **return_predicted** == ``True``.
    """

    outlier_threshold = numpy.atleast_1d(outlier_threshold)
    num_surviving = rhs.size
    iteration = 0
    fit_rhs = numpy.copy(rhs)
    fit_matrix = numpy.copy(matrix)
    outliers = numpy.logical_not(numpy.isfinite(rhs))
    while iteration == 0 or outliers.any():
        num_surviving -= outliers.sum()
        fit_rhs[outliers] = 0
        fit_matrix[outliers, :] = 0

        if num_surviving < matrix.shape[1]:
            fit_coef = numpy.full(matrix.shape[1], numpy.nan)
            residual = numpy.nan
            break

        fit_coef, residual = scipy.linalg.lstsq(fit_matrix, fit_rhs)[:2]
        residual /= num_surviving
        if iteration == max_iterations:
            break
        outliers = flag_outliers(
            fit_rhs - fit_matrix.dot(fit_coef),
            outlier_threshold * numpy.sqrt(rhs.size / num_surviving),
        )
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


def _estimate_smoothing(knots, fit_x, fit_y, fit_w, spline_args):
    """
    Return the smoothing condition the noise about a spline calls for.

    A smoothing spline cannot supply this itself: its squared residuals sum
    to whatever ``s`` it was given. So the noise is instead estimated from a
    least squares fit with the same knots, which is under no such
    constraint, as ``SSR / (m - p)`` for ``p`` coefficients. The smoothing
    condition scipy recommends for that noise is then ``(m - sqrt(2 * m))``
    times it.

    Args:
        knots:    The full knot vector of the smoothing fit, as returned by
            :func:`scipy.interpolate.splrep`\\ .

        fit_x, fit_y, fit_w:    The points to fit and their weights (or
            ``None``).

        spline_args:    The remaining arguments of the smoothing fit.

    Returns:
        float or None:    The smoothing condition, or ``None`` if there are
            no more points than coefficients, leaving no noise to estimate.
    """

    degree = spline_args.get("k", 3)
    num_points = fit_x.size
    num_coefficients = knots.size - degree - 1
    if num_points <= num_coefficients:
        return None

    fixed_knot_args = {
        name: value for name, value in spline_args.items() if name != "s"
    }
    fixed_knot_args["t"] = knots[degree + 1 : -(degree + 1)]
    residuals = fit_y - BSpline(
        *splrep(fit_x, fit_y, w=fit_w, task=-1, **fixed_knot_args)
    )(fit_x)
    if fit_w is not None:
        residuals = residuals * fit_w

    return (
        (num_points - numpy.sqrt(2.0 * num_points))
        * numpy.sum(residuals**2)
        / (num_points - num_coefficients)
    )


def iterative_rej_smoothing_spline(  # pylint: disable=too-many-locals
    x, y, outlier_threshold, max_iterations=numpy.inf, **spline_args
):
    r"""
    Use scipy's UnivariateSpline for iterative rejection smoothing.

    Args:
        x:    The x (independent variable) in the dependence.

        y:    The y (dependenc variable) in the dependence.

        outlier_threshold:    See same name argument of
            :func:`iterative_rej_linear_leastsq`\ . If two values are given, the
            first indicates positive (i.e. rhs > matrix * coefficients) and the
            second negative (rhs < matrix * coefficients) deviations.

        max_iterations:    See same name argument of
            :func:`iterative_rej_linear_leastsq`\ .

        spline_args:    Keyword arguments passed directly to
            :func:`scipy.interpolate.splrep`\ . If the smoothing condition
            ``s`` is given (without knots ``t``), it applies to the first fit
            only, and should be loose enough for the spline not to bend
            towards the outliers: a value suitable for the clean data would
            leave nothing to reject. After each round of rejection, ``s`` is
            re-estimated from the noise about a least squares fit with the
            knots of the latest smoothing fit (see
            :func:`_estimate_smoothing`), and the iterations stop once a
            round rejects nothing and fails to shrink ``s``. If ``s`` is not
            given, :func:`scipy.interpolate.splrep`\ 's own default applies
            to every fit, and the iterations stop once nothing is rejected.

    Returns:
        scipy.interpolate.BSpline:
            The latest iteration of the smoothing spline fit, after either the
            outlier rejection/refitting iterations have converged or
            max_iterations was reached.
    """

    logger = logging.getLogger(__name__)

    iteration = 0
    fit_points = numpy.logical_and(numpy.isfinite(x), numpy.isfinite(y))
    fit_x = x[fit_points]
    fit_y = y[fit_points]
    if "w" in spline_args:
        fit_w = spline_args["w"][fit_points]
        del spline_args["w"]
    else:
        fit_w = None
    # With fixed knots scipy ignores s, so there is nothing to adapt.
    adapt_smoothing = "s" in spline_args and "t" not in spline_args
    while True:
        knots, *spline = splrep(fit_x, fit_y, w=fit_w, **spline_args)
        smooth_func = BSpline(knots, *spline)
        if iteration == max_iterations:
            break

        residuals = fit_y - smooth_func(fit_x)
        logger.debug("Residuals:\n%s", repr(residuals))
        non_outliers = numpy.logical_not(
            flag_outliers(residuals, outlier_threshold)
        )
        logger.debug("Non outliers: %s", repr(non_outliers))
        logger.debug(
            "Rejecting %d / %d points.",
            len(non_outliers) - non_outliers.sum(),
            len(non_outliers),
        )
        found_outliers = not numpy.all(non_outliers)
        if not (found_outliers or adapt_smoothing):
            break

        fit_x = fit_x[non_outliers]
        fit_y = fit_y[non_outliers]
        if fit_w is not None:
            fit_w = fit_w[non_outliers]
        if adapt_smoothing:
            new_smoothing = _estimate_smoothing(
                knots, fit_x, fit_y, fit_w, spline_args
            )
            logger.debug("Estimated smoothing: %s", repr(new_smoothing))
            # Once converged, s alternates between nearly equal values of
            # neighbouring knot sets rather than settling exactly, so stop at
            # the first round that fails to shrink it.
            if not found_outliers and (
                new_smoothing is None or new_smoothing >= spline_args["s"]
            ):
                break
            if new_smoothing is not None:
                spline_args["s"] = new_smoothing
        iteration += 1

    return smooth_func


# pylint: enable=invalid-name
