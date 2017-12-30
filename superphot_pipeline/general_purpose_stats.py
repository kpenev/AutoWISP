"""A collection of general purpose statistical manipulations of numpy arrays."""

import numpy

from superphot_pipeline.pipeline_exceptions import ConvergenceError

git_id = '$Id$'

#Too many arguments indeed, but most would never be needed.
#pylint: disable=too-many-arguments
def iterative_rejection_average(array,
                                outlier_threshold,
                                average_func=numpy.nanmedian,
                                max_iter=numpy.inf,
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
            numpy.nanmean or numpy.nanmedian), must ignore nan values.

        axis:    The axis along which to compute the average.

        max_iter:    The maximum number of rejection - re-fitting iterations to
            perform.

        require_convergence:    If the maximum number of iterations is reached
            and still there are entries that should be rejected this argument
            determines what happens. If True, an exception is raised, if False,
            the last result is returned as final.

        mangle_input:    Is this function allowed to mangle the input array.

        keepdims:    See the keepdims argument of numpy.mean

    Returns:
        average:    An array with all axes of a other than axis being the same
            and the dimension along the axis-th axis being 1. Each entry if of
            average is independently computed from all other entries.
    """

    working_array = (array if mangle_input else numpy.copy(array))

    threshold2 = outlier_threshold**2

    iteration = 0
    found_outliers = True
    while found_outliers and iteration < max_iter:
        average = average_func(working_array, axis=axis, keepdims=True)

        square_difference = numpy.square(working_array - average)
        outliers = (
            square_difference
            >
            threshold2 * numpy.mean(square_difference, axis=axis, keepdims=True)
        )

        found_outliers = numpy.any(outliers)

        if found_outliers:
            working_array[outliers] = numpy.nan

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

    return average if keepdims else numpy.squeeze(average, axis)
#pylint: enable=too-many-arguments
