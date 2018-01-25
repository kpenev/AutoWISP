"""Define classes for smoothing images."""

from abc import ABC, abstractmethod

import scipy
import scipy.linalg
import scipy.interpolate

from superphot_pipeline.iterative_rejection_util import\
    iterative_rej_linear_leastsq

git_id = '$Id$'

class ImageSmoother(ABC):
    """
    Define the interface for applying smoothing algorithms to images.
    """

    @abstractmethod
    def smooth(self, image, **kwargs):
        """
        Return a smooth version of the given image.

        Args:
            image:    The image to smooth.

            kwargs:    Any arguments configuring how smoothing is to
                be performed.

        Returns:
            smooth_image:    The smoothed version of the image per the currently
                defined smoothing.
        """

    def detrend(self, image, **kwargs):
        """De-trend the input image by its smooth version (see smooth)."""

        smooth_image = self.smooth(image, **kwargs)
        return image / smooth_image


class SeparableLinearImageSmoother(ImageSmoother):
    """
    Handle smoothing function = product of linear functions in each dimension.

    In more detail this is a base class that perform smoothing with a smoothing
    function consisting of the product of separate smoothing functions in x and
    y, each of which predicts pixel values as a linear combination of some
    parameters.

    Attrs:
        num_x_terms:    The number of terms in the smoothing function in the x
            direction.

        num_y_terms:    The number of terms in the smoothing function in the y
            direction.
    """

    @abstractmethod
    def get_x_pixel_integrals(self, param_ind, x_resolution):
        """
        Return the x smoothing func. integral over pixels for 1 nonzero param.

        Args:
            param_ind:    The index of the input parameter which is non-zero.

        Returns:
            integrals:    A 1-D scipy array with length equal to the
                x-resolution of the image with the i-th entry being the integral
                of the x part of the smoothing function the i-th pixel.

            x_resolution:    The resolution of the input image in the x
                direction.
        """

    @abstractmethod
    def get_y_pixel_integrals(self, param_ind, y_resolution):
        """See get_x_pixel_integrals."""

    def _get_smoothing_matrix(self,
                              num_x_terms,
                              num_y_terms,
                              y_resolution,
                              x_resolution):
        """
        Return matrix giving flattened smooth image when applied to fit params.

        Args:    See __init__().

        Returns:
            matrix:    An (x_res * y_res) by (num_x_terms * num_y_terms) matrix
                which when applied to a set of parameters returns the value of
                each image pixel per the smoothing function.
        """

        matrix = scipy.empty((x_resolution * y_resolution,
                              num_x_terms * num_y_terms))
        for x_term_index in range(num_x_terms):
            x_integrals = self.get_x_pixel_integrals(x_term_index, x_resolution)
            for y_term_index in range(num_y_terms):
                y_integrals = self.get_y_pixel_integrals(y_term_index,
                                                         y_resolution)
                matrix[
                    :,
                    x_term_index + y_term_index * num_x_terms
                ] = scipy.outer(y_integrals, x_integrals).flatten()

        return matrix

    def __init__(self,
                 *,
                 num_x_terms=None,
                 num_y_terms=None,
                 outlier_threshold=None,
                 max_iterations=None):
        """
        Define the default smoothing configuration (overwritable on use).

        Args:
            num_x_terms:    The number of parameters of the x
                smoothing function.

            num_y_terms:    The number of parameters of the y
                smoothing function.

            y_resolution:    The y-resolution of the image being smoothed.

            x_resolution:    The x-resolution of the image being smoothed.

        Returns:
            None
        """

        self.num_x_terms = num_x_terms
        self.num_y_terms = num_y_terms
        self.outlier_threshold = outlier_threshold
        self.max_iterations = max_iterations

    #The abstract method was deliberately defined wit flexible arguments
    #pylint: disable=arguments-differ
    def smooth(self,
               image,
               *,
               num_x_terms=None,
               num_y_terms=None,
               outlier_threshold=None,
               max_iterations=None):
        """
        Return a smooth version of the given image.

        Args:
            image:    The image to smooth.

        Returns:
            smooth_image:    The best approximation of the input image possible
                with the smoothing function.

            residual:    The root mean square residual returned
                by iterative_rej_linear_leastsq()
        """

        if outlier_threshold is None:
            outlier_threshold = self.outlier_threshold
        if max_iterations is None:
            max_iterations = self.max_iterations
        if num_x_terms is None:
            num_x_terms = self.num_x_terms
        if num_y_terms is None:
            num_y_terms = self.num_y_terms

        matrix = self._get_smoothing_matrix(num_x_terms,
                                            num_y_terms,
                                            *image.shape)

        fit_coef = iterative_rej_linear_leastsq(
            matrix,
            image.flatten(),
            outlier_threshold=outlier_threshold,
            max_iterations=max_iterations
        )[0]
        smooth_image = matrix.dot(fit_coef).reshape(image.shape)
        return smooth_image
    #pylint: enable=arguments-differ

class PolynomialImageSmoother(SeparableLinearImageSmoother):
    """
    Smooth image is modeled as a polynomial in x times another polynomial in y.
    """

    @staticmethod
    def _get_powerlaw_pixel_integrals(power, resolution):
        """
        Return the integrals over one pixel dimension of x^power for each pixel.

        Args:
            resolution:    The resolution of the image in the dimension in which to
                calculate the pixel integrlas.

            power:    The power of the corresponding coordinate of the term we
                are integrating.

        Returns:
            integrals:    A 1-D scipy array with the i-th entry being the
                integral of the scaled x coordinate to the given power over
                each pixel.
        """

        pix_left = scipy.arange(resolution)
        return (
            (2.0 * (pix_left + 1) / resolution - 1)**(power + 1)
            -
            (2.0 * pix_left / resolution - 1)**(power + 1)
        ) / (power + 1)

    get_x_pixel_integrals = _get_powerlaw_pixel_integrals
    get_y_pixel_integrals = _get_powerlaw_pixel_integrals

class SplineImageSmoother(SeparableLinearImageSmoother):
    """Smooth image is modeled as a product of cubic splines in x and y."""

    @staticmethod
    def get_spline_pixel_integrals(node_index, resolution, num_nodes):
        """
        Return the integrals over one pixel dimension of a basis spline.

        The spline basis functions are defined as an interpolating spline (i.e.
        no smoothing) over y values that are one for the `node_index`-th node
        and zero everywhere else.

        Args:
            node_index:    The node at which the spline should evaluate to
                one. All other nodes get a value of zero.

            resolution:    The resolution of the image in the dimension in which
                to calculate the pixel integrlas.

            num_nodes:    The number of nodes in the spline. The image is scaled
                to have a dimension of `num_nodes - 1` and nodes are set at
                integer values.

        Returns:
             integrals:    A 1-D scipy array with the i-th entry being the
                integral of the spline over the i-th pixel.
        """

        interp_y = scipy.zeros(num_nodes)
        interp_y[node_index] = 1.0
        integrate = scipy.interpolate.InterpolatedUnivariateSpline(
            scipy.arange(num_nodes),
            interp_y
        ).antiderivative()
        cumulative_integrals = integrate(scipy.arange(resolution + 1)
                                         *
                                         ((num_nodes  - 1) / resolution))
        return cumulative_integrals[1:] - cumulative_integrals[:-1]

    def get_x_pixel_integrals(self, param_ind, x_resolution):
        """
        Return integrals of the param_ind-th x direction spline basis function.

        Args:    See SeparableLinearImageSmoother.get_x_pixel_integrals()

        Returns:
            See SeparableLinearImageSmoother.get_x_pixel_integrals()
        """

        return self.get_spline_pixel_integrals(param_ind,
                                               x_resolution,
                                               self.num_x_nodes)

    def get_y_pixel_integrals(self, param_ind, y_resolution):
        """
        Return integrals of the param_ind-th y direction spline basis function.

        Args:    See SeparableLinearImageSmoother.get_y_pixel_integrals()

        Returns:
            See SeparableLinearImageSmoother.get_y_pixel_integrals()
        """

        return self.get_spline_pixel_integrals(param_ind,
                                               y_resolution,
                                               self.num_y_nodes)

    def __init__(self,
                 *,
                 num_x_nodes=None,
                 num_y_nodes=None,
                 **kwargs):
        """
        Set-up spline interpolation with the given number of nodes.

        Args:
            num_x_nodes:    The number of spline nodes for the x spline.

            num_y_nodes:    The number of spline nodes for the y spline.

            kwargs:    Forwarded directly
                to SeparableLinearImageSmoother.__init__() after
                `num_x_terms = num_x_nodes` and `num_y_terms = num_y_nodes`.

        Returns:
            None
        """

        super().__init__(num_x_terms=num_x_nodes,
                         num_y_terms=num_y_nodes,
                         **kwargs)
        self.num_x_nodes = num_x_nodes
        self.num_y_nodes = num_y_nodes

    #Different parameters are deliberate
    #pylint: disable=arguments-differ
    def smooth(self, image, *, num_x_nodes=None, num_y_nodes=None, **kwargs):
        """
        Handle change in interpolation nodes needed by integrals functions.

        Args:    See SeparableLinearImageSmoother.smooth() except the names of
            num_x_terms and num_y_terms have been changed to num_x_nodes and
            num_y_nodes respectively.

        Returns:
            None
        """

        if num_x_nodes is not None:
            self.num_x_nodes = num_x_nodes
        if num_y_nodes is not None:
            self.num_y_nodes = num_y_nodes

        return super().smooth(image,
                              num_x_terms=num_x_nodes,
                              num_y_terms=num_y_nodes,
                              **kwargs)
    #pylint: enable=arguments-differ

class ChainSmoother(ImageSmoother):
    """
    Combine more than one smoothers, each is applied sequentially.

    Works much like a list of smoothers, except it adds smooth and detrend
    methods.

    Attrs:
        smoothing_chain:    The current list of smoothers and the order in which
            they are applied. The first will be applied to the input image, the
            second will be applied to the result of the first etc.
    """

    def __init__(self, *smoothers):
        """
        Create a chain combining the given smoothers in the given order.

        Args:
            smoothers:    A list of the image smoothers to combine.
        """


        self.smoothing_chain = []
        self.extend(smoothers)

    def append(self, smoother):
        """Add a new smoother to the end of the sequence."""

        assert isinstance(ImageSmoother, smoother)
        self.smoothing_chain.append(smoother)

    def extend(self, smoothers):
        """Add multiple smoothers to the end of the chain."""

        if isinstance(ChainSmoother, smoothers):
            self.smoothing_chain.extend(smoothers.smoothing_chain)
        else:
            for smth in smoothers:
                assert isinstance(ImageSmoother, smth)
            self.smoothing_chain.extend(smoothers)

    def insert(self, position, smoother):
        """Like list insert."""

        assert isinstance(ImageSmoother, smoother)
        self.smoothing_chain.insert(position, smoother)

    def remove(self, smoother):
        """Like list remove."""

        self.smoothing_chain.remove(smoother)

    def pop(self, position=-1):
        """Like list pop."""

        self.smoothing_chain.pop(position)

    def clear(self):
        """Like list clear."""

        self.smoothing_chain.clear()

    def __delitem__(self, position):
        """Delete smoothe at position."""

        del self.smoothing_chain[position]

    def __setitem__(self, position, smoother):
        """Replace the smoother at position."""

        self.smoothing_chain[position] = smoother

    #It makes no sense to take configuration argumens.
    #pylint: disable=arguments-differ
    def smooth(self, image):
        """Smooth the given image using the current chain of smoothers."""

        smooth_image = image
        for smoother in self.smoothing_chain:
            smooth_image = smoother.smooth(smooth_image)
        return smooth_image
    #pylint: enable=arguments-differ
