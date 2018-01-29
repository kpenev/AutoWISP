"""Define classes for smoothing images."""

from abc import ABC, abstractmethod

import scipy
import scipy.linalg
import scipy.interpolate

from superphot_pipeline.iterative_rejection_util import\
    iterative_rej_linear_leastsq
from superphot_pipeline.image_utilities import zoom_image, bin_image

git_id = '$Id$'

class ImageSmoother(ABC):
    """
    Define the interface for applying smoothing algorithms to images.

    Attrs:
        bin_factor:    See same name argument to smooth().

        zoom_interp_order:    See same name argument to smooth().
    """

    @abstractmethod
    def _apply_smoothing(self, image, **kwargs):
        """
        Return a smooth version of the given image (no pre-shrinking).

        Args:
            image:    The image to smooth.

            kwargs:    Any arguments configuring how smoothing is to
                be performed.

        Returns:
            smooth_image:    The smoothed version of the image per the currently
                defined smoothing.
        """

    def __init__(self, **kwargs):
        """
        Set default pre-shrink/post-zoom bin factor and interpolation order.

        Args:
            kwargs:    All arguments except bin_factor, interp_order and padding
                mode(see smooth()) are ignored.

        Returns:
            None
        """

        self.bin_factor = kwargs.get('bin_factor', None)
        self.zoom_interp_order = kwargs.get('zoom_interp_order', None)
        self.padding_mode = kwargs.get('padding_mode', None)

    def smooth(self,
               image,
               *,
               bin_factor=None,
               zoom_interp_order=None,
               padding_mode='reflect',
               **kwargs):
        """
        Sandwich smoothing between initial binning and final zoom.

        Args:
            image:    The image to smooth.

            bin_factor:    The factor to pre-bin the image by (see
                bin_image() for details). After smoothing the image is zoomed by
                the same factor to recover the original size. If None, the value
                defined at construction is used, otherwise this value applies
                for this image only.

            interp_order:    The interpolation order to use when zooming the
                image at the end. If None, the interpolation order defined at
                construction is used, otherwise this value applies for this
                image only.

            padding_mode:    How to pad the image to have an integer number of
                bins, if pre-binning and post-zooming is used.

            kwargs:    Any arguments configuring how smoothing is to
                be performed.

        Returns:
            smooth_image:    The smoothed version of the image.
        """

        if bin_factor is None:
            bin_factor = self.bin_factor
        if zoom_interp_order is None:
            zoom_interp_order = self.zoom_interp_order

        if bin_factor is None or zoom_interp_order is None:
            return self._apply_smoothing(image, **kwargs)

        y_res, x_res = image.shape

        if x_res % bin_factor != 0 or y_res % bin_factor != 0:
            y_padding = int(scipy.ceil(y_res / bin_factor)) * bin_factor - y_res
            x_padding = int(scipy.ceil(x_res / bin_factor)) * bin_factor - x_res
            left_padding = x_padding // 2
            bottom_padding = y_padding // 2
            image = scipy.pad(image,
                              ((bottom_padding, y_padding - bottom_padding),
                               (left_padding, x_padding - left_padding)),
                              mode=(padding_mode or self.padding_mode))

        smooth_image = zoom_image(
            self._apply_smoothing(
                bin_image(image, bin_factor),
                **kwargs
            ),
            bin_factor,
            zoom_interp_order
        )
        if x_res % bin_factor == 0 and y_res % bin_factor == 0:
            return smooth_image


        return smooth_image[bottom_padding : bottom_padding + y_res,
                            left_padding : left_padding + x_res]

    def detrend(self, image, **kwargs):
        """De-trend the input image by its smooth version (see smooth)."""

        smooth_image = self.smooth(image, **kwargs)
        return image / smooth_image * scipy.mean(smooth_image)

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

            x_resolution:    The resolution of the input image in the
                x direction.

        Returns:
            integrals:    A 1-D scipy array with length equal to the
                x-resolution of the image with the i-th entry being the integral
                of the x part of the smoothing function the i-th pixel.
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
                 max_iterations=None,
                 **kwargs):
        """
        Define the default smoothing configuration (overwritable on use).

        Args:
            num_x_terms:    The number of parameters of the x
                smoothing function.

            num_y_terms:    The number of parameters of the y
                smoothing function.

            y_resolution:    The y-resolution of the image being smoothed.

            x_resolution:    The x-resolution of the image being smoothed.

            kwargs:    Any arguments to pass to parent's __init__.

        Returns:
            None
        """

        super().__init__(**kwargs)
        self.num_x_terms = num_x_terms
        self.num_y_terms = num_y_terms
        self.outlier_threshold = outlier_threshold
        self.max_iterations = max_iterations

    #The abstract method was deliberately defined wit flexible arguments
    #pylint: disable=arguments-differ
    def _apply_smoothing(self,
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
            power:    The power of the corresponding coordinate of the term we
                are integrating.

            resolution:    The resolution of the image in the dimension in which to
                calculate the pixel integrlas.

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
    def get_spline_pixel_integrals(node_index,
                                   resolution,
                                   num_nodes,
                                   spline_degree):
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

            spline_degree:    The degree of the spline to use.

        Returns:
             integrals:    A 1-D scipy array with the i-th entry being the
                integral of the spline over the i-th pixel.
        """

        interp_y = scipy.zeros(num_nodes)
        interp_y[node_index] = 1.0
        integrate = scipy.interpolate.InterpolatedUnivariateSpline(
            scipy.arange(num_nodes),
            interp_y,
            k=spline_degree
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
                                               self.num_x_nodes,
                                               self.spline_degree)

    def get_y_pixel_integrals(self, param_ind, y_resolution):
        """
        Return integrals of the param_ind-th y direction spline basis function.

        Args:    See SeparableLinearImageSmoother.get_y_pixel_integrals()

        Returns:
            See SeparableLinearImageSmoother.get_y_pixel_integrals()
        """

        return self.get_spline_pixel_integrals(param_ind,
                                               y_resolution,
                                               self.num_y_nodes,
                                               self.spline_degree)

    def __init__(self,
                 *,
                 num_x_nodes=None,
                 num_y_nodes=None,
                 spline_degree=3,
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
        self.spline_degree = spline_degree

    #Different parameters are deliberate
    #pylint: disable=arguments-differ
    def _apply_smoothing(self,
                         image,
                         *,
                         num_x_nodes=None,
                         num_y_nodes=None,
                         **kwargs):
        """
        Handle change in interpolation nodes needed by integrals functions.

        Args:    See SeparableLinearImageSmoother._apply_smoothing() except the
            names of num_x_terms and num_y_terms have been changed to
            num_x_nodes and num_y_nodes respectively.

        Returns:
            None
        """

        if num_x_nodes is not None:
            self.num_x_nodes = num_x_nodes
        if num_y_nodes is not None:
            self.num_y_nodes = num_y_nodes

        return super()._apply_smoothing(image,
                                        num_x_terms=num_x_nodes,
                                        num_y_terms=num_y_nodes,
                                        **kwargs)
    #pylint: enable=arguments-differ

class WrapFilterAsSmoother(ImageSmoother):
    """Wrap one of the scipy or astropy image filters in a smoother."""

    def __init__(self,
                 smoothing_filter,
                 *,
                 bin_factor=None,
                 zoom_interp_order=None,
                 **filter_config):
        """
        Apply the given filter through the ImageSmoother interface.

        Args:
            smoothing_filter:    The filter to apply to smooth images.

            bin_factor:    See ImageSmoother.smooth().

            zoom_interp_order:    See ImageSmoother.smooth().

            filter_config:    Any arguments to pass to the filter when applying,
                unless overwritten by arguments to smooth() or detrend().

        Returns:
            None
        """

        super().__init__(bin_factor=bin_factor,
                         zoom_interp_order=zoom_interp_order)
        self.filter_config = filter_config
        self.filter = smoothing_filter

    def _apply_smoothing(self, image, **kwargs):
        """
        Smooth the given image with the filter supplied on construction.

        Args:
            image:    The image to smooth.

            kwargs:    Additional or replacement arguments to pass to the filter
                when applying.
        """

        filter_kwargs = self.filter_config
        filter_kwargs.update(kwargs)
        return self.filter(image, **filter_kwargs)

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

    def __init__(self, *smoothers, **kwargs):
        """
        Create a chain combining the given smoothers in the given order.

        Args:
            smoothers:    A list of the image smoothers to combine.

            kwargs:    Any arguments to pass to parent constructor.
        """


        super().__init__(**kwargs)
        self.smoothing_chain = []
        self.extend(smoothers)

    def append(self, smoother):
        """Add a new smoother to the end of the sequence."""

        assert isinstance(smoother, ImageSmoother)
        self.smoothing_chain.append(smoother)

    def extend(self, smoothers):
        """Add multiple smoothers to the end of the chain."""

        if isinstance(smoothers, ChainSmoother):
            self.smoothing_chain.extend(smoothers.smoothing_chain)
        else:
            for smth in smoothers:
                assert isinstance(smth, ImageSmoother)
            self.smoothing_chain.extend(smoothers)

    def insert(self, position, smoother):
        """Like list insert."""

        assert isinstance(smoother, ImageSmoother)
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
    def _apply_smoothing(self, image):
        """Smooth the given image using the current chain of smoothers."""

        smooth_image = image
        for smoother in self.smoothing_chain:
            smooth_image = smoother.smooth(smooth_image)
        return smooth_image
    #pylint: enable=arguments-differ
