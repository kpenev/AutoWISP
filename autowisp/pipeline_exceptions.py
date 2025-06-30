"""Collection of non-standard exceptions raised by the pipeline."""

git_id = "$Id$"


class OutsideImageError(IndexError):
    """Attempt to access image data outside the bounds of the image."""


class ImageMismatchError(ValueError):
    """Attempt to combine incompatible images in some way."""


class BadImageError(ValueError):
    """An image does not look like it is expected to."""


class ConvergenceError(RuntimeError):
    """Some iterative procedure failed to converge."""


class HDF5LayoutError(RuntimeError):
    """Error caused by invalid specification of HDF5 layout."""
