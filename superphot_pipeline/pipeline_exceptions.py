"""Collection of non-standard exceptions raised by the pipeline."""

class OutsideImageError(IndexError):
    """Attempt to access image data outside the bounds of the image."""


class ImageMismatchError(ValueError):
    """Attempt to combine incompatible images in some way."""

class ConvergenceError(RuntimeError):
    """Some iterative procedure failed to converge."""
