"""Define a Hashable wrapper for ndarray objects."""

from hashlib import sha1
import numpy


class HashableArray:
    """
    Hashable wrapper for ndarray objects

    Copied from:
    http://stackoverflow.com/questions/1939228/constructing-a-python-set-from-a-numpy-matrix

    Instances of ndarray are not hashable, meaning they cannot be added to
    sets, nor used as keys in dictionaries. This is by design - ndarray
    objects are mutable, and therefore cannot reliably implement the
    __hash__() method.

    The hashable class allows a way around this limitation. It implements
    the required methods for hashable objects in terms of an encapsulated
    ndarray object. This can be either a copied instance (which is safer)
    or the original object (which requires the user to be careful enough
    not to modify it).
    """

    def __init__(self, wrapped, tight=False):
        """
        Creates a new hashable object encapsulating an ndarray.

        Args:
            - wrapped: The wrapped ndarray.
            - tight: If True, a copy of the input ndaray is created.
        """

        assert not isinstance(wrapped, HashableArray)
        self._tight = tight
        self._wrapped = numpy.array(wrapped) if tight else wrapped
        self._hash = int(sha1(wrapped.view(numpy.uint8)).hexdigest(), 16)
        self.dtype = wrapped.dtype

    # Other is expected to be of the same class.
    def __eq__(self, other):

        # pylint:disable=protected-access
        return isinstance(other, HashableArray) and all(
            self._wrapped == other._wrapped
        )
        # pylint:enable=protected-access

    def __hash__(self):
        return self._hash

    def unwrap(self):
        """
        Return the encapsulated ndarray.

        If the wrapper is "tight", a copy of the encapsulated ndarray is
        returned. Otherwise, the encapsulated ndarray itself is returned.
        """

        if self._tight:
            return numpy.array(self._wrapped)

        return self._wrapped
