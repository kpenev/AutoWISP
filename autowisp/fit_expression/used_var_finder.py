"""Define dict-like class tracking vars used when evaluating fit expressions."""

import numpy


class UsedVarFinder:
    """Record the names of all keys not in default symtable ever requested."""

    def __contains__(self, key):
        """Is the given key in either the default symtable or data dtype."""

        return True

    def __getitem__(self, key):
        """1.0 if key is one of the data otherwise default symtable entry"""

        if key in self._default_symtable:
            return self._default_symtable[key]

        self._used.add(key)
        return numpy.array([1.0])

    def __init__(self, default_symtable):
        """Set the symtable before data is given and dtype to track usage of."""

        self._default_symtable = default_symtable
        self._used = set()

    def get_used_vars(self):
        """Return the names of the variables used so far."""

        return self._used
