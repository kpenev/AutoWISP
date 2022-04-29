"""More convenient interface to estaver interpreters."""

from os import path

from asteval import asteval
from astropy.io import fits

from superphot_pipeline.fits_utilities import get_primary_header

class Evaluator(asteval.Interpreter):
    """Evaluator for expressions involving fields of numpy structured array."""

    def __init__(self, data):
        """Get ready."""

        super().__init__()
        if hasattr(data, 'dtype'):
            for varname in data.dtype.names:
                self.symtable[varname] = data[varname]
        elif (
            (isinstance(data, str) and path.exists(data))
            or
            isinstance(data, fits.HDUList)
        ):
            self.symtable.update(get_primary_header(data))
        else:
            self.symtable.update(data)
