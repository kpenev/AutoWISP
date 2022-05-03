"""More convenient interface to estaver interpreters."""

from os import path

from asteval import asteval
from astropy.io import fits

from superphot_pipeline.fits_utilities import get_primary_header

class Evaluator(asteval.Interpreter):
    """Evaluator for expressions involving fields of numpy structured array."""

    def __init__(self, *data):
        """
        Get ready to evaluate expressions given data.

        Args:
            data([dict-like]):    A mapping between variable names that
                will participate in the expressions to be evaluated and the
                value that should be used. In case of repeating keys, later
                entries overwrite earlier ones.

        Returns:
            None
        """

        super().__init__()
        for data_entry in data:
            if hasattr(data_entry, 'dtype'):
                for varname in data_entry.dtype.names:
                    self.symtable[varname] = data_entry[varname]
            elif (
                (isinstance(data_entry, str) and path.exists(data_entry))
                or
                isinstance(data_entry, fits.HDUList)
            ):
                self.symtable.update(get_primary_header(data_entry))
            else:
                self.symtable.update(data_entry)
