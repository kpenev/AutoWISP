"""More convenient interface to estaver interpreters."""

from asteval import asteval

class Evaluator(asteval.Interpreter):
    """Evaluator for expressions involving fields of numpy structured array."""

    def __init__(self, data):
        """Get ready."""

        super().__init__()
        for varname in data.dtype.names:
            self.symtable[varname] = data[varname]
