"""More convenient interface to asteval interpreters."""

from os import path

import numpy
from asteval import asteval
from astropy.io import fits
from astropy import units
import pandas

from autowisp.fits_utilities import get_primary_header
from autowisp.data_reduction.data_reduction_file import DataReductionFile


class EvaluatorBase(asteval.Interpreter):
    """Asteval interpreter with the symbols all AutoWISP expressions share."""

    # The NaN-ignoring aggregates available to every user expression. Listed
    # explicitly rather than derived from ``dir(numpy)`` so that the names
    # users can rely on are a property of AutoWISP and not of whichever numpy
    # or asteval version happens to be installed. Asteval's default symtable
    # supplies only some of these, and which ones has varied between releases.
    nan_aggregates = (
        "nanmean",
        "nanmedian",
        "nanstd",
        "nanvar",
        "nansum",
        "nanprod",
        "nanmin",
        "nanmax",
        "nanpercentile",
        "nanquantile",
        "nanargmin",
        "nanargmax",
        "nancumsum",
        "nancumprod",
    )

    def __init__(self):
        """
        Create an interpreter with the NaN-ignoring aggregates defined.

        Sub-classes should add their data to the symbol table after invoking
        this, so a variable named like one of the aggregates shadows the
        function rather than the other way around.

        Returns:
            None
        """

        super().__init__()
        for func_name in self.nan_aggregates:
            self.symtable[func_name] = getattr(numpy, func_name)

    def __call__(self, *args, **kwargs):
        """
        Evaluate the expression, raising rather than returning None on error.

        Asteval defaults to printing the error and returning ``None``, which
        only relocates the failure: the ``None`` flows on and breaks somewhere
        unrelated, long after the message about the offending expression has
        scrolled away. Callers wanting the permissive behaviour can still pass
        ``raise_errors=False`` explicitly.
        """

        if "raise_errors" not in kwargs:
            kwargs["raise_errors"] = True
        return super().__call__(*args, **kwargs)


class Evaluator(EvaluatorBase):
    """Evaluator for expressions involving fields of numpy array or headers."""

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
            if hasattr(data_entry, "dtype"):
                for varname in data_entry.dtype.names:
                    self.symtable[varname] = data_entry[varname]
            elif isinstance(data_entry, pandas.DataFrame):
                for varname in data_entry:
                    self.symtable[varname] = data_entry[varname].to_numpy()
            elif isinstance(data_entry, str) and path.exists(data_entry):
                if path.splitext(data_entry)[-1] in [".h5", ".hdf5"]:
                    with DataReductionFile(data_entry, "r") as dr_file:
                        self.__init__(dr_file.get_frame_header())
                else:
                    with fits.open(data_entry, "readonly") as opened_fits:
                        self.__init__(get_primary_header(opened_fits))
            elif isinstance(data_entry, fits.HDUList):
                self.__init__(get_primary_header(data_entry))
            else:
                for hdr_key, hdr_val in data_entry.items():
                    self.symtable[hdr_key.replace("-", "_")] = hdr_val
        self.symtable["units"] = units


# Needed to implement real-time lookup of datasets
# pylint: disable=too-few-public-methods
class LightCurveLookUp:
    """Look up datasets under specific key prefix."""

    def __init__(self, key_prefix, evaluator):
        """Look up datasets with keys like <key_prefix>.<something>."""

        self._prefix = key_prefix
        self._evaluator = evaluator

    def __getattr__(self, key_suffix):
        """Return the given attribute, reading dataset if needed."""

        dset_key = self._prefix + "." + key_suffix
        if dset_key in self._evaluator.lightcurve.elements["dataset"]:
            result = self._evaluator.lightcurve.read_data(
                dset_key, **self._evaluator.lc_substitutions
            )
            if self._evaluator.lc_points_selection is not None:
                result = result[self._evaluator.lc_points_selection]
        else:
            result = LightCurveLookUp(dset_key, self._evaluator)

        setattr(self, key_suffix, result)
        return result


# pylint: enable=too-few-public-methods


class LightCurveEvaluator(EvaluatorBase):
    """Evaluator for expressions involving lightcurve datasets."""

    def _reset(self):
        for name in self._extra_names:
            self.symtable[name] = LightCurveLookUp(name, self)

    def __init__(
        self, lightcurve, lc_points_selection=None, **lc_substitutions
    ):
        """Get ready to evaluate expressions against the given light curve."""

        super().__init__()
        self.lightcurve = lightcurve
        self._lc_substitutions = lc_substitutions
        self._lc_points_selection = lc_points_selection
        self._extra_names = set(
            element.split(".")[0] for element in lightcurve.elements["dataset"]
        )
        self._reset()

    def update_substitutions(self, new_substitutions):
        """Like update for dict but resets previously read datasets."""

        self._lc_substitutions.update(new_substitutions)
        self._reset()

    @property
    def lc_substitutions(self):
        """
        Substitutions to apply to resolve lightcurve datasets.

        These can be changed after creating the instance.
        """

        return self._lc_substitutions

    @property
    def lc_points_selection(self):
        """Getter for the poins selection property."""

        return self._lc_points_selection

    @lc_points_selection.setter
    def lc_points_selection(self, new_selection):
        """
        Indexing applied to the LC poins to possible exclude some.

        Anything that can be used an index into the numpy arrays that will be
        read from the lightcurve to filter points to use. Just like
        ``lc_substitutions`` the value can be changed post-construction.
        """

        self._lc_points_selection = new_selection
        self._reset()
