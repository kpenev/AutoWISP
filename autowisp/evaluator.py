"""More convenient interface to estaver interpreters."""

from os import path

import numpy
from asteval import asteval
from astropy.io import fits
from astropy import units
import pandas

try:
    from autowisp.fits_utilities import get_primary_header
    from autowisp.data_reduction import DataReductionFile
except:
    pass


class Evaluator(asteval.Interpreter):
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

    def __call__(self, *args, **kwargs):
        """Evaluate the expression enabling error."""

        if "raise_errors" not in kwargs:
            kwargs["raise_errors"] = True
        return super().__call__(*args, **kwargs)


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


class LightCurveEvaluator(asteval.Interpreter):
    """
    Evaluator for expressions involving lightcurve datasets.

    Attributes:
        lc_substitutions(dict):    Any substitutions that must be applied to
            resolve lightcurve datasets. These can be changed after creating the
            instance.

        lc_points_selection(numpy array index):    Anything that can be used an
            index into the numpy arrays that will be read from the lightcurve to
            filter points to use. Just like ``lc_substitutions`` the value can
            be changed post-construction.
    """

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
        self.symtable["nanmean"] = numpy.nanmean
        self.symtable["nanmedian"] = numpy.nanmedian
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
        """Make lc_substitutions read only."""

        return self._lc_substitutions

    @property
    def lc_points_selection(self):
        """Getter for the poins selection property."""

        return self._lc_points_selection

    @lc_points_selection.setter
    def lc_points_selection(self, new_selection):
        """Setter for the points selection property."""

        self._lc_points_selection = new_selection
        self._reset()
