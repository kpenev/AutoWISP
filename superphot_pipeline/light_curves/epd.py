#!/usr/bin/env python3

"""Define class for performing EPD correction on lightcurves."""

import scipy

from superphot_pipeline import LightCurveFile
from superphot_pipeline.evaluator import Evaluator
from superphot_pipeline.fit_expression import\
    Interface as FitTermsInterface,\
    iterative_fit
from superphot_pipeline.light_curves.lc_data_io import _config_dset_key_rex

class EPDCorrection:
    """
    Class for deriving and applying EPD corrections to lightcurves.

    Attributes:
        fit_points_filter_expression:    See __init__.

        fit_terms_expression:    See __init__.

        fit_datasets:    See __init__.

        iterative_fit_config(dict):    Configuration to use for iterative
            fitting. See iterative_fit() for details.

        _config_indices(dict):    A dictionary of the already read-in
            configuration indices (re-used if requested again).
    """

    def _get_config_indices(self,
                            dataset_key,
                            light_curve,
                            **substitutions):
        """Return the config index dset for indexing the given config dset."""

        substitution_key = frozenset(substitutions.items())
        config_component = dataset_key
        while True:
            try:
                result = self._config_indices.get(config_component)
                if result is not None:
                    result = result.get(substitution_key)
                if result is None:
                    result = light_curve.get_dataset(
                        config_component + '.cfg_index',
                        **substitutions
                    )
                    if config_component not in self._config_indices:
                        self._config_indices[config_component] = dict()
                    self._config_indices[config_component][substitution_key] = (
                        result
                    )

                return result
            except KeyError:
                config_component = config_component.rsplit('.', 1)[0]

    def _get_independent_variables(self, light_curve):
        """Return scipy structured array of the independent variables."""

        def result_column_dtype(dset_key):
            """The type to use for the given column in the result."""

            result = light_curve.get_dtype(dset_key)
            if result == scipy.string_:
                return scipy.dtype('O')
            return result

        def create_empty_result(result_size):
            """Create an uninitialized dasates to hold the result."""

            return scipy.empty(
                result_size,
                dtype=[
                    (vname, result_column_dtype(dset_key))
                    for vname, (dset_key, subs) in self.used_variables.items()
                ]
            )

        result = None
        for var_name, (dataset_key,
                       substitutions) in self.used_variables.items():
            data = light_curve.get_dataset(dataset_key, **substitutions)

            if _config_dset_key_rex.search(dataset_key):
                print('Converting configuration: ' + repr(data))
                config_indices = self._get_config_indices(
                    dataset_key,
                    light_curve,
                    **substitutions
                )
                data = data[config_indices]
                print('To: ' + repr(data))

            if result is None:
                result = create_empty_result(data.size)
            else:
                assert data.shape == result.shape
            result[var_name] = data

        return result

    def __init__(self,
                 *,
                 used_variables,
                 fit_points_filter_expression,
                 fit_terms_expression,
                 fit_datasets,
                 fit_weights=None,
                 **iterative_fit_config):
        """
        Configure the fitting.

        Args:
            used_variables(dict):    Keys are variables used in
                `fit_points_filter_expression` and `fit_terms_expression` and
                the corresponding values are 2-tuples of pipeline keys
                corresponding to each variable and an associated dictionary of
                path substitutions. Each entry defines a unique independent
                variable to use in the fit or based on which to select points to
                fit.

            fit_points_filter_expression(str):    An expression using
                `used_variables` which evalutes to either True or False
                indicating if a given point in the lightcurve should be fit and
                corrected.

            fit_terms_expression(str):    A fitting terms expression involving
                only variables from `used_variables` which expands to the
                various terms to use in a linear least squares EPD correction.

            fit_datasets([]):    A list of 2-tuples of pipeline keys
                corresponding to each variable and an associated dictionary of
                path substitutions identifying datasets to fit and correct.

            fit_weights([]):    Weights to use when fitting each fit_dataset.
                Should have exactly the same structure and shape as
                fit_datasets. If None all points get equal weight.

            iterative_fit_config:    Any other arguments to pass directly to
                iterative_fit().

        Returns:
            None
        """

        self.used_variables = used_variables
        self.fit_points_filter_expression = fit_points_filter_expression
        self.fit_terms = FitTermsInterface(fit_terms_expression)
        self.fit_datasets = fit_datasets
        self.fit_weights = fit_weights
        self.iterative_fit_config = iterative_fit_config

        self._config_indices = dict()

    def __call__(self, lc_fname):
        """Fit and correct the given lightcurve."""

        with LightCurveFile(lc_fname, 'r') as light_curve:
            indep_variables = self._get_independent_variables(light_curve)
            print('Indep vars (shape: %s) = %s' % (repr(indep_variables.shape),
                                                   repr(indep_variables)))

            evaluate = Evaluator(indep_variables)

            predictors = self.fit_terms(evaluate)

            fit_points = (
                evaluate(self.fit_points_filter_expression)
                if self.fit_points_filter_expression is not None else
                scipy.ones(predictors.shape[1], dtype=bool)
            )
            predictors = predictors[:, fit_points]

            assert (self.fit_weights is None
                    or
                    len(self.fit_datasets) == len(self.fit_weights))

            for to_fit in (self.fit_datasets if self.fit_weights is None
                           else zip(self.fit_datasets, fit_weights)):

                if self.fit_weights is None:
                    fit_data, substitutions = to_fit
                    fit_weights = None
                else:
                    (fit_data, substitutions), fit_weights = to_fit
                    fit_weights = light_curve.get_dataset(
                        fit_weights,
                        **substitutions
                    )[fit_points]
                fit_data = light_curve.get_dataset(fit_data,
                                                   **substitutions)[fit_points]


                #Those should come from self.iteritave_fit_config.
                #pylint: disable=missing-kwoa
                (
                    best_fit_coef,
                    square_residuals,
                    non_rejected_points
                ) = iterative_fit(
                    predictors=predictors,
                    target_values=fit_data,
                    weights=fit_weights,
                    **self.iterative_fit_config
                )
                #pylint: enable=missing-kwoa
                assert best_fit_coef is not None

                corrected = fit_data - scipy.dot(best_fit_coef,
                                                 predictors)
                print('Corrected ' + repr(to_fit) + ': ' + repr(corrected))

        return indep_variables

if __name__ == '__main__':
    correct = EPDCorrection(
        used_variables=dict(
            x=('srcproj.x', dict()),
            y=('srcproj.y', dict()),
            fistarS=('srcextract.psf_map.eval',
                     dict(srcextract_psf_param='S')),
            fistarD=('srcextract.psf_map.eval',
                     dict(srcextract_psf_param='D')),
            fistarK=('srcextract.psf_map.eval',
                     dict(srcextract_psf_param='K')),
            num_fit_src_s=('srcextract.psf_map.num_fit_src',
                           dict(srcextract_psf_param='S')),
            num_fit_src_d=('srcextract.psf_map.num_fit_src',
                           dict(srcextract_psf_param='D')),
            num_fit_src_k=('srcextract.psf_map.num_fit_src',
                           dict(srcextract_psf_param='K')),
            residual_s=('srcextract.psf_map.residual',
                        dict(srcextract_psf_param='S')),
            residual_d=('srcextract.psf_map.residual',
                        dict(srcextract_psf_param='D')),
            residual_k=('srcextract.psf_map.residual',
                        dict(srcextract_psf_param='K')),
            fnum=('fitsheader.fnum', dict()),
            channel=('fitsheader.cfg.color', dict())
        ),
        fit_points_filter_expression=None,
        fit_terms_expression='O2{x}',
        fit_datasets=[('srcproj.y', dict()), ('srcproj.x', dict())],
        error_avg='nanmedian',
        rej_level=5,
        max_rej_iter=20,
        fit_identifier='position vs x',
    )
    print(
        repr(
            correct('usage_examples/test_data/10-20170306/lcs/0-180-3618.hdf5')
        )
    )
