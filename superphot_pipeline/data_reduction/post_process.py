"""Add methods for post-processing of data reduction files."""

import scipy
import h5py

from superphot_pipeline.database.hdf5_file_structure import\
    HDF5FileDatabaseStructure
from superphot_pipeline.fit_expression import\
    Interface as FitTermsInterface,\
    iterative_fit
from superphot_pipeline.evaluator import Evaluator

#Out of my control (most ancestors come from h5py module).
#pylint: disable=too-many-ancestors
#This is intended to be an abstract class
#pylint: disable=abstract-method
class DataReductionPostProcess(HDF5FileDatabaseStructure):
    """A collection of methods for post-processing DR files."""

    #Broken up into internal functions.
    #pylint: disable=too-many-locals
    def smooth_srcextract_psf(self,
                              psf_parameters,
                              fit_terms_expression,
                              weights_expression,
                              *,
                              error_avg,
                              rej_level,
                              max_rej_iter,
                              **path_substitutions):
        """
        Fit PSF parameters as polynomials of srcextract and catalogue info.

        Args:
            psf_parameters([str]):    A list of the variables from the source
                extracted datasets to smooth.

            fit_terms_expression(str):    A fitting terms expression defining
                the terms to include in the fit.

            weights_expression(str):    An expression involving source
                extraction and/or catalogue variables for the weights to use for
                the smoothing fit.

            error_avg:    See iterative_fit().

            rej_level:    See iterative_fit().

            max_rej_iter:    See iterative_fit().

            path_substitutions:    Any substitutions required to resolve the
                path to extracted sources, catalogue sources and the
                destinationdatasets and attributes created by this method.

        Returns:
            None
        """

        def get_root_path(pipeline_key):
            """Return the path to the group of columns for the given compon."""

            root_path = (
                self._file_structure[pipeline_key].abspath
                %
                dict(srcextract_column_name='',
                     catalogue_column_name='',
                     **path_substitutions)
            )
            assert root_path[-1] == '/'
            return root_path


        def get_predictors_and_weights(match):
            """Return the matrix of predictors to use for fitting."""

            root = dict(srcextract=get_root_path('srcextract.sources'),
                        catalogue=get_root_path('catalogue.columns'))

            def get_predictor_items(select_indices=(None, None)):
                for indices, component in zip(select_indices,
                                              ['catalogue', 'srcextract']):
                    for var_name, var_dset in self[root[component]].items():
                        if (
                                isinstance(var_dset, h5py.Dataset)
                                and
                                (
                                    component == 'catalogue'
                                    or
                                    var_name not in psf_parameters
                                )
                        ):
                            if indices is None:
                                yield var_name, var_dset
                            else:
                                yield var_name, var_dset[:][indices]

            num_matched_sources = match.shape[0]

            input_data_dtype = []
            for var_name, var_dset in get_predictor_items():
                input_data_dtype.append((var_name, var_dset.dtype))

            input_data = scipy.empty(num_matched_sources,
                                     dtype=input_data_dtype)

            for var_name, var_dset in get_predictor_items(match.transpose()):
                input_data[var_name] = var_dset

            return (FitTermsInterface(fit_terms_expression)(input_data),
                    Evaluator(input_data)(weights_expression))

        def get_psf_param(srcextract_indices):
            """Return a scipy structured array of the PSF parameters."""

            num_matched_sources = srcextract_indices.size
            result = scipy.empty(
                num_matched_sources,
                dtype=[(param, scipy.float64) for param in psf_parameters]
            )
            for param in psf_parameters:
                result[param] = self.get_dataset(
                    'srcextract.sources',
                    srcextract_column_name=param,
                    **path_substitutions
                )[srcextract_indices]
            return result

        def save_fit_results(fit_results):
            """Create the datasets and attributes holding the fit results."""

            self.add_dataset('srcextract.psf_map',
                             scipy.stack(fit_results['coefficients'][param_name]
                                         for param_name in psf_parameters),
                             **path_substitutions)

            for param_key, param_value in [
                    (
                        'cfg.psf_params',
                        scipy.array([name.encode('ascii') for name
                                     in psf_parameters])
                    ),
                    ('cfg.terms', fit_terms_expression.encode('ascii')),
                    ('cfg.weights', weights_expression.encode('ascii')),
                    ('cfg.error_avg', error_avg.encode('ascii')),
                    ('cfg.rej_level', rej_level),
                    ('cfg.max_rej_iter', max_rej_iter),
                    (
                        'residual',
                        scipy.array([
                            fit_results['fit_res2'][param_name]**0.5
                            for param_name in psf_parameters
                        ])
                    ),
                    (
                        'num_fit_src',
                        scipy.array([
                            fit_results['num_fit_src'][param_name]
                            for param_name in psf_parameters
                        ])
                    )
            ]:
                self.add_attribute('srcextract.psf_map.' + param_key,
                                   param_value,
                                   **path_substitutions)


        match = self.get_dataset('skytoframe.matched',
                                 **path_substitutions)
        predictors, weights = get_predictors_and_weights(match)

        fit_results = dict(coefficients=dict(),
                           fit_res2=dict(),
                           num_fit_src=dict())

        psf_param = get_psf_param(match[:, 1])
        for param_name in psf_parameters:
            (
                fit_results['coefficients'][param_name],
                fit_results['fit_res2'][param_name],
                fit_results['num_fit_src'][param_name]
            ) = iterative_fit(
                predictors,
                psf_param[param_name],
                weights=weights,
                error_avg=error_avg,
                rej_level=rej_level,
                max_rej_iter=max_rej_iter,
                fit_identifier='Extracted sources PSF %s map' % param_name
            )

        save_fit_results(fit_results)
    #pylint: enable=too-many-locals
#pylint: enable=too-many-ancestors
#pylint: enable=abstract-method
