"""Implement magnitude fitting using linear regression."""

import scipy
import scipy.optimize

from superphot_pipeline.magnitude_fitting.base import MagnitudeFit
from superphot_pipeline.fit_expression import Interface as FitTermsInterface
from superphot_pipeline.fit_expression import iterative_fit

#Public methods determined by base class.
#pylint: disable=too-few-public-methods
class LinearMagnitudeFit(MagnitudeFit):
    """Differential photometry correction using linear regression."""

    def _fit(self, fit_data):

        def get_no_fit_indices(num_photometries):
            """Lists the indices in fit_data to exclude from the fit."""

            finite = True
            for var in ['x', 'y', 'bg', 'bg_err']:
                finite = scipy.logical_and(finite,
                                           scipy.isfinite(fit_data[var]))

            result = []
            for phot_ind in range(num_photometries):
                finite_phot = finite
                for var in ['mag', 'mag_err', 'ref_mag', 'ref_mag_err']:
                    finite_phot = scipy.logical_and(
                        finite_phot,
                        scipy.isfinite(fit_data[var][:, 0, phot_ind])
                    )
                exclude = scipy.logical_not(finite_phot)
                exclude = scipy.logical_or(
                    exclude,
                    (
                        fit_data['mag_err'][:, 0, phot_ind]
                        >
                        self.config.max_mag_err
                    )
                )
                result.append(exclude.nonzero()[0])

            return result

        def calculate_photometry_result(phot_ind,
                                        phot_predictors,
                                        no_fit_indices,
                                        fit_group,
                                        fit_group_ids):
            """Calculate and return result entry for a single photometry."""

            weights = 1.0 / (fit_data['mag_err'][:, 0, phot_ind]
                             +
                             self.config.noise_offset)
            mag_difference = (fit_data['ref_mag'][:, 0, phot_ind]
                              -
                              fit_data['mag'][:, 0, phot_ind])

            phot_skip_indices = no_fit_indices[phot_ind]
            if phot_skip_indices.size:
                self.logger.debug('Skipping %d sources from fitting.',
                                  phot_skip_indices.size)
                phot_predictors = scipy.delete(phot_predictors,
                                               phot_skip_indices,
                                               1)
                weights = scipy.delete(weights, phot_skip_indices)
                mag_difference = scipy.delete(mag_difference, phot_skip_indices)

            self.logger.debug('Smallest weight for photometry %d: %g',
                              phot_ind,
                              weights.min())

            assert scipy.isfinite(phot_predictors).all()
            assert scipy.isfinite(mag_difference).all()
            assert (weights > 0).all()

            group_results = []
            for group_id in fit_group_ids:
                if group_id is not None:
                    in_group = (fit_group == group_id)
                    if phot_skip_indices.size:
                        in_group = scipy.delete(in_group, phot_skip_indices)

                self.logger.debug('Fitting group %s', str(group_id))
                coefficients, fit_res2, final_src_count = iterative_fit(
                    (
                        phot_predictors if group_id is None
                        else phot_predictors[:, in_group]
                    ),
                    (
                        mag_difference if group_id is None
                        else scipy.copy(mag_difference[in_group])
                    ),
                    weights=(
                        weights if group_id is None
                        else scipy.copy(weights[in_group])
                    ),
                    error_avg=self.config.error_avg,
                    rej_level=self.config.rej_level,
                    max_rej_iter=self.config.max_rej_iter,
                    fit_identifier=('%s, photometry #%d'
                                    %
                                    (self._dr_fname, phot_ind))
                )

                if coefficients is None:
                    self.logger.error(
                        'Rejection resulted in fewer sources than '
                        'parameters when fitting group %d\n',
                        group_id
                    )

                fit_residual = (None if fit_res2 is None
                                else scipy.sqrt(fit_res2))

                group_results.append(dict(coefficients=coefficients,
                                          residual=fit_residual,
                                          initial_src_count=len(predictors[0]),
                                          final_src_count=final_src_count,
                                          group_id=group_id))
            return group_results

        assert fit_data['mag'].shape[1] == 1
        num_photometries = fit_data['mag'].shape[2]

        no_fit_indices = get_no_fit_indices(num_photometries)
        result = []
        fit_group = (fit_data['fit_group']
                     if 'fit_group' in fit_data.dtype.names else
                     [None])
        fit_group_ids = scipy.unique(fit_group)
        predictors = self.fit_terms(fit_data)
        print('Predictors shape: ' + repr(predictors.shape))
        for phot_ind in range(num_photometries):
            result.append(
                calculate_photometry_result(
                    phot_ind,
                    predictors,
                    no_fit_indices,
                    fit_group,
                    fit_group_ids
                )
            )
        return result

    def _apply_fit(self, phot, fit_results):

        assert len(fit_results) == phot['mag'].shape[2]
        fitted = scipy.full((phot['mag'].shape[0], phot['mag'].shape[2]),
                            scipy.nan)
        predictors = self.fit_terms(phot)
        for phot_ind, phot_fit_results in enumerate(fit_results):

            for group_fit_results in phot_fit_results:
                fit_coef = group_fit_results['coefficients']
                if fit_coef is None:
                    continue

                if group_fit_results['group_id'] is None:
                    fitted[:, phot_ind] = (
                        phot['mag'][:, 0, phot_ind]
                        +
                        scipy.dot(fit_coef, predictors)
                    )
                else:
                    in_group = (phot['fit_group']
                                ==
                                group_fit_results['group_id'])
                    fitted[in_group, phot_ind] = (
                        phot['mag'][in_group, 0, phot_ind]
                        +
                        scipy.dot(fit_coef, predictors[:, in_group])
                    )

        return fitted

    def __init__(self, *, config, **kwargs):
        """
        Initialize a magnitude fitting object using linear least squares.

        Args:
            config:    An object with attributes configuring how to perform
                magnitude fitting. It should provide at least the arguments
                required by the parent class and the following:

                    * correction_parametrization: As string that expands to the
                      terms to include in the magnitude fitting correction.

                    * max_mag_err: The largest the formal magnitude error is
                      allowed to be before the source is excluded.

                    * noise_offset: Additional offset to format magnitude error
                      estimates when they are used to determine the fitting
                      weights.

                    * error_avg: See same name argument to
                      superphot_pipeline.fit_expression.iterative_fit().

                    * rej_level: See same name argument to
                      superphot_pipeline.fit_expression.iterative_fit().

                    * max_rej_iter: See same name argument to
                      superphot_pipeline.fit_expression.iterative_fit().

        Returns:
            None
        """


        super().__init__(config=config, **kwargs)
        self.fit_terms = FitTermsInterface(config.correction_parametrization)
#pylint: enable=too-few-public-methods
