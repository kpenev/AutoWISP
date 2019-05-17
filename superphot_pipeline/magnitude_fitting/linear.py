"""Implement magnitude fitting using linear regression."""

import scipy
import scipy.optimize

from superphot_pipeline.magnitude_fitting.base import MagnitudeFit
from superphot_pipeline.fit_terms import Interface as FitTermsInterface

#Public methods determined by base class.
#pylint: disable=too-few-public-methods
class LinearMagnitudeFit(MagnitudeFit):
    """Differential photometry correction using linear regression."""

    #TODO: switch to linear least squares
    def _group_fit(self, derivatives, mag_difference, weights, phot_ind):
        """
        Fit for the coefficients of a single fitting group.

        Args:
            derivatives:    The derivatives w.r.t. to the fit coefficients of
                the predicted values (i.e. the matrix defining the fitting
                problem).

            mag_difference:    The magnitude difference values wey are trying
                to reproduce.

            weights:    The weight to give to each entry in `mag_difference`.

            phot_ind:    The index of the photometry being fit (only used for
                reporting errors).

        Returns:
            coefficients:     The best fit coefficients.

            fit_res2:    The square residuals of the best fit.
        """

        def rejected_indices(weighted_fit_diff, weights):
            """Return indices of outlier sources and squared fit residual."""

            self.logger.debug('Weigthed difference: %s',
                              repr(weighted_fit_diff))
            self.logger.debug('Weigths: %s', repr(weights))
            fit_diff2 = pow(weighted_fit_diff/weights, 2)
            self.logger.debug('Square difference: %s', repr(fit_diff2))
            if self.config.error_avg == 'weightedmean':
                res2 = (scipy.mean(pow(weighted_fit_diff, 2))
                        /
                        scipy.mean(pow(weights, 2)))
            else:
                avg = getattr(scipy, self.config.error_avg)
                res2 = avg(fit_diff2)
            max_diff2 = self.config.rej_level**2*res2
            self.logger.debug('max square difference: %s', repr(max_diff2))
            if res2 < 0:
                self.logger.debug(
                    '%s',
                    '\n'.join([
                        repr(self._header),
                        '\tNegative square residual: ' + repr(res2),
                        (
                            '\tWeights (min, max): ('
                            +
                            repr(min(weights))
                            +
                            ', '
                            +
                            repr(max(weights))
                            +
                            ')'
                        ),
                        (
                            '\tweighted_fit_diff (min, max): ('
                            +
                            repr(min(weighted_fit_diff))
                            +
                            ', '
                            +
                            repr(max(weighted_fit_diff))
                            +
                            ')'
                        )
                    ])
                )
            return (fit_diff2 > max_diff2).nonzero(), res2

        num_free_coef = len(derivatives)
        error_func = lambda coef: scipy.dot(coef, derivatives) - mag_difference
        deriv_func = lambda coef: derivatives
        initial_guess = scipy.zeros(num_free_coef)
        rej_iter = 0
        while True:
            self.logger.debug('%d sources, %d coefficients\n'
                              %
                              (len(mag_difference), num_free_coef))
            if len(mag_difference) < num_free_coef:
                return None, None, 0
            fit_results = scipy.optimize.leastsq(
                error_func,
                initial_guess,
                Dfun=deriv_func,
                col_deriv=1,
                full_output=1
            )
            if fit_results[4] not in [1, 2, 3, 4]:
                raise RuntimeError("Linear least squares fitting for "
                                   "aperture %d failed for '%s': %s"
                                   %
                                   (phot_ind, self._header, fit_results[3]))
            bad_ind, fit_res2 = rejected_indices(fit_results[2]['fvec'],
                                                 weights)
            rej_iter += 1
            if not bad_ind or rej_iter == self.config.max_rej_iter:
                return fit_results[0], fit_res2, len(mag_difference)
            derivatives = scipy.delete(derivatives, bad_ind, 1)
            mag_difference = scipy.delete(mag_difference, bad_ind)
            weights = scipy.delete(weights, bad_ind)

    def _fit(self, fit_data):

        def get_no_fit_indices(num_photometries):
            """Lists the indices in fit_data to exclude from the fit."""

            finite = True
            for var in ['x', 'y', 'bg', 'bg_err']:
                finite = scipy.logical_and(finite, fit_data[var])

            result = []
            for phot_ind in range(num_photometries):
                finite_phot = finite
                for var in ['mag', 'mag_err', 'ref_mag', 'ref_mag_err']:
                    finite_phot = scipy.logical_and(
                        finite_phot,
                        fit_data[var][:, 0, phot_ind]
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
                result.append(exclude.nonzero())

            return result

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

            phot_predictors = scipy.copy(predictors)
            weights = 1.0 / (fit_data['mag_err'][:, 0, phot_ind]
                             +
                             self.config.noise_offset)
            mag_difference = (fit_data['ref_mag'][:, 0, phot_ind]
                              -
                              fit_data['mag'][:, 0, phot_ind])

            phot_skip_indices = no_fit_indices[phot_ind]
            if phot_skip_indices:
                scipy.delete(phot_predictors, phot_skip_indices)
                scipy.delete(weights, phot_skip_indices)
                scipy.delete(mag_difference, phot_skip_indices)

            derivatives = scipy.multiply(phot_predictors, weights)
            print('Derivatives shape: ' + repr(derivatives.shape))

            group_results = []
            for group_id in fit_group_ids:
                if group_id is not None:
                    in_group = (fit_group == group_id)
                    if phot_skip_indices:
                        scipy.delete(in_group, phot_skip_indices)

                self.logger.debug('Fitting group %s', str(group_id))
                coefficients, fit_res2, final_src_count = self._group_fit(
                    (
                        derivatives if group_id is None
                        else scipy.copy(derivatives[in_group])
                    ),
                    (
                        mag_difference if group_id is None
                        else scipy.copy(mag_difference[in_group])
                    ),
                    (
                        weights if group_id is None
                        else scipy.copy(weights[in_group])
                    ),
                    phot_ind
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
            result.append(group_results)
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

                in_group = (
                    None if group_fit_results['group_id'] is None
                    else (phot['fit_groups'] == group_fit_results['group_id'])
                )
                if in_group is None:
                    fitted[:, phot_ind] = (
                        phot['mag'][:, 0, phot_ind]
                        +
                        scipy.dot(fit_coef, predictors)
                    )
                else:
                    fitted[in_group, phot_ind] = (
                        phot['mag'][:, 0, phot_ind][in_group]
                        +
                        scipy.dot(fit_coef, predictors[:, in_group])
                    )


                predictors = (
                    predictors
                    if in_group is None else
                    [p[in_group] for p in predictors]
                )

        return fitted

    def __init__(self, *, config, **kwargs):

        super().__init__(config=config, **kwargs)
        self.fit_terms = FitTermsInterface(config.correction_parametrization)
#pylint: enable=too-few-public-methods
