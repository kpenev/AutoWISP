#!/usr/bin/env python3
import sys

import scipy

from antlr4 import InputStream, CommonTokenStream
from superphot_pipeline.fit_expression.FitTermsLexer import FitTermsLexer
from superphot_pipeline.fit_expression.FitTermsParser import FitTermsParser
from superphot_pipeline.fit_expression import Interface as FitTermsInterface

if __name__ == '__main__':
    fit_terms = FitTermsInterface(sys.argv[1])

    print('Number terms: ' + repr(fit_terms.number_terms()))

    term_str_list = fit_terms.get_term_str_list()
    print(
        'Terms:\n\t'
        +
        '\n\t'.join(
            [str(i) + ': ' + repr(t) for i, t in enumerate(term_str_list)]
        )
    )

    variables = scipy.empty(
        shape=3,
        dtype=([(field, scipy.float64) for field in 'abcdewxyz'])
    )
    for var, value in zip('abcdewxyz',
                          [2, 3, 5, 7, 11, 13, 16, 19,  23]):
        variables[var] = value
    evaluated = fit_terms(variables)

    print('Values (%d x %d):' % evaluated.shape)
    for term_str, term_values in zip(term_str_list, evaluated):
        print('\t%30.30s ' % term_str + repr(term_values))
    print()
