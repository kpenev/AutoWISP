#!/usr/bin/env python3
import sys

import scipy

from antlr4 import FileStream, CommonTokenStream
from superphot_pipeline.fit_terms.FitTermsLexer import FitTermsLexer
from superphot_pipeline.fit_terms.FitTermsParser import FitTermsParser
from superphot_pipeline.fit_terms import \
    ListTermsVisitor,\
    CountTermsVisitor,\
    EvaluateTermsVisitor

if __name__ == '__main__':
    lexer = FitTermsLexer(FileStream(sys.argv[1]))
    stream = CommonTokenStream(lexer)
    parser = FitTermsParser(stream)
    tree = parser.fit_terms_expression()

    print('Number terms: ' + repr(CountTermsVisitor().visit(tree)))

    term_str_list = ListTermsVisitor().visit(tree)
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
    evaluated = EvaluateTermsVisitor(variables).visit(tree)

    print('Values (%d x %d):' % evaluated.shape)
    for term_str, term_values in zip(term_str_list, evaluated):
        print('\t%30.30s ' % term_str + repr(term_values))
    print()
