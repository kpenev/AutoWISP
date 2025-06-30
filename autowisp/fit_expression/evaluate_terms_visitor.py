"""Implement visitor to parsed fit terms expression that evaluates all terms."""

import numpy

from asteval import asteval

from autowisp.evaluator import Evaluator
from .FitTermsParser import FitTermsParser
from .process_terms_visitor import ProcessTermsVisitor


class EvaluateTermsVisitor(ProcessTermsVisitor):
    """Visitor to parsed fit terms expression evaluating all terms."""

    def _start_expansion(self, num_output_terms, term_length, dtype):
        """Allocate the output array for an expannsion operation."""

        assert self._current_expansion_terms is None
        assert self._expansion_term_index is None
        self._current_expansion_terms = numpy.empty(
            shape=(num_output_terms, term_length), dtype=dtype
        )

    def _process_term_product(self, input_terms, term_powers=None):
        """Add to the current expansion the product of input terms to powers."""

        self._current_expansion_terms[self._expansion_term_index] = 1.0
        for term_index, term in enumerate(input_terms):
            pwr = 1 if term_powers is None else term_powers[term_index]
            if pwr == 0:
                continue
            self._current_expansion_terms[self._expansion_term_index] *= (
                term if pwr == 1 else term**pwr
            )
        self._expansion_term_index += 1

    def _end_expansion(self):
        """Return the result of an expansion and clean-up."""

        result = self._current_expansion_terms
        self._current_expansion_terms = None
        self._expansion_term_index = None
        return result

    def _start_polynomial_expansion(self, num_output_terms, input_terms_list):

        self._start_expansion(
            num_output_terms,
            input_terms_list[0].size,
            input_terms_list[0].dtype,
        )
        self._current_expansion_terms[0] = 1.0
        self._expansion_term_index = 1

    def _process_polynomial_term(self, input_terms, term_powers):

        self._process_term_product(input_terms, term_powers)

    def _end_polynomial_expansion(self):

        return self._end_expansion()

    def _start_cross_product_expansion(self, input_term_sets):

        num_output_terms = 1
        for term_set in input_term_sets:
            num_output_terms *= len(term_set)
        self._start_expansion(
            num_output_terms,
            input_term_sets[0][0].size,
            input_term_sets[0][0].dtype,
        )
        self._expansion_term_index = 0

    def _process_cross_product_term(self, sub_terms):

        self._process_term_product(sub_terms)

    def _end_cross_product_expansion(self):

        return self._end_expansion()

    def __init__(self, *data):
        """
        Define the data used to evaluate the terms.

        Args:
            data(numpy structured array or asteval.Interpreter):    If array,
                should contain fields with all variables required by the terms
                in the expression. If interpreter should have all these
                variables in its a symbol table.

        Returns:
            None
        """

        if len(data) == 1 and isinstance(data[0], asteval.Interpreter):
            self.evaluate_term = data[0]
        else:
            self.evaluate_term = Evaluator(*data)
        self._current_expansion_terms = None
        self._expansion_term_index = None

    # Visit a parse tree produced by FitTermsParser#fit_term.
    def visitFit_term(self, ctx: FitTermsParser.Fit_termContext):
        """Evaluate the corresponding term."""

        return self.evaluate_term(ctx.TERM().getText().strip())

    # Visit a parse tree produced by FitTermsParser#fit_terms_expression.
    def visitFit_terms_expression(
        self, ctx: FitTermsParser.Fit_terms_expressionContext
    ):
        """Return all terms defined by the term expression."""

        term_list = []
        for child in ctx.fit_terms_set_cross_product():
            new_terms = self.visit(child)
            if term_list and (new_terms[0] == 1).all():
                term_list.append(new_terms[1:])
            else:
                term_list.append(new_terms)

        return numpy.concatenate(term_list)
