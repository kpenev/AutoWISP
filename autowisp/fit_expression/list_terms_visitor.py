"""Implement a visitor to parsed fit terms expressions that prints all terms."""

from autowisp.fit_expression.FitTermsParser import FitTermsParser
from autowisp.fit_expression.process_terms_visitor import ProcessTermsVisitor


class ListTermsVisitor(ProcessTermsVisitor):
    """Visitor to parsed fit terms expressions listing all terms."""

    def _end_expansion(self):
        """The final step of performing a term set expansion operation."""

        result = self._current_expansion_terms
        self._current_expansion_terms = None
        return result

    def _start_polynomial_expansion(self, num_output_terms, input_terms_list):

        assert self._current_expansion_terms is None
        self._current_expansion_terms = ["1"]

    def _process_polynomial_term(self, input_terms, term_powers):
        """Add a human readable, yet evaluatable string of the term."""

        term_factors = []
        for term, power in zip(input_terms, term_powers):
            if power == 1:
                term_factors.append(term)
            elif power > 1:
                term_factors.append(f"{term!s}**{power:d}")
        self._current_expansion_terms.append(" * ".join(term_factors))

    def _end_polynomial_expansion(self):

        return self._end_expansion()

    def _start_cross_product_expansion(self, input_term_sets):

        assert self._current_expansion_terms is None
        self._current_expansion_terms = []

    def _process_cross_product_term(self, sub_terms):

        def format_term_in_product(term):
            """Format the given term suitably for including in a product."""

            if term == "1":
                return None
            return term

        term = " * ".join(filter(None, map(format_term_in_product, sub_terms)))
        self._current_expansion_terms.append(term or "1")

    def _end_cross_product_expansion(self):
        return self._end_expansion()

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self._current_expansion_terms = None

    # Visit a parse tree produced by FitTermsParser#fit_term.
    def visitFit_term(self, ctx: FitTermsParser.Fit_termContext):
        """Return a string representaiton of the corresponding term."""

        return "(" + ctx.TERM().getText().strip() + ")"

    # Visit a parse tree produced by FitTermsParser#fit_terms_expression.
    def visitFit_terms_expression(
        self, ctx: FitTermsParser.Fit_terms_expressionContext
    ):
        """Return all terms defined by the term expression."""

        result = []
        for child in ctx.fit_terms_set_cross_product():
            new_terms = self.visit(child)
            if result and (new_terms[0] == "1"):
                result.extend(new_terms[1:])
            else:
                result.extend(new_terms)

        return result
