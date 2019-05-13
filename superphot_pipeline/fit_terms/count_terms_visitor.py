"""Implement a visitor to parsed fit terms expressions that counts the terms."""

from scipy.special import binom

from superphot_pipeline.fit_terms.FitTermsParser import FitTermsParser
from superphot_pipeline.fit_terms.FitTermsParserVisitor import \
    FitTermsParserVisitor

class CountTermsVisitor(FitTermsParserVisitor):
    """Visitor to parsed fit terms expressions counting the expanded terms."""

    # Visit a parse tree produced by FitTermsParser#fit_terms_list.
    def visitFit_terms_list(self, ctx: FitTermsParser.Fit_terms_listContext):
        """Return the number of terms in the list."""

        return len(ctx.fit_term())

    # Visit a parse tree produced by FitTermsParser#fit_polynomial.
    def visitFit_polynomial(self, ctx: FitTermsParser.Fit_polynomialContext):
        """Return the number of terms after the polynomial expansion."""

        max_order = int(ctx.order.text)
        num_terms = self.visit(ctx.fit_terms_list())
        return int(binom(num_terms + max_order, max_order))

    # Visit a parse tree produced by FitTermsParser#fit_terms_set.
    def visitFit_terms_set(self, ctx: FitTermsParser.Fit_terms_setContext):
        """Return the number of the terms in the set."""

        return self.visit(ctx.getChild(0))

    # Visit a parse tree produced by FitTermsParser#fit_terms_set_cross_product.
    def visitFit_terms_set_cross_product(
            self,
            ctx: FitTermsParser.Fit_terms_set_cross_productContext
    ):
        """Return the number of terms combining one term from each input set."""

        result = 1
        for term_set in ctx.fit_terms_set():
            result *= self.visit(term_set)

        return result

    def visitFit_terms_expression(
            self,
            ctx: FitTermsParser.Fit_terms_expressionContext
    ):
        """Return the total number of terms the term expression expands to."""

        result = 0
        for child in ctx.fit_terms_set_cross_product():
            result += self.visit(child)

        return result
