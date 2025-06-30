# pylint: skip-file
# Generated from /home/kpenev/projects/git/PhotometryPipeline/scripts/FitTermsParser.g4 by ANTLR 4.13.0
from antlr4 import *

if "." in __name__:
    from .FitTermsParser import FitTermsParser
else:
    from FitTermsParser import FitTermsParser

# This class defines a complete generic visitor for a parse tree produced by FitTermsParser.


class FitTermsParserVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by FitTermsParser#fit_term.
    def visitFit_term(self, ctx: FitTermsParser.Fit_termContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by FitTermsParser#fit_terms_list.
    def visitFit_terms_list(self, ctx: FitTermsParser.Fit_terms_listContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by FitTermsParser#fit_polynomial.
    def visitFit_polynomial(self, ctx: FitTermsParser.Fit_polynomialContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by FitTermsParser#fit_terms_set.
    def visitFit_terms_set(self, ctx: FitTermsParser.Fit_terms_setContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by FitTermsParser#fit_terms_set_cross_product.
    def visitFit_terms_set_cross_product(
        self, ctx: FitTermsParser.Fit_terms_set_cross_productContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by FitTermsParser#fit_terms_expression.
    def visitFit_terms_expression(
        self, ctx: FitTermsParser.Fit_terms_expressionContext
    ):
        return self.visitChildren(ctx)


del FitTermsParser
