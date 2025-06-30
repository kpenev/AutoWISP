# pylint: skip-file
# Generated from /home/kpenev/projects/git/PhotometryPipeline/scripts/FitTermsParser.g4 by ANTLR 4.13.0
from antlr4 import *

if "." in __name__:
    from .FitTermsParser import FitTermsParser
else:
    from FitTermsParser import FitTermsParser


# This class defines a complete listener for a parse tree produced by FitTermsParser.
class FitTermsParserListener(ParseTreeListener):

    # Enter a parse tree produced by FitTermsParser#fit_term.
    def enterFit_term(self, ctx: FitTermsParser.Fit_termContext):
        pass

    # Exit a parse tree produced by FitTermsParser#fit_term.
    def exitFit_term(self, ctx: FitTermsParser.Fit_termContext):
        pass

    # Enter a parse tree produced by FitTermsParser#fit_terms_list.
    def enterFit_terms_list(self, ctx: FitTermsParser.Fit_terms_listContext):
        pass

    # Exit a parse tree produced by FitTermsParser#fit_terms_list.
    def exitFit_terms_list(self, ctx: FitTermsParser.Fit_terms_listContext):
        pass

    # Enter a parse tree produced by FitTermsParser#fit_polynomial.
    def enterFit_polynomial(self, ctx: FitTermsParser.Fit_polynomialContext):
        pass

    # Exit a parse tree produced by FitTermsParser#fit_polynomial.
    def exitFit_polynomial(self, ctx: FitTermsParser.Fit_polynomialContext):
        pass

    # Enter a parse tree produced by FitTermsParser#fit_terms_set.
    def enterFit_terms_set(self, ctx: FitTermsParser.Fit_terms_setContext):
        pass

    # Exit a parse tree produced by FitTermsParser#fit_terms_set.
    def exitFit_terms_set(self, ctx: FitTermsParser.Fit_terms_setContext):
        pass

    # Enter a parse tree produced by FitTermsParser#fit_terms_set_cross_product.
    def enterFit_terms_set_cross_product(
        self, ctx: FitTermsParser.Fit_terms_set_cross_productContext
    ):
        pass

    # Exit a parse tree produced by FitTermsParser#fit_terms_set_cross_product.
    def exitFit_terms_set_cross_product(
        self, ctx: FitTermsParser.Fit_terms_set_cross_productContext
    ):
        pass

    # Enter a parse tree produced by FitTermsParser#fit_terms_expression.
    def enterFit_terms_expression(
        self, ctx: FitTermsParser.Fit_terms_expressionContext
    ):
        pass

    # Exit a parse tree produced by FitTermsParser#fit_terms_expression.
    def exitFit_terms_expression(
        self, ctx: FitTermsParser.Fit_terms_expressionContext
    ):
        pass


del FitTermsParser
