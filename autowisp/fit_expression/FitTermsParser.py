# pylint: skip-file
# Generated from /home/kpenev/projects/git/PhotometryPipeline/scripts/FitTermsParser.g4 by ANTLR 4.13.0
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys

if sys.version_info[1] > 5:
    from typing import TextIO
else:
    from typing.io import TextIO


def serializedATN():
    return [
        4,
        1,
        9,
        50,
        2,
        0,
        7,
        0,
        2,
        1,
        7,
        1,
        2,
        2,
        7,
        2,
        2,
        3,
        7,
        3,
        2,
        4,
        7,
        4,
        2,
        5,
        7,
        5,
        1,
        0,
        1,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        5,
        1,
        19,
        8,
        1,
        10,
        1,
        12,
        1,
        22,
        9,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        1,
        2,
        1,
        2,
        1,
        2,
        1,
        3,
        1,
        3,
        3,
        3,
        32,
        8,
        3,
        1,
        4,
        1,
        4,
        1,
        4,
        5,
        4,
        37,
        8,
        4,
        10,
        4,
        12,
        4,
        40,
        9,
        4,
        1,
        5,
        1,
        5,
        1,
        5,
        5,
        5,
        45,
        8,
        5,
        10,
        5,
        12,
        5,
        48,
        9,
        5,
        1,
        5,
        0,
        0,
        6,
        0,
        2,
        4,
        6,
        8,
        10,
        0,
        0,
        47,
        0,
        12,
        1,
        0,
        0,
        0,
        2,
        14,
        1,
        0,
        0,
        0,
        4,
        25,
        1,
        0,
        0,
        0,
        6,
        31,
        1,
        0,
        0,
        0,
        8,
        33,
        1,
        0,
        0,
        0,
        10,
        41,
        1,
        0,
        0,
        0,
        12,
        13,
        5,
        9,
        0,
        0,
        13,
        1,
        1,
        0,
        0,
        0,
        14,
        15,
        5,
        1,
        0,
        0,
        15,
        20,
        3,
        0,
        0,
        0,
        16,
        17,
        5,
        8,
        0,
        0,
        17,
        19,
        3,
        0,
        0,
        0,
        18,
        16,
        1,
        0,
        0,
        0,
        19,
        22,
        1,
        0,
        0,
        0,
        20,
        18,
        1,
        0,
        0,
        0,
        20,
        21,
        1,
        0,
        0,
        0,
        21,
        23,
        1,
        0,
        0,
        0,
        22,
        20,
        1,
        0,
        0,
        0,
        23,
        24,
        5,
        7,
        0,
        0,
        24,
        3,
        1,
        0,
        0,
        0,
        25,
        26,
        5,
        3,
        0,
        0,
        26,
        27,
        5,
        2,
        0,
        0,
        27,
        28,
        3,
        2,
        1,
        0,
        28,
        5,
        1,
        0,
        0,
        0,
        29,
        32,
        3,
        2,
        1,
        0,
        30,
        32,
        3,
        4,
        2,
        0,
        31,
        29,
        1,
        0,
        0,
        0,
        31,
        30,
        1,
        0,
        0,
        0,
        32,
        7,
        1,
        0,
        0,
        0,
        33,
        38,
        3,
        6,
        3,
        0,
        34,
        35,
        5,
        4,
        0,
        0,
        35,
        37,
        3,
        6,
        3,
        0,
        36,
        34,
        1,
        0,
        0,
        0,
        37,
        40,
        1,
        0,
        0,
        0,
        38,
        36,
        1,
        0,
        0,
        0,
        38,
        39,
        1,
        0,
        0,
        0,
        39,
        9,
        1,
        0,
        0,
        0,
        40,
        38,
        1,
        0,
        0,
        0,
        41,
        46,
        3,
        8,
        4,
        0,
        42,
        43,
        5,
        5,
        0,
        0,
        43,
        45,
        3,
        8,
        4,
        0,
        44,
        42,
        1,
        0,
        0,
        0,
        45,
        48,
        1,
        0,
        0,
        0,
        46,
        44,
        1,
        0,
        0,
        0,
        46,
        47,
        1,
        0,
        0,
        0,
        47,
        11,
        1,
        0,
        0,
        0,
        48,
        46,
        1,
        0,
        0,
        0,
        4,
        20,
        31,
        38,
        46,
    ]


class FitTermsParser(Parser):

    grammarFileName = "FitTermsParser.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [DFA(ds, i) for i, ds in enumerate(atn.decisionToState)]

    sharedContextCache = PredictionContextCache()

    literalNames = [
        "<INVALID>",
        "'{'",
        "<INVALID>",
        "'O'",
        "'*'",
        "'+'",
        "<INVALID>",
        "'}'",
        "','",
    ]

    symbolicNames = [
        "<INVALID>",
        "TERM_LIST_START",
        "UINT",
        "POLY_START",
        "CROSSPRODUCT",
        "UNION",
        "WS",
        "TERM_LIST_END",
        "TERM_SEP",
        "TERM",
    ]

    RULE_fit_term = 0
    RULE_fit_terms_list = 1
    RULE_fit_polynomial = 2
    RULE_fit_terms_set = 3
    RULE_fit_terms_set_cross_product = 4
    RULE_fit_terms_expression = 5

    ruleNames = [
        "fit_term",
        "fit_terms_list",
        "fit_polynomial",
        "fit_terms_set",
        "fit_terms_set_cross_product",
        "fit_terms_expression",
    ]

    EOF = Token.EOF
    TERM_LIST_START = 1
    UINT = 2
    POLY_START = 3
    CROSSPRODUCT = 4
    UNION = 5
    WS = 6
    TERM_LIST_END = 7
    TERM_SEP = 8
    TERM = 9

    def __init__(self, input: TokenStream, output: TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.13.0")
        self._interp = ParserATNSimulator(
            self, self.atn, self.decisionsToDFA, self.sharedContextCache
        )
        self._predicates = None

    class Fit_termContext(ParserRuleContext):
        __slots__ = "parser"

        def __init__(
            self,
            parser,
            parent: ParserRuleContext = None,
            invokingState: int = -1,
        ):
            super().__init__(parent, invokingState)
            self.parser = parser

        def TERM(self):
            return self.getToken(FitTermsParser.TERM, 0)

        def getRuleIndex(self):
            return FitTermsParser.RULE_fit_term

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterFit_term"):
                listener.enterFit_term(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitFit_term"):
                listener.exitFit_term(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitFit_term"):
                return visitor.visitFit_term(self)
            else:
                return visitor.visitChildren(self)

    def fit_term(self):

        localctx = FitTermsParser.Fit_termContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_fit_term)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 12
            self.match(FitTermsParser.TERM)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class Fit_terms_listContext(ParserRuleContext):
        __slots__ = "parser"

        def __init__(
            self,
            parser,
            parent: ParserRuleContext = None,
            invokingState: int = -1,
        ):
            super().__init__(parent, invokingState)
            self.parser = parser

        def TERM_LIST_START(self):
            return self.getToken(FitTermsParser.TERM_LIST_START, 0)

        def fit_term(self, i: int = None):
            if i is None:
                return self.getTypedRuleContexts(FitTermsParser.Fit_termContext)
            else:
                return self.getTypedRuleContext(
                    FitTermsParser.Fit_termContext, i
                )

        def TERM_LIST_END(self):
            return self.getToken(FitTermsParser.TERM_LIST_END, 0)

        def TERM_SEP(self, i: int = None):
            if i is None:
                return self.getTokens(FitTermsParser.TERM_SEP)
            else:
                return self.getToken(FitTermsParser.TERM_SEP, i)

        def getRuleIndex(self):
            return FitTermsParser.RULE_fit_terms_list

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterFit_terms_list"):
                listener.enterFit_terms_list(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitFit_terms_list"):
                listener.exitFit_terms_list(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitFit_terms_list"):
                return visitor.visitFit_terms_list(self)
            else:
                return visitor.visitChildren(self)

    def fit_terms_list(self):

        localctx = FitTermsParser.Fit_terms_listContext(
            self, self._ctx, self.state
        )
        self.enterRule(localctx, 2, self.RULE_fit_terms_list)
        self._la = 0  # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 14
            self.match(FitTermsParser.TERM_LIST_START)
            self.state = 15
            self.fit_term()
            self.state = 20
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 8:
                self.state = 16
                self.match(FitTermsParser.TERM_SEP)
                self.state = 17
                self.fit_term()
                self.state = 22
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 23
            self.match(FitTermsParser.TERM_LIST_END)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class Fit_polynomialContext(ParserRuleContext):
        __slots__ = "parser"

        def __init__(
            self,
            parser,
            parent: ParserRuleContext = None,
            invokingState: int = -1,
        ):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.order = None  # Token

        def POLY_START(self):
            return self.getToken(FitTermsParser.POLY_START, 0)

        def fit_terms_list(self):
            return self.getTypedRuleContext(
                FitTermsParser.Fit_terms_listContext, 0
            )

        def UINT(self):
            return self.getToken(FitTermsParser.UINT, 0)

        def getRuleIndex(self):
            return FitTermsParser.RULE_fit_polynomial

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterFit_polynomial"):
                listener.enterFit_polynomial(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitFit_polynomial"):
                listener.exitFit_polynomial(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitFit_polynomial"):
                return visitor.visitFit_polynomial(self)
            else:
                return visitor.visitChildren(self)

    def fit_polynomial(self):

        localctx = FitTermsParser.Fit_polynomialContext(
            self, self._ctx, self.state
        )
        self.enterRule(localctx, 4, self.RULE_fit_polynomial)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 25
            self.match(FitTermsParser.POLY_START)
            self.state = 26
            localctx.order = self.match(FitTermsParser.UINT)
            self.state = 27
            self.fit_terms_list()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class Fit_terms_setContext(ParserRuleContext):
        __slots__ = "parser"

        def __init__(
            self,
            parser,
            parent: ParserRuleContext = None,
            invokingState: int = -1,
        ):
            super().__init__(parent, invokingState)
            self.parser = parser

        def fit_terms_list(self):
            return self.getTypedRuleContext(
                FitTermsParser.Fit_terms_listContext, 0
            )

        def fit_polynomial(self):
            return self.getTypedRuleContext(
                FitTermsParser.Fit_polynomialContext, 0
            )

        def getRuleIndex(self):
            return FitTermsParser.RULE_fit_terms_set

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterFit_terms_set"):
                listener.enterFit_terms_set(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitFit_terms_set"):
                listener.exitFit_terms_set(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitFit_terms_set"):
                return visitor.visitFit_terms_set(self)
            else:
                return visitor.visitChildren(self)

    def fit_terms_set(self):

        localctx = FitTermsParser.Fit_terms_setContext(
            self, self._ctx, self.state
        )
        self.enterRule(localctx, 6, self.RULE_fit_terms_set)
        try:
            self.state = 31
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [1]:
                self.enterOuterAlt(localctx, 1)
                self.state = 29
                self.fit_terms_list()
                pass
            elif token in [3]:
                self.enterOuterAlt(localctx, 2)
                self.state = 30
                self.fit_polynomial()
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class Fit_terms_set_cross_productContext(ParserRuleContext):
        __slots__ = "parser"

        def __init__(
            self,
            parser,
            parent: ParserRuleContext = None,
            invokingState: int = -1,
        ):
            super().__init__(parent, invokingState)
            self.parser = parser

        def fit_terms_set(self, i: int = None):
            if i is None:
                return self.getTypedRuleContexts(
                    FitTermsParser.Fit_terms_setContext
                )
            else:
                return self.getTypedRuleContext(
                    FitTermsParser.Fit_terms_setContext, i
                )

        def CROSSPRODUCT(self, i: int = None):
            if i is None:
                return self.getTokens(FitTermsParser.CROSSPRODUCT)
            else:
                return self.getToken(FitTermsParser.CROSSPRODUCT, i)

        def getRuleIndex(self):
            return FitTermsParser.RULE_fit_terms_set_cross_product

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterFit_terms_set_cross_product"):
                listener.enterFit_terms_set_cross_product(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitFit_terms_set_cross_product"):
                listener.exitFit_terms_set_cross_product(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitFit_terms_set_cross_product"):
                return visitor.visitFit_terms_set_cross_product(self)
            else:
                return visitor.visitChildren(self)

    def fit_terms_set_cross_product(self):

        localctx = FitTermsParser.Fit_terms_set_cross_productContext(
            self, self._ctx, self.state
        )
        self.enterRule(localctx, 8, self.RULE_fit_terms_set_cross_product)
        self._la = 0  # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 33
            self.fit_terms_set()
            self.state = 38
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 4:
                self.state = 34
                self.match(FitTermsParser.CROSSPRODUCT)
                self.state = 35
                self.fit_terms_set()
                self.state = 40
                self._errHandler.sync(self)
                _la = self._input.LA(1)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class Fit_terms_expressionContext(ParserRuleContext):
        __slots__ = "parser"

        def __init__(
            self,
            parser,
            parent: ParserRuleContext = None,
            invokingState: int = -1,
        ):
            super().__init__(parent, invokingState)
            self.parser = parser

        def fit_terms_set_cross_product(self, i: int = None):
            if i is None:
                return self.getTypedRuleContexts(
                    FitTermsParser.Fit_terms_set_cross_productContext
                )
            else:
                return self.getTypedRuleContext(
                    FitTermsParser.Fit_terms_set_cross_productContext, i
                )

        def UNION(self, i: int = None):
            if i is None:
                return self.getTokens(FitTermsParser.UNION)
            else:
                return self.getToken(FitTermsParser.UNION, i)

        def getRuleIndex(self):
            return FitTermsParser.RULE_fit_terms_expression

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterFit_terms_expression"):
                listener.enterFit_terms_expression(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitFit_terms_expression"):
                listener.exitFit_terms_expression(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitFit_terms_expression"):
                return visitor.visitFit_terms_expression(self)
            else:
                return visitor.visitChildren(self)

    def fit_terms_expression(self):

        localctx = FitTermsParser.Fit_terms_expressionContext(
            self, self._ctx, self.state
        )
        self.enterRule(localctx, 10, self.RULE_fit_terms_expression)
        self._la = 0  # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 41
            self.fit_terms_set_cross_product()
            self.state = 46
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 5:
                self.state = 42
                self.match(FitTermsParser.UNION)
                self.state = 43
                self.fit_terms_set_cross_product()
                self.state = 48
                self._errHandler.sync(self)
                _la = self._input.LA(1)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx
