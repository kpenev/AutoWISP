# pylint: skip-file
# Generated from /home/kpenev/projects/git/PhotometryPipeline/scripts/FitTermsParser.g4 by ANTLR 4.7.1
# encoding: utf-8
from antlr4 import *
from io import StringIO
from typing.io import TextIO
import sys

def serializedATN():
    with StringIO() as buf:
        buf.write("\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\3\13")
        buf.write("\64\4\2\t\2\4\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4\7\t\7\3")
        buf.write("\2\3\2\3\3\3\3\3\3\3\3\7\3\25\n\3\f\3\16\3\30\13\3\3\3")
        buf.write("\3\3\3\4\3\4\3\4\3\4\3\5\3\5\5\5\"\n\5\3\6\3\6\3\6\7\6")
        buf.write("\'\n\6\f\6\16\6*\13\6\3\7\3\7\3\7\7\7/\n\7\f\7\16\7\62")
        buf.write("\13\7\3\7\2\2\b\2\4\6\b\n\f\2\2\2\61\2\16\3\2\2\2\4\20")
        buf.write("\3\2\2\2\6\33\3\2\2\2\b!\3\2\2\2\n#\3\2\2\2\f+\3\2\2\2")
        buf.write("\16\17\7\13\2\2\17\3\3\2\2\2\20\21\7\3\2\2\21\26\5\2\2")
        buf.write("\2\22\23\7\n\2\2\23\25\5\2\2\2\24\22\3\2\2\2\25\30\3\2")
        buf.write("\2\2\26\24\3\2\2\2\26\27\3\2\2\2\27\31\3\2\2\2\30\26\3")
        buf.write("\2\2\2\31\32\7\t\2\2\32\5\3\2\2\2\33\34\7\5\2\2\34\35")
        buf.write("\7\4\2\2\35\36\5\4\3\2\36\7\3\2\2\2\37\"\5\4\3\2 \"\5")
        buf.write("\6\4\2!\37\3\2\2\2! \3\2\2\2\"\t\3\2\2\2#(\5\b\5\2$%\7")
        buf.write("\6\2\2%\'\5\b\5\2&$\3\2\2\2\'*\3\2\2\2(&\3\2\2\2()\3\2")
        buf.write("\2\2)\13\3\2\2\2*(\3\2\2\2+\60\5\n\6\2,-\7\7\2\2-/\5\n")
        buf.write("\6\2.,\3\2\2\2/\62\3\2\2\2\60.\3\2\2\2\60\61\3\2\2\2\61")
        buf.write("\r\3\2\2\2\62\60\3\2\2\2\6\26!(\60")
        return buf.getvalue()


class FitTermsParser ( Parser ):

    grammarFileName = "FitTermsParser.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'{'", "<INVALID>", "'O'", "'*'", "'+'", 
                     "<INVALID>", "'}'", "','" ]

    symbolicNames = [ "<INVALID>", "TERM_LIST_START", "UINT", "POLY_START", 
                      "CROSSPRODUCT", "UNION", "WS", "TERM_LIST_END", "TERM_SEP", 
                      "TERM" ]

    RULE_fit_term = 0
    RULE_fit_terms_list = 1
    RULE_fit_polynomial = 2
    RULE_fit_terms_set = 3
    RULE_fit_terms_set_cross_product = 4
    RULE_fit_terms_expression = 5

    ruleNames =  [ "fit_term", "fit_terms_list", "fit_polynomial", "fit_terms_set", 
                   "fit_terms_set_cross_product", "fit_terms_expression" ]

    EOF = Token.EOF
    TERM_LIST_START=1
    UINT=2
    POLY_START=3
    CROSSPRODUCT=4
    UNION=5
    WS=6
    TERM_LIST_END=7
    TERM_SEP=8
    TERM=9

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.7.1")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None



    class Fit_termContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def TERM(self):
            return self.getToken(FitTermsParser.TERM, 0)

        def getRuleIndex(self):
            return FitTermsParser.RULE_fit_term

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterFit_term" ):
                listener.enterFit_term(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitFit_term" ):
                listener.exitFit_term(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitFit_term" ):
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

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def TERM_LIST_START(self):
            return self.getToken(FitTermsParser.TERM_LIST_START, 0)

        def fit_term(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(FitTermsParser.Fit_termContext)
            else:
                return self.getTypedRuleContext(FitTermsParser.Fit_termContext,i)


        def TERM_LIST_END(self):
            return self.getToken(FitTermsParser.TERM_LIST_END, 0)

        def TERM_SEP(self, i:int=None):
            if i is None:
                return self.getTokens(FitTermsParser.TERM_SEP)
            else:
                return self.getToken(FitTermsParser.TERM_SEP, i)

        def getRuleIndex(self):
            return FitTermsParser.RULE_fit_terms_list

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterFit_terms_list" ):
                listener.enterFit_terms_list(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitFit_terms_list" ):
                listener.exitFit_terms_list(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitFit_terms_list" ):
                return visitor.visitFit_terms_list(self)
            else:
                return visitor.visitChildren(self)




    def fit_terms_list(self):

        localctx = FitTermsParser.Fit_terms_listContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_fit_terms_list)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 14
            self.match(FitTermsParser.TERM_LIST_START)
            self.state = 15
            self.fit_term()
            self.state = 20
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==FitTermsParser.TERM_SEP:
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

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.order = None # Token

        def POLY_START(self):
            return self.getToken(FitTermsParser.POLY_START, 0)

        def fit_terms_list(self):
            return self.getTypedRuleContext(FitTermsParser.Fit_terms_listContext,0)


        def UINT(self):
            return self.getToken(FitTermsParser.UINT, 0)

        def getRuleIndex(self):
            return FitTermsParser.RULE_fit_polynomial

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterFit_polynomial" ):
                listener.enterFit_polynomial(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitFit_polynomial" ):
                listener.exitFit_polynomial(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitFit_polynomial" ):
                return visitor.visitFit_polynomial(self)
            else:
                return visitor.visitChildren(self)




    def fit_polynomial(self):

        localctx = FitTermsParser.Fit_polynomialContext(self, self._ctx, self.state)
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

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def fit_terms_list(self):
            return self.getTypedRuleContext(FitTermsParser.Fit_terms_listContext,0)


        def fit_polynomial(self):
            return self.getTypedRuleContext(FitTermsParser.Fit_polynomialContext,0)


        def getRuleIndex(self):
            return FitTermsParser.RULE_fit_terms_set

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterFit_terms_set" ):
                listener.enterFit_terms_set(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitFit_terms_set" ):
                listener.exitFit_terms_set(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitFit_terms_set" ):
                return visitor.visitFit_terms_set(self)
            else:
                return visitor.visitChildren(self)




    def fit_terms_set(self):

        localctx = FitTermsParser.Fit_terms_setContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_fit_terms_set)
        try:
            self.state = 31
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [FitTermsParser.TERM_LIST_START]:
                self.enterOuterAlt(localctx, 1)
                self.state = 29
                self.fit_terms_list()
                pass
            elif token in [FitTermsParser.POLY_START]:
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

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def fit_terms_set(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(FitTermsParser.Fit_terms_setContext)
            else:
                return self.getTypedRuleContext(FitTermsParser.Fit_terms_setContext,i)


        def getRuleIndex(self):
            return FitTermsParser.RULE_fit_terms_set_cross_product

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterFit_terms_set_cross_product" ):
                listener.enterFit_terms_set_cross_product(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitFit_terms_set_cross_product" ):
                listener.exitFit_terms_set_cross_product(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitFit_terms_set_cross_product" ):
                return visitor.visitFit_terms_set_cross_product(self)
            else:
                return visitor.visitChildren(self)




    def fit_terms_set_cross_product(self):

        localctx = FitTermsParser.Fit_terms_set_cross_productContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_fit_terms_set_cross_product)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 33
            self.fit_terms_set()
            self.state = 38
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==FitTermsParser.CROSSPRODUCT:
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

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def fit_terms_set_cross_product(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(FitTermsParser.Fit_terms_set_cross_productContext)
            else:
                return self.getTypedRuleContext(FitTermsParser.Fit_terms_set_cross_productContext,i)


        def getRuleIndex(self):
            return FitTermsParser.RULE_fit_terms_expression

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterFit_terms_expression" ):
                listener.enterFit_terms_expression(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitFit_terms_expression" ):
                listener.exitFit_terms_expression(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitFit_terms_expression" ):
                return visitor.visitFit_terms_expression(self)
            else:
                return visitor.visitChildren(self)




    def fit_terms_expression(self):

        localctx = FitTermsParser.Fit_terms_expressionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_fit_terms_expression)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 41
            self.fit_terms_set_cross_product()
            self.state = 46
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==FitTermsParser.UNION:
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





