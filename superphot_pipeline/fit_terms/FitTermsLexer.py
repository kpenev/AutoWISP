# pylint: skip-file
# Generated from /home/kpenev/projects/git/PhotometryPipeline/scripts/FitTermsLexer.g4 by ANTLR 4.7.1
from antlr4 import *
from io import StringIO
from typing.io import TextIO
import sys


def serializedATN():
    with StringIO() as buf:
        buf.write("\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\2\13")
        buf.write("8\b\1\b\1\4\2\t\2\4\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4\7")
        buf.write("\t\7\4\b\t\b\4\t\t\t\4\n\t\n\4\13\t\13\3\2\3\2\3\2\3\2")
        buf.write("\3\3\3\3\3\4\3\4\3\5\3\5\3\6\3\6\3\7\6\7&\n\7\r\7\16\7")
        buf.write("\'\3\7\3\7\3\b\3\b\3\b\3\b\3\t\3\t\3\n\6\n\63\n\n\r\n")
        buf.write("\16\n\64\3\13\3\13\2\2\f\4\3\6\4\b\5\n\6\f\7\16\b\20\t")
        buf.write("\22\n\24\13\26\2\4\2\3\4\5\2\13\f\17\17\"\"\5\2..}}\177")
        buf.write("\177\2\67\2\4\3\2\2\2\2\6\3\2\2\2\2\b\3\2\2\2\2\n\3\2")
        buf.write("\2\2\2\f\3\2\2\2\2\16\3\2\2\2\3\20\3\2\2\2\3\22\3\2\2")
        buf.write("\2\3\24\3\2\2\2\4\30\3\2\2\2\6\34\3\2\2\2\b\36\3\2\2\2")
        buf.write("\n \3\2\2\2\f\"\3\2\2\2\16%\3\2\2\2\20+\3\2\2\2\22/\3")
        buf.write("\2\2\2\24\62\3\2\2\2\26\66\3\2\2\2\30\31\7}\2\2\31\32")
        buf.write("\3\2\2\2\32\33\b\2\2\2\33\5\3\2\2\2\34\35\4\62;\2\35\7")
        buf.write("\3\2\2\2\36\37\7Q\2\2\37\t\3\2\2\2 !\7,\2\2!\13\3\2\2")
        buf.write("\2\"#\7-\2\2#\r\3\2\2\2$&\t\2\2\2%$\3\2\2\2&\'\3\2\2\2")
        buf.write("\'%\3\2\2\2\'(\3\2\2\2()\3\2\2\2)*\b\7\3\2*\17\3\2\2\2")
        buf.write("+,\7\177\2\2,-\3\2\2\2-.\b\b\4\2.\21\3\2\2\2/\60\7.\2")
        buf.write("\2\60\23\3\2\2\2\61\63\5\26\13\2\62\61\3\2\2\2\63\64\3")
        buf.write("\2\2\2\64\62\3\2\2\2\64\65\3\2\2\2\65\25\3\2\2\2\66\67")
        buf.write("\n\3\2\2\67\27\3\2\2\2\6\2\3\'\64\5\7\3\2\b\2\2\6\2\2")
        return buf.getvalue()


class FitTermsLexer(Lexer):

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    TERM_LIST = 1

    TERM_LIST_START = 1
    UINT = 2
    POLY_START = 3
    CROSSPRODUCT = 4
    UNION = 5
    WS = 6
    TERM_LIST_END = 7
    TERM_SEP = 8
    TERM = 9

    channelNames = [ u"DEFAULT_TOKEN_CHANNEL", u"HIDDEN" ]

    modeNames = [ "DEFAULT_MODE", "TERM_LIST" ]

    literalNames = [ "<INVALID>",
            "'{'", "'O'", "'*'", "'+'", "'}'", "','" ]

    symbolicNames = [ "<INVALID>",
            "TERM_LIST_START", "UINT", "POLY_START", "CROSSPRODUCT", "UNION", 
            "WS", "TERM_LIST_END", "TERM_SEP", "TERM" ]

    ruleNames = [ "TERM_LIST_START", "UINT", "POLY_START", "CROSSPRODUCT", 
                  "UNION", "WS", "TERM_LIST_END", "TERM_SEP", "TERM", "TERMCHAR" ]

    grammarFileName = "FitTermsLexer.g4"

    def __init__(self, input=None, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.7.1")
        self._interp = LexerATNSimulator(self, self.atn, self.decisionsToDFA, PredictionContextCache())
        self._actions = None
        self._predicates = None


