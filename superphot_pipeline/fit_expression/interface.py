"""An interface for working with fitting terms expressions."""

from antlr4 import InputStream, CommonTokenStream
from .FitTermsLexer import FitTermsLexer
from .FitTermsParser import FitTermsParser
from .list_terms_visitor import ListTermsVisitor
from .count_terms_visitor import CountTermsVisitor
from .evaluate_terms_visitor import EvaluateTermsVisitor

class Interface:
    """
    Interface class for working with fit terms expressions.

    Attributes:
        _number_terms:    Once terms are counted, this attribute stores the
            result for re-use.
    """

    def __init__(self, expression):
        """Create an interface for working with the given expression."""

        lexer = FitTermsLexer(InputStream(expression))
        stream = CommonTokenStream(lexer)
        parser = FitTermsParser(stream)
        self._tree = parser.fit_terms_expression()
        self._number_terms = None

    def number_terms(self):
        """Return the number of terms the expression expands to."""

        if self._number_terms is None:
            self._number_terms = CountTermsVisitor().visit(self._tree)

        return self._number_terms

    def get_term_str_list(self):
        """Return strings of the individual terms the expression expands to."""

        return ListTermsVisitor().visit(self._tree)

    def __call__(self, data):
        """Return an array of the term values for the given data."""

        return EvaluateTermsVisitor(data).visit(self._tree)
