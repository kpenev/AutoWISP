"""An interface for working with fitting terms expressions."""

from asteval import asteval

from antlr4 import InputStream, CommonTokenStream
from autowisp.fit_expression.FitTermsLexer import FitTermsLexer
from autowisp.fit_expression.FitTermsParser import FitTermsParser
from autowisp.fit_expression.list_terms_visitor import ListTermsVisitor
from autowisp.fit_expression.count_terms_visitor import CountTermsVisitor
from autowisp.fit_expression.evaluate_terms_visitor import EvaluateTermsVisitor
from autowisp.fit_expression.used_var_finder import UsedVarFinder


# TODO: add detailed description of the expansion terms language
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
        self._var_names = None

    def number_terms(self):
        """Return the number of terms the expression expands to."""

        if self._number_terms is None:
            self._number_terms = CountTermsVisitor().visit(self._tree)

        return self._number_terms

    def get_term_str_list(self):
        """Return strings of the individual terms the expression expands to."""

        return ListTermsVisitor().visit(self._tree)

    def get_var_names(self):
        """Return the names of the variables used in expression."""

        if self._var_names is None:
            interpreter = asteval.Interpreter()
            interpreter.symtable = UsedVarFinder(interpreter.symtable)
            EvaluateTermsVisitor(interpreter).visit(self._tree)
            self._var_names = interpreter.symtable.get_used_vars()

        return self._var_names

    def __call__(self, *data):
        """Return an array of the term values for the given data."""

        return EvaluateTermsVisitor(*data).visit(self._tree)


if __name__ == "__main__":
    istream = InputStream("{x**2}")
    from multiprocessing import Pool

    with Pool(3) as p:
        p.map(istream, [{"x": 1}, {"x": 2}, {"x": 3}])
