"""Implement a visitor to parsed fit terms expressions that prints all terms."""

import sys
from antlr4 import FileStream, CommonTokenStream
from FitTermsLexer import FitTermsLexer
from FitTermsParser import FitTermsParser
from FitTermsParserVisitor import FitTermsParserVisitor

class ListTermsVisitor(FitTermsParserVisitor):
    """Base class for parsing expressions for fitting terms."""

    # Visit a parse tree produced by FitTermsParser#fit_term.
    def visitFit_term(self, ctx: FitTermsParser.Fit_termContext):
        self.visitChildren(ctx)
        return '(' + ctx.TERM().getText().strip() + ')'

    # Visit a parse tree produced by FitTermsParser#fit_terms_list.
    def visitFit_terms_list(self, ctx: FitTermsParser.Fit_terms_listContext):
        """Return a list of  the terms in the list."""

        result = [self.visit(term) for term in ctx.fit_term()]
        return result

    # Visit a parse tree produced by FitTermsParser#fit_polynomial.
    def visitFit_polynomial(self, ctx: FitTermsParser.Fit_polynomialContext):
        """Return a list of the terms the polynomial expression expands to."""

        def next_power_set(powerlaw_indices):
            """
            Advance to the next term powerlaw indices preserting total order.

            Args:
                powerlaw_indices:    The powerlaw indices of the current term.
                    One for each term in the list from which the polynomial is
                    buing built.

            Returns:
                bool:
                    Whether there are any more terms left in the expansion.
            """

            for i in range(len(powerlaw_indices) - 1):
                if powerlaw_indices[i]:
                    powerlaw_indices[i + 1] += 1
                    powerlaw_indices[0] = powerlaw_indices[i] - 1
                    if i:
                        powerlaw_indices[i] = 0
                    return True
            return False

        def format_output_term(input_terms, term_powers):
            """
            Create a single term in the expanded term list.

            Args:
                input_terms([str]):    The list of terms the polynomial
                    expansion is being performed on.

                term_Powers([int]):    The powerlaw indices for each term in
                    ``input_terms``.

            Returns:
                str:
                    A human readable, yet evaluatable string representing the
                    term corresponding to the inputs.
            """

            result = []
            for term, power in zip(input_terms, term_powers):
                if power == 1:
                    result.append(term)
                elif power > 1:
                    result.append('%s**%d' % (term, power))
            return ' * '.join(result)

        max_order = int(ctx.order.text)
        terms_list = self.visit(ctx.fit_terms_list())
        term_powers = [0 for term in terms_list]
        result = ['1']
        for total_order in range(max_order):
            term_powers[-1] = 0
            term_powers[0] = total_order + 1
            result.append(format_output_term(terms_list, term_powers))
            while next_power_set(term_powers):
                result.append(format_output_term(terms_list, term_powers))
        return result

    # Visit a parse tree produced by FitTermsParser#fit_terms_set.
    def visitFit_terms_set(self, ctx: FitTermsParser.Fit_terms_setContext):
        """Return a list of the terms in the set."""

        return self.visit(ctx.getChild(0))

    # Visit a parse tree produced by FitTermsParser#fit_terms_set_cross_product.
    def visitFit_terms_set_cross_product(
            self,
            ctx: FitTermsParser.Fit_terms_set_cross_productContext
    ):
        """Return all possible terms combining one term from each input set."""

        def format_term_in_product(term):
            """Format the given term suitably for including in a product."""

            if term == '1':
                return None
            return term

        term_sets = [self.visit(tset) for tset in ctx.fit_terms_set()]
        term_indices = [0 for s in term_sets]
        more_terms = True
        result = []
        while more_terms:
            term = ' * '.join(
                filter(
                    None,
                    [
                        format_term_in_product(
                            term_sets[set_ind][term_indices[set_ind]]
                        )
                        for set_ind in range(len(term_sets))
                    ]
                )
            )
            result.append(term or '1')
            more_terms = False
            for set_ind, term_ind in enumerate(term_indices):
                if term_ind < len(term_sets[set_ind]) - 1:
                    term_indices[set_ind] += 1
                    for i in range(0, set_ind):
                        term_indices[i] = 0
                    more_terms = True
                    break
        return result

    # Visit a parse tree produced by FitTermsParser#fit_terms_expression.
    def visitFit_terms_expression(
            self,
            ctx: FitTermsParser.Fit_terms_expressionContext
    ):
        """Return all terms defined by the term expression."""

        result = []
        for child in ctx.fit_terms_set_cross_product():
            result.extend(self.visit(child))
        return result

if __name__ == '__main__':
    lexer = FitTermsLexer(FileStream(sys.argv[1]))
    stream = CommonTokenStream(lexer)
    parser = FitTermsParser(stream)
    tree = parser.fit_terms_expression()

    visitor = ListTermsVisitor()
    print('Terms:\n\t' + '\n\t'.join([repr(t) for t in visitor.visit(tree)]))
