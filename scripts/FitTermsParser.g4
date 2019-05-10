/** Grammar for parsing expressions for terms to include in a fit. */
parser grammar FitTermsParser ;

options { tokenVocab=FitTermsLexer; }

/**
 * Any mathematical expression involving available variables, floating point
 * numbers and pi. Basically anything expression numpy can evaluate.
 */
fit_term : TERM ;

/**
 * Simple listing of terms to include.
 */
fit_terms_list : TERM_LIST_START fit_term ( TERM_SEP fit_term )* TERM_LIST_END ;

/**
 * Expands to all polynomial terms of up to combined order <integer> of the
 * entries in list
 */
fit_polynomial : POLY_START order=UINT fit_terms_list ;

fit_terms_set : fit_terms_list | fit_polynomial ;

/**
 * Expands to the cross product of all sets.
 */
fit_terms_set_cross_product : fit_terms_set ( '*' fit_terms_set )* ;

/**
 * merge the terms of all cross products together.
 */
fit_terms_expression : fit_terms_set_cross_product ( '+' fit_terms_set_cross_product )* ;
