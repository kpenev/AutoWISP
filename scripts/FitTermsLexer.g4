lexer grammar FitTermsLexer ;

TERM_LIST_START : '{'  -> pushMode(TERM_LIST);

UINT : '0'..'9' ;

POLY_START : 'O';

CROSSPRODUCT : '*';

UNION : '+';

WS : [ \r\t\n]+ -> skip ;

mode TERM_LIST ;

TERM_LIST_END : '}' -> popMode;

TERM_SEP : ',';

TERM : TERMCHAR+ ;
fragment TERMCHAR : ~[,{}] ;



