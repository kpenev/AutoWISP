#!/bin/bash

#function antlr4()
#{
#    java \
#        -Xmx500M \
#        -cp "/usr/local/lib/antlr-4.7.1-complete.jar:$CLASSPATH" \
#        org.antlr.v4.Tool \
#        "$@"
#}

#ANTLR set-up
#export CLASSPATH=".:/usr/local/lib/antlr-4.7.1-complete.jar:$CLASSPATH"

THIS_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DESTINATION=${THIS_PATH}/../superphot_pipeline/fit_expression

rm -f tmp/*
mkdir -p tmp
antlr4 ${THIS_PATH}/FitTermsLexer.g4 ${THIS_PATH}/FitTermsParser.g4 -Dlanguage=Python3 -visitor -o tmp

mkdir -p ${DESTINATION}

for f in tmp/FitTerms*.py; do
    FNAME=$(basename $f)
    echo "Creating ${DESTINATION}/${FNAME}"
    (echo "# pylint: skip-file"; cat $f) > ${DESTINATION}/${FNAME}
done
