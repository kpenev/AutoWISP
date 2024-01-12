#!/bin/bash

if [ "$1" == "reset" ]; then
    rm -f .perfect_lints
fi

touch .perfect_lints

ROOT_DIR=$(pwd)

for FNAME in $(find . -name '*.py'); do 

    if grep -q $FNAME .perfect_lints; then 
        continue 
    fi

    cd $(dirname $FNAME)
    LINT_RESULT=$(pylint --disable=fixme $(basename $FNAME);)

    cd $ROOT_DIR

    echo "$LINT_RESULT" | grep -q "Your code has been rated at 10.00/10" 

    if [ "$?" == "0" ]; then 
        echo "$FNAME" >> .perfect_lints
    else
        echo "$FNAME"
        echo "$LINT_RESULT"
    fi
    
done
