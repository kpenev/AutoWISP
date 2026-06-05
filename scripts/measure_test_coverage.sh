#!/bin/bash

REPO_ROOT=~/projects/git/AutoWISP

export COVERAGE_FILE=${REPO_ROOT}/.coverage 
export COVERAGE_PROCESS_START=${REPO_ROOT}/.coveragerc 

cd ${REPO_ROOT}
coverage erase
coverage run     --rcfile=${REPO_ROOT}/.coveragerc -m autowisp.tests failed_test
coverage combine --rcfile=${REPO_ROOT}/.coveragerc ${REPO_ROOT}
coverage report  --rcfile=${REPO_ROOT}/.coveragerc -m
coverage html    --rcfile=${REPO_ROOT}/.coveragerc -d ${REPO_ROOT}/htmlcov
