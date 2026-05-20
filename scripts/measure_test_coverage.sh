#!/bin/bash

REPO_ROOT=~/projects/git/AutoWISP

export COVERAGE_FILE=${REPO_ROOT}/.coverage 
export COVERAGE_PROCESS_START=${REPO_ROOT}/.coveragerc 

coverage run --rcfile=${REPO_ROOT}/.coveragerc -m autowisp.tests failed_test
coverage combine . ${REPO_ROOT}
coverage report -m --rcfile=${REPO_ROOT}/.coveragerc
coverage html --rcfile=${REPO_ROOT}/.coveragerc -d ${REPO_ROOT}/htmlcov
