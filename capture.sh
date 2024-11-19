#!/bin/bash -e
mkdir -p output
log="output/run-${1}-${2}.log"
exitcode_file="output/run-${1}-${2}-exitcode"
touch $log

set +e
set -o pipefail
if [ $(which unbuffer) ]; then
    unbuffer ${@:3} 2>&1 | tee -a $log
else
    ${@:3} 2>&1 | tee -a $log
fi
exitcode=$?
set -e

echo $exitcode > $exitcode_file
