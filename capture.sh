#!/bin/bash -e
mkdir -p output
log="output/run-${1}-${2}.log"
exitcode_file="output/run-${1}-${2}-exitcode"
touch $log

set +e
set -o pipefail
${@:3} 2>&1 | tee -a $log
exitcode=$?
set -e

echo $exitcode > $exitcode_file
