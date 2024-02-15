#!/bin/bash

if [ "$1" == "--post" ]; then POST=1; fi

# array of minor versions based off of biweekly milestone & sprint
declare -a minorVersions=(21 22 23 24 25 31 32 33 34 35 36 37 38 40 42 43 44 45 46 51 52 53 54 55 56)

git fetch --tags
currentVer=$(git describe --tags --abbrev=0 2>/dev/null) || exit_code=$?
if [[ -z "$currentVer" ]]; then
    currentVer="v0.0.0"
fi

echo "Latest tag found: $currentVer"
currentMinorVer=$(cut -d '.' -f2 <<< $currentVer)
# calculates index of minorVersion based off of biweekly offset from last day of first sprint (8/2/2023)
pendingMinorVer=${minorVersions[$(( ($(date +%s) - $(date -d "2023-08-02" -u +%s)) / (60*60*24*14) ))]}
pendingVer="v0.$pendingMinorVer.0"
patchVer=$(echo $currentVer | perl -pe 's/(\d+)(?!.*\d+)/$1+1/e')

if (($pendingMinorVer > $currentMinorVer)); then
    newVer=$pendingVer
else
    newVer=$patchVer
fi

REPO_URL="https://gitlab.jatic.net/api/v4/projects/151/repository/"
TAG_QUERY="tags?tag_name=$newVer&ref=main&message='DAML%20$newVer'"
if [[ $POST ]]; then
    echo "Tagging main branch with $newVer"
    curl --verbose --request POST --header "PRIVATE-TOKEN: $DAML_BUILD_PAT" "${REPO_URL}${TAG_QUERY}"
else
    echo "Command that will be sent with --post:"
    echo curl --verbose --request POST --header "PRIVATE-TOKEN: $DAML_BUILD_PAT" "${REPO_URL}${TAG_QUERY}"
fi