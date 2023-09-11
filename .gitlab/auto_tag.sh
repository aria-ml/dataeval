#!/bin/bash
git fetch --tags

currentVer=$(git describe --tags --abbrev=0 2>/dev/null) || exit_code=$?
if [[ -z "$currentVer" ]]; then
    currentVer="v0.0.0"
fi

echo "Latest tag found: $currentVer"
currentMinorVer=$(cut -d '.' -f2 <<< $currentVer)
pendingMinorVer=$(( 21+($(date +%s) - $(date -d "2023-07-19 UTC" +%s)) / (60*60*24*14) ))
pendingVer="v0.$pendingMinorVer.0"
patchVer=$(echo $currentVer | perl -pe 's/(\d+)(?!.*\d+)/$1+1/e')

if (($pendingMinorVer > $currentMinorVer)); then
    newVer=$pendingVer
else
    newVer=$patchVer
fi

echo "Tagging main branch with $newVer"
REPO_URL="https://gitlab.jatic.net/api/v4/projects/151/repository/"
TAG_QUERY="tags?tag_name=$newVer&ref=main&release_description='DAML%20$newVer'"
curl --verbose --request POST --header "PRIVATE-TOKEN: $DAML_BUILD_PAT" "${REPO_URL}${TAG_QUERY}"
