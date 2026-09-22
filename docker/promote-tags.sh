#!/usr/bin/env bash
# Decide which floating tags a build should be promoted to, and print them.
#
#   git tag -l 'v*' | docker/promote-tags.sh cpu
#
# Reads the repo's tag list on stdin and CI_COMMIT_TAG from the environment
# (unset/empty on a branch pipeline). Prints zero or more floating tag names,
# one per line, for the caller to apply with `docker buildx imagetools create`.
#
# Three pointers exist, and which ones a build earns depends on where its tag
# sits relative to every other release:
#
#   edge-<variant>        a branch build; never a release pointer
#   <major>.<minor>-<variant>  the release line this tag belongs to
#   <variant>             "latest" — only the highest stable release holds it
#
# The series pointer is what makes release branches safe. Once release/v1.1 is
# cut and v1.2.0 has shipped, a v1.1.3 patch must update :1.1-cpu and leave
# :cpu pointing at the v1.2 line. Handing every tag pipeline the bare pointer
# would move :cpu *backwards* onto the older patch.
#
# Prereleases get nothing floating. DataEval tags release candidates (v1.1.0-rc6
# and friends), and an rc is reachable by its immutable version tag alone, so
# nobody pulling a floating tag is ever handed one by accident.

set -euo pipefail

VARIANT="${1:?usage: promote-tags.sh <variant>}"
TAG="${CI_COMMIT_TAG:-}"
BRANCH="${CI_COMMIT_BRANCH:-}"

# Stable releases only: vMAJOR.MINOR.PATCH with nothing trailing. Anything else
# (v1.2.0-rc1, v1.2.0-a1) is a prerelease and is excluded from both the series
# pointer and the "is this the highest release" comparison below.
STABLE_RE='^v([0-9]+)\.([0-9]+)\.([0-9]+)$'

if [[ -z "$TAG" ]]; then
    # `edge` means "newest build from main", so a release branch may not claim it —
    # an untagged v1.1 build would otherwise overwrite main's edge image. Release
    # branches build containers so an image can be scanned before its tag is
    # pushed, and that validation wants the immutable tag, not a floating one.
    if [[ "$BRANCH" == release/* ]]; then
        exit 0
    fi
    echo "edge-${VARIANT}"
    exit 0
fi

if [[ ! "$TAG" =~ $STABLE_RE ]]; then
    exit 0
fi

echo "${BASH_REMATCH[1]}.${BASH_REMATCH[2]}-${VARIANT}"

# `sort -V` orders by version rather than lexically, so v1.10.0 correctly outranks
# v1.2.0. `|| true` because grep exits 1 when no stable tag exists yet, which is a
# valid state (a repo whose only tags are prereleases), not a failure.
#
# $TAG joins the candidates rather than being assumed present in stdin. A tag
# pipeline runs against a checkout whose fetch may not have brought the tag being
# built back down; without this, that build would quietly forfeit the bare pointer
# it had earned. Duplicates are harmless — only the maximum is read.
HIGHEST=$( { grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' || true; echo "$TAG"; } | sort -V | tail -1)

if [[ "$TAG" == "$HIGHEST" ]]; then
    echo "${VARIANT}"
fi
