#!/usr/bin/env bash
# Decide whether the release being built should take the floating `latest`
# pointer, and print it if so.
#
#   git tag -l 'v*' | docker/promote-latest.sh
#
# Reads the repo's tag list on stdin and CI_COMMIT_TAG from the environment.
# Prints `latest` when this tag is the highest stable release, and nothing
# otherwise, for the caller to apply with `docker buildx imagetools create`.
#
# `latest` means "newest release", so only the highest stable version may hold
# it. Without that check a v1.1.4 patch cut after v1.2.0 has shipped would drag
# `latest` backwards onto the older line -- users pulling it would silently
# downgrade. The branch build publishes `${CI_DEFAULT_BRANCH}-<variant>` and is
# never a candidate here.
#
# Prereleases get nothing. A v1.2.0-rc1 is reachable by its own version tag, so
# nobody pulling `latest` is ever handed a release candidate by accident.

set -euo pipefail

TAG="${CI_COMMIT_TAG:-}"

# Stable releases only: vMAJOR.MINOR.PATCH with nothing trailing.
STABLE_RE='^v([0-9]+)\.([0-9]+)\.([0-9]+)$'

[[ -z "$TAG" ]] && exit 0
[[ ! "$TAG" =~ $STABLE_RE ]] && exit 0

# `sort -V` orders by version rather than lexically, so v1.10.0 correctly
# outranks v1.2.0. `|| true` because grep exits 1 when no stable tag exists yet,
# which is a valid state rather than a failure.
#
# $TAG joins the candidates rather than being assumed present on stdin: a tag
# pipeline runs against a checkout whose fetch may not have brought the tag
# being built back down, and without this that build would forfeit the pointer
# it had earned. Duplicates are harmless, only the maximum is read.
HIGHEST=$( { grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' || true; echo "$TAG"; } | sort -V | tail -1)

[[ "$TAG" == "$HIGHEST" ]] && echo "latest"
exit 0
