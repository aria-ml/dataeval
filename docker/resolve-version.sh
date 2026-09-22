#!/usr/bin/env bash
# Resolve a version from git that is both PEP 440 compliant (for hatch-vcs via
# SETUPTOOLS_SCM_PRETEND_VERSION) and OCI-tag-safe (for `docker -t`).
#
# Emits two `export` statements on stdout; eval the output to set the vars:
#
#   eval "$(./docker/resolve-version.sh)"
#
# Resulting environment:
#   DATAEVAL_VERSION    PEP 440 form, e.g. 1.1.2 or 1.1.3.dev5+gabc1234
#   DATAEVAL_IMAGE_TAG  OCI-safe form, e.g. 1.1.2 or 1.1.3.dev5-gabc1234
#
# The PEP form uses `+local`; the OCI form swaps `+` for `-`, since OCI tags
# disallow `+`. Both encode tag + commit distance + short sha + dirty flag.
# Requires the full tag history (CI: GIT_DEPTH=0 + `git fetch --tags`).

set -euo pipefail

# `v[0-9]*` rather than `v*` so the moving `latest-known-good` tag, which
# `tag release candidate` repoints on every green main pipeline, can never be
# selected as the version basis.
raw=$(git describe --tags --long --dirty --match 'v[0-9]*' --always 2>/dev/null || true)

# Captures, in order: release base (X.Y.Z), optional prerelease suffix
# (-rc1, -a2, ...), commit distance, short sha, optional dirty marker.
DESCRIBE_RE='^v([0-9]+\.[0-9]+\.[0-9]+)(-[0-9A-Za-z.]+)?-([0-9]+)-g([0-9a-f]+)(-dirty)?$'

if [[ "$raw" =~ $DESCRIBE_RE ]]; then
  base="${BASH_REMATCH[1]}"
  pre="${BASH_REMATCH[2]:-}"
  dist="${BASH_REMATCH[3]}"
  sha="${BASH_REMATCH[4]}"
  dirty="${BASH_REMATCH[5]:-}"

  if [[ "$dist" == "0" && -z "$dirty" ]]; then
    # Sitting exactly on a tag: publish that version verbatim. PEP 440 wants the
    # prerelease separator gone (1.2.0-rc1 -> 1.2.0rc1); OCI tags keep the dash,
    # which is legal there and stays readable in the registry listing.
    pep="${base}${pre//-/}"
    tag="${base}${pre}"
  else
    # Past a tag. Which release this is heading toward depends on what was tagged:
    #   v1.1.2-5-gabc     -> the next patch, 1.1.3.dev5
    #   v1.2.0-rc1-3-gabc -> 1.2.0 itself, 1.2.0.dev3 (bumping the patch here
    #                        would overshoot the release the rc is staged for)
    if [[ -n "$pre" ]]; then
      next="$base"
    else
      IFS=. read -r maj min pat <<< "$base"
      next="${maj}.${min}.$((pat + 1))"
    fi
    pep="${next}.dev${dist}+g${sha}${dirty:+.dirty}"
    tag="${next}.dev${dist}-g${sha}${dirty:+-dirty}"
  fi
else
  sha=$(git rev-parse --short=7 HEAD 2>/dev/null || echo "unknown")
  pep="0.0.0.dev0+g${sha}"
  tag="0.0.0.dev0-g${sha}"
fi

printf 'export DATAEVAL_VERSION=%q\n' "$pep"
printf 'export DATAEVAL_IMAGE_TAG=%q\n' "$tag"
