#!/bin/bash
# Restore or store the notebook datasets (docs/source/notebooks/data) in Harbor.
#
# Usage: ./harbor-datasets-cache.sh pull|push
#   pull  restore the datasets, unless the directory is already populated
#   push  store the datasets, if the preceding pull missed and a build then downloaded them
#
# Called from the docs CI job around the docs build and nowhere else: the build
# itself (conf.py, data.py) knows nothing about Harbor, so when this cache is
# absent or broken it downloads the datasets from the original sources as usual.
# The artifact is tagged with a hash of data.py, so changing the dataset list
# invalidates it.
#
# This is only a cache: a missing token, a miss or any Harbor failure is
# reported and ignored. Only runs in CI with HARBOR_TOKEN set (the group-level
# secret for `robot$aria+aria-bot`).

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NOTEBOOKS_DIR="$ROOT/docs/source/notebooks"
DATA_DIR="$NOTEBOOKS_DIR/data"
MISSED="${TMPDIR:-/tmp}/harbor-datasets-missed"
HARBOR_URL="harbor.jatic.net"
# `$aria` must stay inside single quotes or the shell expands it to nothing.
HARBOR_USER='robot$aria+aria-bot'
REF="${DOCS_DATASETS_REF:-$HARBOR_URL/aria/cache:docs-datasets-$(sha256sum "$ROOT/docs/source/data.py" | cut -c1-12)}"
ORAS_VERSION="1.2.3"
ORAS_SHA256="b4efc97a91f471f323f193ea4b4d63d8ff443ca3aab514151a30751330852827"

ACTION="$1"
case "$ACTION" in pull | push) ;; *) echo "usage: $0 pull|push" >&2; exit 2 ;; esac
if [ -z "$CI" ] || [ -z "$HARBOR_TOKEN" ]; then
    echo "harbor datasets cache: not in CI or HARBOR_TOKEN is not set, skipping $ACTION"
    exit 0
fi

populated() { [ -d "$DATA_DIR" ] && [ -n "$(ls -A "$DATA_DIR")" ]; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

install_oras() {
    curl -fsSL -o "$WORK/oras.tar.gz" \
        "https://github.com/oras-project/oras/releases/download/v$ORAS_VERSION/oras_${ORAS_VERSION}_linux_amd64.tar.gz" &&
        echo "$ORAS_SHA256  $WORK/oras.tar.gz" | sha256sum -c - >/dev/null &&
        tar -xzf "$WORK/oras.tar.gz" -C "$WORK" oras &&
        echo "$HARBOR_TOKEN" | "$WORK/oras" login "$HARBOR_URL" -u "$HARBOR_USER" --password-stdin
}

pull() {
    rm -f "$MISSED"
    if populated; then
        echo "harbor datasets cache: $DATA_DIR is already populated, not pulling"
        return 0
    fi
    install_oras || return 1
    if "$WORK/oras" pull "$REF" -o "$WORK"; then
        tar -xf "$WORK/datasets.tar" -C "$NOTEBOOKS_DIR" && echo "harbor datasets cache: restored $REF"
    else
        echo "harbor datasets cache: miss for $REF"
        touch "$MISSED"
    fi
}

push() {
    if [ ! -e "$MISSED" ] || ! populated; then
        echo "harbor datasets cache: nothing to push (no preceding miss, or no datasets downloaded)"
        return 0
    fi
    install_oras || return 1
    if "$WORK/oras" manifest fetch "$REF" >/dev/null 2>&1; then
        echo "harbor datasets cache: $REF already exists, not pushing"
        return 0
    fi
    tar -cf "$WORK/datasets.tar" -C "$NOTEBOOKS_DIR" data
    (cd "$WORK" && ./oras push "$REF" datasets.tar:application/x-tar) && echo "harbor datasets cache: stored $REF"
}

start=$SECONDS
"$ACTION" || echo "harbor datasets cache: $ACTION failed, continuing without it"
echo "harbor datasets cache: $ACTION took $((SECONDS - start))s"
exit 0
