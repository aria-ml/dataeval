#!/bin/bash
# Fetch .jupyter_cache from orphan branch docs-artifacts/<branch-name>
#
# Usage: ./fetch-docs-cache.sh [branch-name]
#   If branch-name is not provided, uses current branch name

set -e

# Determine target branch
if [ -n "$1" ]; then
    BRANCH_NAME="$1"
elif [ -n "$CI_COMMIT_REF_NAME" ]; then
    # GitLab CI
    BRANCH_NAME="$CI_COMMIT_REF_NAME"
elif [ -n "$GITHUB_REF_NAME" ]; then
    # GitHub Actions
    BRANCH_NAME="$GITHUB_REF_NAME"
else
    # Local execution
    BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD)
fi

ARTIFACT_BRANCH="docs-artifacts/$BRANCH_NAME"
FALLBACK_BRANCH="docs-artifacts/main"
CACHE_DIR="docs/source/.jupyter_cache"

# Fetch and extract .jupyter_cache from an artifact branch.
# Returns 0 on success, 1 on failure.
fetch_cache() {
    local branch="$1"

    if ! git ls-remote --exit-code --heads origin "$branch" >/dev/null 2>&1; then
        echo "  Branch '$branch' does not exist on remote"
        return 1
    fi

    echo "✓ Artifact branch '$branch' exists on remote"
    git fetch --depth=1 origin "$branch"

    if ! git show "origin/$branch:.jupyter_cache" >/dev/null 2>&1; then
        echo "  No .jupyter_cache found in '$branch'"
        return 1
    fi

    echo "Extracting .jupyter_cache from $branch..."

    CACHE_SOURCE_FILE="$CACHE_DIR/.cache_source"
    if [ -d "$CACHE_DIR" ] && [ -f "$CACHE_SOURCE_FILE" ] && [ "$(cat "$CACHE_SOURCE_FILE")" = "$branch" ]; then
        echo "  Cache source matches '$branch', preserving newer local files"
        git archive "origin/$branch" .jupyter_cache | tar -x --keep-newer-files -C docs/source/
    else
        echo "  Replacing cache (source branch changed or no prior cache)"
        rm -rf "$CACHE_DIR"
        mkdir -p "$CACHE_DIR"
        git archive "origin/$branch" .jupyter_cache | tar -x -C docs/source/
    fi
    echo "$branch" > "$CACHE_SOURCE_FILE"

    FILE_COUNT=$(find "$CACHE_DIR" -type f 2>/dev/null | wc -l)
    CACHE_SIZE=$(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1)
    echo "✓ Cache restored successfully from $branch"
    echo "  - Files: $FILE_COUNT"
    echo "  - Size: $CACHE_SIZE"
    return 0
}

echo "================================================"
echo "Fetching Jupyter cache for branch: $BRANCH_NAME"
echo "Artifact branch: $ARTIFACT_BRANCH"
echo "================================================"

if ! fetch_cache "$ARTIFACT_BRANCH"; then
    if [ "$ARTIFACT_BRANCH" != "$FALLBACK_BRANCH" ]; then
        echo "Falling back to $FALLBACK_BRANCH..."
        if ! fetch_cache "$FALLBACK_BRANCH"; then
            echo "⚠ No cache available (normal for first-time builds)"
        fi
    else
        echo "⚠ No cache available (normal for first-time builds)"
    fi
fi

echo "================================================"
echo "Cache fetch completed"
echo "================================================"
