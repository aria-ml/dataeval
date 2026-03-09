#!/bin/bash
# Fetch .jupyter_cache from orphan branch docs-artifacts/<branch-name>
#
# Usage: ./fetch-docs-cache.sh [branch-name]
#   If branch-name is not provided, uses current branch name

set -e

# Determine target branch
if [ -n "$1" ]; then
    BRANCH_NAME="$1"
elif [ -n "$READTHEDOCS_GIT_IDENTIFIER" ]; then
    # ReadTheDocs (detached HEAD, so we need the env var)
    BRANCH_NAME="$READTHEDOCS_GIT_IDENTIFIER"
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

# For version tags (e.g., v1.0.1), find the nearest prior version's artifact branch
# so patch releases pull cache from the prior release instead of main.
find_version_fallback() {
    local version="$1"
    # Extract major.minor.patch
    if [[ "$version" =~ ^v([0-9]+)\.([0-9]+)\.([0-9]+) ]]; then
        local major="${BASH_REMATCH[1]}"
        local minor="${BASH_REMATCH[2]}"
        local patch="${BASH_REMATCH[3]}"

        # List all docs-artifacts/v* branches on remote and find the best match
        local best_branch=""
        local best_patch=-1

        while IFS= read -r ref; do
            local ref_name="${ref##*/}"  # strip refs/heads/docs-artifacts/
            ref_name="${ref#*refs/heads/docs-artifacts/}"
            if [[ "$ref_name" =~ ^v([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
                local ref_major="${BASH_REMATCH[1]}"
                local ref_minor="${BASH_REMATCH[2]}"
                local ref_patch="${BASH_REMATCH[3]}"
                # Same major.minor, lower or equal patch, higher than current best
                if [ "$ref_major" -eq "$major" ] && [ "$ref_minor" -eq "$minor" ] && \
                   [ "$ref_patch" -lt "$patch" ] && [ "$ref_patch" -gt "$best_patch" ]; then
                    best_patch="$ref_patch"
                    best_branch="docs-artifacts/v${ref_major}.${ref_minor}.${ref_patch}"
                fi
            fi
        done < <(git ls-remote --heads origin 'refs/heads/docs-artifacts/v*' 2>/dev/null)

        if [ -n "$best_branch" ]; then
            echo "$best_branch"
            return 0
        fi
    fi
    return 1
}

# For release branches (e.g., release/v1.0), find the latest version artifact branch
find_release_branch_fallback() {
    local branch="$1"
    if [[ "$branch" =~ ^release/v([0-9]+)\.([0-9]+) ]]; then
        local major="${BASH_REMATCH[1]}"
        local minor="${BASH_REMATCH[2]}"

        local best_branch=""
        local best_patch=-1

        while IFS= read -r ref; do
            local ref_name="${ref#*refs/heads/docs-artifacts/}"
            if [[ "$ref_name" =~ ^v([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
                local ref_major="${BASH_REMATCH[1]}"
                local ref_minor="${BASH_REMATCH[2]}"
                local ref_patch="${BASH_REMATCH[3]}"
                if [ "$ref_major" -eq "$major" ] && [ "$ref_minor" -eq "$minor" ] && \
                   [ "$ref_patch" -gt "$best_patch" ]; then
                    best_patch="$ref_patch"
                    best_branch="docs-artifacts/v${ref_major}.${ref_minor}.${ref_patch}"
                fi
            fi
        done < <(git ls-remote --heads origin 'refs/heads/docs-artifacts/v*' 2>/dev/null)

        if [ -n "$best_branch" ]; then
            echo "$best_branch"
            return 0
        fi
    fi
    return 1
}

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

    # Use FETCH_HEAD instead of origin/$branch because single-branch clones
    # (e.g., ReadTheDocs) don't create remote tracking refs for other branches
    if ! git show "FETCH_HEAD:.jupyter_cache" >/dev/null 2>&1; then
        echo "  No .jupyter_cache found in '$branch'"
        return 1
    fi

    echo "Extracting .jupyter_cache from $branch..."

    CACHE_SOURCE_FILE="$CACHE_DIR/.cache_source"
    if [ -z "$CI" ] && [ -d "$CACHE_DIR" ] && [ -f "$CACHE_SOURCE_FILE" ] && [ "$(cat "$CACHE_SOURCE_FILE")" = "$branch" ]; then
        echo "  Cache source matches '$branch', preserving newer local files"
        git archive "FETCH_HEAD" .jupyter_cache | tar -x --keep-newer-files -C docs/source/
    else
        echo "  Replacing cache (source branch changed or no prior cache)"
        rm -rf "$CACHE_DIR"
        mkdir -p "$CACHE_DIR"
        git archive "FETCH_HEAD" .jupyter_cache | tar -x -C docs/source/
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
    # For version tags or release branches, try the nearest version artifact branch
    VERSION_FALLBACK=""
    if [[ "$BRANCH_NAME" =~ ^v[0-9]+\.[0-9]+\.[0-9]+ ]]; then
        VERSION_FALLBACK=$(find_version_fallback "$BRANCH_NAME") || true
    elif [[ "$BRANCH_NAME" =~ ^release/v ]]; then
        VERSION_FALLBACK=$(find_release_branch_fallback "$BRANCH_NAME") || true
    fi

    if [ -n "$VERSION_FALLBACK" ]; then
        echo "Falling back to nearest version artifact: $VERSION_FALLBACK..."
        if ! fetch_cache "$VERSION_FALLBACK"; then
            echo "Falling back to $FALLBACK_BRANCH..."
            if ! fetch_cache "$FALLBACK_BRANCH"; then
                echo "⚠ No cache available (normal for first-time builds)"
            fi
        fi
    elif [ "$ARTIFACT_BRANCH" != "$FALLBACK_BRANCH" ]; then
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
