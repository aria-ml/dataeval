#!/bin/bash
# Push .jupyter_cache and notebook .ipynb files to orphan branch docs-artifacts/<branch-name>
# If HEAD has a version tag, also pushes to docs-artifacts/<tag> for Colab links.
#
# Usage: ./push-docs-cache.sh [branch-name]
#   If branch-name is not provided, uses current branch name

set -e

# Determine target branch
if [ -n "$1" ]; then
    BRANCH_NAME="$1"
elif [ -n "$CI_COMMIT_REF_NAME" ]; then
    # GitLab CI
    BRANCH_NAME="$CI_COMMIT_REF_NAME"
# elif [ -n "$GITHUB_REF_NAME" ]; then
#     # GitHub Actions
#     BRANCH_NAME="$GITHUB_REF_NAME"
else
    # Local execution
    BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD)
fi

ARTIFACT_BRANCH="docs-artifacts/$BRANCH_NAME"
SOURCE_CACHE_DIR="docs/source/.jupyter_cache"
CACHE_SIZE=$(du -sh "$SOURCE_CACHE_DIR" 2>/dev/null | cut -f1 || echo "0")
FILE_COUNT=$(find "$SOURCE_CACHE_DIR" -type f 2>/dev/null | wc -l || echo "0")

echo "================================================"
echo "Pushing Jupyter cache for branch: $BRANCH_NAME"
echo "Artifact branch: $ARTIFACT_BRANCH"
echo "Cache size: $CACHE_SIZE ($FILE_COUNT files)"
echo "================================================"

# Verify cache directory exists
if [ ! -d "$SOURCE_CACHE_DIR" ]; then
    echo "Error: Cache directory not found at $SOURCE_CACHE_DIR"
    echo "Documentation build may have failed or cache was not generated"
    exit 1
fi

# Get repository URL before changing directory
if [ -n "$DATAEVAL_BUILD_PAT" ]; then
    # GitLab CI - use bot account with PAT for push access
    REPO_URL="https://dataeval-bot:${DATAEVAL_BUILD_PAT}@gitlab.jatic.net/jatic/aria/dataeval.git"
# elif [ -n "$GITHUB_TOKEN" ]; then
#     # GitHub Actions - construct authenticated URL
#     REPO_URL="https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
else
    # Local execution - get origin URL from current repo
    REPO_URL=$(git config --get remote.origin.url)
    if [ -z "$REPO_URL" ]; then
        echo "Error: Could not determine repository URL"
        echo "Please ensure you have an 'origin' remote configured"
        exit 1
    fi
fi

echo "Repository URL: ${REPO_URL%%@*}@***" # Mask credentials in output

# Capture git identity before leaving the repo directory
GIT_USER_NAME="${GITLAB_USER_NAME:-$(git config user.name)}"
GIT_USER_EMAIL="${GITLAB_USER_EMAIL:-$(git config user.email)}"

# Create a temporary directory for the orphan branch
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

cd "$TEMP_DIR"

# Clear git environment variables that may leak from CI runners.
# If GIT_DIR or GIT_WORK_TREE point to the original repo, git operations
# here would use the original repo's .gitignore (which ignores .jupyter_cache/).
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_CEILING_DIRECTORIES

# Initialize new orphan branch
git init
git config user.name "$GIT_USER_NAME"
git config user.email "$GIT_USER_EMAIL"
git config http.postBuffer 100m

# Copy cache files
echo "Copying cache files..."
cp -r "$OLDPWD/$SOURCE_CACHE_DIR" .jupyter_cache/

# Copy notebook ipynb files (generated during docs build by jupytext)
NOTEBOOK_DIR="$OLDPWD/docs/source/notebooks"
if [ -d "$NOTEBOOK_DIR" ]; then
    mkdir -p notebooks/
    cp "$NOTEBOOK_DIR"/*.ipynb notebooks/ 2>/dev/null || true
    NB_COUNT=$(find notebooks -name "*.ipynb" 2>/dev/null | wc -l)
    echo "Copied $NB_COUNT notebook files to notebooks/"
fi

# Create README for the artifact branch
cat > README.md <<EOF
# Documentation Artifacts: $BRANCH_NAME

This orphan branch stores build artifacts for the \`$BRANCH_NAME\` branch.

## Contents

- \`.jupyter_cache/\`: Cached Jupyter notebook execution results
- \`notebooks/\`: Generated .ipynb files for Google Colab

## Purpose

This branch is automatically managed by CI/CD pipelines to:
- Speed up documentation builds by reusing cached notebook executions
- Avoid committing large binary files to the main branch
- Persist cache across CI runs

## Last Updated

$(date -u +"%Y-%m-%d %H:%M:%S UTC")

Branch: $BRANCH_NAME
Files: $FILE_COUNT
Size: $CACHE_SIZE

---
**Note:** This is an orphan branch with no history connection to main.
Do not merge this branch into main or other working branches.
EOF

# Verify cache has files before committing
COPIED_COUNT=$(find .jupyter_cache -type f 2>/dev/null | wc -l)
if [ "$COPIED_COUNT" -eq 0 ]; then
    echo "Error: .jupyter_cache has no files after copy"
    echo "Contents of temp dir:"
    find . -not -path './.git/*' -not -path './.git'
    exit 1
fi
echo "Verified: $COPIED_COUNT files in .jupyter_cache"

# Add and commit (--force bypasses any inherited .gitignore rules)
git add --force .
git commit -m "Update docs cache for $BRANCH_NAME

Generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Files: $FILE_COUNT
Size: $CACHE_SIZE"

# Release branches (release/v*) only push to docs-artifacts/<tag>, never to
# docs-artifacts/release/v* which would be an orphan nobody fetches from.
if [[ "$BRANCH_NAME" =~ ^release/v ]]; then
    echo "Release branch detected — skipping push to $ARTIFACT_BRANCH"
    echo "Will only push to versioned artifact branch if a tag exists on HEAD"
else
    # Push to artifact branch (force push since it's orphan)
    echo "Pushing to $ARTIFACT_BRANCH..."
    git push --force "$REPO_URL" HEAD:"refs/heads/$ARTIFACT_BRANCH"
    echo "================================================"
    echo "✓ Cache pushed successfully to $ARTIFACT_BRANCH"
    echo "================================================"
fi

cd "$OLDPWD"

# If HEAD has a version tag, also push to docs-artifacts/<tag> for Colab links
git fetch --tags origin 2>/dev/null || true
VERSION_TAG=$(git tag --points-at HEAD 2>/dev/null | grep '^v[0-9]' | head -1)
if [ -n "$VERSION_TAG" ]; then
    VERSION_ARTIFACT_BRANCH="docs-artifacts/$VERSION_TAG"
    echo ""
    echo "================================================"
    echo "Detected version tag: $VERSION_TAG"
    echo "Pushing to $VERSION_ARTIFACT_BRANCH..."
    echo "================================================"
    cd "$TEMP_DIR"
    git push --force "$REPO_URL" HEAD:"refs/heads/$VERSION_ARTIFACT_BRANCH"
    cd "$OLDPWD"
    echo "✓ Pushed to $VERSION_ARTIFACT_BRANCH"
elif [[ "$BRANCH_NAME" =~ ^release/v ]]; then
    echo ""
    echo "⚠ Release branch has no version tag on HEAD — no artifact branch pushed"
    echo "  Artifacts will be pushed when create_patch_release.py tags this commit"
fi
