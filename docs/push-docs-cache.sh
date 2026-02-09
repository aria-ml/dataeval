#!/bin/bash
# Push .jupyter_cache to orphan branch docs-artifacts/<branch-name>
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

# Initialize new orphan branch
git init
git config user.name "$GIT_USER_NAME"
git config user.email "$GIT_USER_EMAIL"
git config http.postBuffer 100m

# Copy cache files
echo "Copying cache files..."
cp -r "$OLDPWD/$SOURCE_CACHE_DIR" .jupyter_cache/

# Create README for the artifact branch
cat > README.md <<EOF
# Documentation Artifacts: $BRANCH_NAME

This orphan branch stores build artifacts for the \`$BRANCH_NAME\` branch.

## Contents

- \`.jupyter_cache/\`: Cached Jupyter notebook execution results

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

# Add and commit
git add .
git commit -m "Update docs cache for $BRANCH_NAME

Generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Files: $FILE_COUNT
Size: $CACHE_SIZE"

# Push to artifact branch (force push since it's orphan)
echo "Pushing to $ARTIFACT_BRANCH..."
git push --force "$REPO_URL" HEAD:"refs/heads/$ARTIFACT_BRANCH"

cd "$OLDPWD"

echo "================================================"
echo "✓ Cache pushed successfully to $ARTIFACT_BRANCH"
echo "================================================"
