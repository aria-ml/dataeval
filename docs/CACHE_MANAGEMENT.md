# Documentation Cache Management

This directory contains scripts for managing Jupyter notebook execution cache using Git orphan branches.

## Overview

To keep the main branch clean and avoid committing large binary files, we store `.jupyter_cache` artifacts in separate
orphan branches following the pattern `docs-artifacts/<branch-name>`.

### Benefits

- **Smaller repository**: Main branch doesn't contain large cache files
- **Persistent cache**: Cache survives across CI runs (no 7-day expiration)
- **Faster builds**: Reuse cached notebook execution results
- **Branch isolation**: Each branch has its own cache

## Architecture

```text
main branch (code only)
  ↓
  CI builds docs
  ↓
docs-artifacts/main (orphan branch, cache only)
```

### Orphan Branches

Orphan branches have no parent commits and are disconnected from main branch history:

- `docs-artifacts/main` - Cache for main branch
- `docs-artifacts/release/1.0` - Cache for release/1.0 branch
- `docs-artifacts/feature/xyz` - Cache for feature branches (optional)

**Important**: These branches should NEVER be merged into working branches.

## Scripts

### `fetch-docs-cache.sh`

Fetches the `.jupyter_cache` from the artifact branch before building documentation.

**Usage:**

```bash
./docs/fetch-docs-cache.sh [branch-name]
```

**Environment variables:**

- `CI_COMMIT_REF_NAME` (GitLab) or `GITHUB_REF_NAME` (GitHub) - Auto-detected in CI
- Falls back to current branch name if not in CI

**Behavior:**

- If artifact branch exists: Extracts cache to `docs/source/.jupyter_cache/`
- If artifact branch doesn't exist: Continues gracefully (first-time build)

### `push-docs-cache.sh`

Pushes the generated `.jupyter_cache` to the artifact branch after building documentation.

**Usage:**

```bash
./docs/push-docs-cache.sh [branch-name]
```

**Environment variables:**

- `CI_REPOSITORY_URL` (GitLab) or `GITHUB_TOKEN` (GitHub) - For authenticated push
- `GIT_AUTHOR_NAME`, `GIT_AUTHOR_EMAIL` - Git commit metadata

**Behavior:**

- Creates/updates orphan branch with latest cache
- Includes metadata (timestamp, file count, size) in commit message
- Force pushes (safe because orphan branch has no shared history)

## CI/CD Integration

### GitHub Actions

The `docs.yml` workflow:

1. **Checkout** with full history (`fetch-depth: 0`)
1. **Build docs** with nox (nox internally calls `fetch-docs-cache.sh`)
1. **Push cache** is disabled (CI is primarily operated on GitLab)

```yaml
permissions:
  contents: write # Required to push to artifact branches
```

### GitLab CI

The `docs` job in `.gitlab/ci/docs.yml`:

1. **before_script**: Install dependencies (uv, nox)
1. **script**: Build docs with nox (nox internally calls `fetch-docs-cache.sh`)
1. **after_script**: Push cache using `push-docs-cache.sh` (only on main/release branches)

```yaml
variables:
  GIT_STRATEGY: clone # Full clone needed for artifact branch operations
```

## Cache Policy

### When Cache is Pushed

Cache is pushed to artifact branches only for:

- `main` branch
- `release/*` branches

This prevents proliferation of artifact branches for short-lived feature branches.

### Cache Invalidation

The cache is automatically updated on every docs build for tracked branches. To manually clear cache:

```bash
# Delete the artifact branch
git push origin --delete docs-artifacts/main

# Or delete locally and remotely
git branch -D docs-artifacts/main
git push origin :docs-artifacts/main
```

The next CI run will create a fresh cache.

## Local Usage

### Fetch cache for local builds

```bash
# Fetch cache for current branch
./docs/fetch-docs-cache.sh

# Fetch cache for specific branch
./docs/fetch-docs-cache.sh main
```

### Push cache after local build

```bash
# Build docs first
uvx nox -e docs

# Push cache for current branch
./docs/push-docs-cache.sh

# Push cache for specific branch
./docs/push-docs-cache.sh main
```

**Note**: Pushing requires write access to the repository.

## Troubleshooting

### "Artifact branch does not exist yet"

This is normal for first-time builds. The cache will be created after the first successful docs build.

### "Cache directory not found"

The docs build failed or didn't generate a cache. Check the build logs for errors.

### Permission denied when pushing

Ensure the CI has proper permissions:

- **GitHub**: `permissions: contents: write` in workflow
- **GitLab**: Repository settings > CI/CD > Enable "Push to repository"

### Large artifact branches

If artifact branches grow too large:

```bash
# Check size
git clone --single-branch --branch docs-artifacts/main <repo-url> cache-check
du -sh cache-check

# If too large, delete and recreate
git push origin --delete docs-artifacts/main
```

## Implementation Details

### Why orphan branches?

- **Isolated history**: No commit ancestry, can't accidentally merge
- **Simple structure**: Just cache files, no code
- **Garbage collection**: Can delete entire branch without affecting main
- **Atomic updates**: Each push is a complete snapshot

### Why force push?

Orphan branches are force-pushed because:

- They have no shared history to preserve
- We only need the latest cache, not historical versions
- Force pushing prevents branch bloat

### Security

- Scripts validate branch existence before operations
- Only writes to `docs-artifacts/*` branches
- Requires explicit CI permissions configuration
- Local usage requires repository write access
