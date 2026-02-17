# Release and Documentation Workflow

## Notebook Architecture

Notebooks live in `docs/source/notebooks/` as paired `.md` and `.ipynb` files managed by
[jupytext](https://jupytext.readthedocs.io/). The pairing is configured in
`docs/source/notebooks/jupytext.toml`.

- **`.md` files** are the committed source of truth (MyST Markdown format)
- **`.ipynb` files** are gitignored and generated locally for editing in VS Code / Jupyter
- **Google Colab** links reference `.ipynb` files stored on `docs-artifacts/*` orphan branches

## Working with Notebooks

### Initial Setup (after cloning)

```bash
nox -e docsync    # generates .ipynb files from .md source
```

### Creating a New Notebook

1. Create `docs/source/notebooks/<name>.ipynb` in VS Code or Jupyter
2. Edit cells and run them
3. Run `nox -e docsync` to generate the `.md` pair and format it
4. Commit only the `.md` file

### Editing an Existing Notebook

1. Open the `.ipynb` in VS Code or Jupyter
2. Make changes, save
3. Run `nox -e docsync` to sync changes back to `.md`
4. Commit the `.md` changes

### Editing Markdown Directly

1. Edit the `.md` file in any editor
2. Run `nox -e docsync` to regenerate the `.ipynb`

### How `docsync` Works

`nox -e docsync` runs three steps in order:

1. **Orphan detection** -- generates `.md` for any new `.ipynb` without a markdown pair
2. **Bidirectional sync** -- `jupytext --sync` updates whichever side is stale based on
   file modification timestamps
3. **Format and regenerate** -- `mdformat` formats the markdown, then `jupytext` regenerates
   `.ipynb` to match the formatted `.md`

## Documentation Build

`nox -e docs` builds the full documentation site:

1. Converts `.md` notebooks to `.ipynb` via jupytext
2. Fetches cached notebook execution results from `docs-artifacts/<branch>` orphan branches
   (`docs/fetch-docs-cache.sh`)
3. Runs `sphinx-build` with MyST-NB (executes notebooks on cache miss)
4. Pushes updated cache and generated `.ipynb` files to the artifact branch
   (`docs/push-docs-cache.sh`)

## Artifact Branches (`docs-artifacts/*`)

Orphan branches named `docs-artifacts/<ref>` store build artifacts that are too large or too
transient for the main branch:

```text
docs-artifacts/<ref>/
  .jupyter_cache/    # cached notebook execution results (speeds up builds)
  notebooks/         # generated .ipynb files (for Google Colab links)
  README.md
```

### Branch lifecycle

| Branch                            | Created by                                       | Cleaned up by                                     |
| --------------------------------- | ------------------------------------------------ | ------------------------------------------------- |
| `docs-artifacts/main`             | Docs CI on every main commit                     | Never (always kept)                               |
| `docs-artifacts/<feature-branch>` | Docs CI on MR builds                             | `remove_docs_artifact_branches.py` after MR merge |
| `docs-artifacts/<version-tag>`    | `push-docs-cache.sh` when HEAD has a version tag | Never (preserved for Colab links)                 |

> **Note:** Release branches (`release/v*`) do **not** create `docs-artifacts/release/v*` branches.
> They only push to `docs-artifacts/<tag>` once the patch release is tagged.

### Colab links

In the checked-in documentation, Colab links point to `docs-artifacts/main`:

```text
https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/main/notebooks/<file>.ipynb
```

During a release, `releasegen.py` rewrites these to the versioned artifact branch:

```text
https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/v0.96.0/notebooks/<file>.ipynb
```

## Release Process

Releases are managed via GitLab CI scheduled pipelines and mirrored to GitHub.

### Full Release

Triggered by setting `CREATE_NEW_RELEASE=true` in a scheduled pipeline:

1. **`create_release.py`** analyzes merged MRs since the last tag, determines the version
   bump (major/minor), updates `CHANGELOG.md`, rewrites Colab links to the new version, and
   commits to `main`
2. The commit is **tagged** `vX.Y.Z`
3. An **API pipeline** is triggered on `main` which builds docs, then
   `push-docs-cache.sh` detects the version tag on HEAD and pushes artifacts to both
   `docs-artifacts/main` and `docs-artifacts/vX.Y.Z`
4. The tag push triggers **GitHub Actions** (`publish.yml`) which builds and publishes to PyPI
   and creates a GitHub Release

### Pre-Release

Triggered by setting `CREATE_PRE_RELEASE=true`:

- Creates a pre-release tag like `v1.0.0-rc0` on main
- Updates changelog and Colab links to the pre-release version
- An API pipeline on main creates `docs-artifacts/v1.0.0-rc0`

### On-Demand Release Branch Creation

Release branches are created on-demand when a patch is needed for an older version, rather
than being created automatically at release time. This keeps the repository clean and avoids
accumulating unused branches.

To create a release branch:

1. Go to GitLab > CI/CD > Run Pipeline
2. Set variable `CREATE_RELEASE_BRANCH=vX.Y` (e.g., `CREATE_RELEASE_BRANCH=v1.2`)
3. The pipeline runs **`create_release_branch.py`**, which finds the latest `vX.Y.*` tag
   and creates `release/vX.Y` from that tag's commit

Once the branch exists:

- Cherry-pick fixes or create MRs targeting `release/vX.Y`
- Future `release::fix` merges to main will auto-cherry-pick to this branch
- When fixes merge, `create_patch_release.py` auto-tags the next patch version

### Patch Release

Triggered automatically on commits to `release/v*` branches:

- Only accepts `release::fix` labeled MRs
- Calculates next patch version (e.g., `v1.2.0` to `v1.2.1`)
- Updates changelog and tags the release branch

### MR Labels

Every MR targeting `main` must have a release label:

| Label                  | Version bump                  | Description               |
| ---------------------- | ----------------------------- | ------------------------- |
| `release::major`       | Major                         | Breaking changes          |
| `release::feature`     | Minor                         | New functionality         |
| `release::improvement` | Minor                         | Enhancements              |
| `release::deprecation` | Minor                         | Deprecations and removals |
| `release::fix`         | Patch (release branches only) | Bug fixes                 |
| `release::misc`        | Minor                         | Other changes             |

## CI Jobs

### Documentation stage

| Job            | Trigger                                | Purpose                                |
| -------------- | -------------------------------------- | -------------------------------------- |
| `docs`         | Main commits, MRs with doc/src changes | Full docs build with GPU, pushes cache |
| `doclint`      | Main commits, MRs                      | Extracts and lints notebook code       |
| `doctest`      | Main commits, MRs                      | Runs doctests                          |
| `linkchecker`  | Main commits, MRs                      | Validates markdown links               |
| `markdownlint` | Main commits, MRs                      | Lints markdown formatting              |

### Release stage

| Job                             | Trigger                                    | Purpose                                            |
| ------------------------------- | ------------------------------------------ | -------------------------------------------------- |
| `create release`                | Scheduled (`CREATE_NEW_RELEASE`)           | Creates version tag on main                        |
| `create pre-release`            | Scheduled (`CREATE_PRE_RELEASE`)           | Creates pre-release tag on main                    |
| `create release branch`         | Web UI (`CREATE_RELEASE_BRANCH=vX.Y`)      | Creates release branch from latest tag on demand   |
| `create patch release`          | Commits to `release/v*`                    | Creates patch version tag                          |
| `remove docs artifact branches` | Main commits                               | Cleans up artifact branches for merged MRs         |
| `cherry-pick fixes to releases` | Main commits                               | Auto-cherry-picks fixes to active release branches |

## Key Files

| File                                               | Purpose                                                                |
| -------------------------------------------------- | ---------------------------------------------------------------------- |
| `noxfile.py`                                       | Build orchestration (`docs`, `docsync`, `doclint`, `doctest` sessions) |
| `docs/source/notebooks/jupytext.toml`              | Configures md/ipynb pairing                                            |
| `docs/push-docs-cache.sh`                          | Pushes cache + notebooks to artifact branches                          |
| `docs/fetch-docs-cache.sh`                         | Fetches cached results from artifact branches                          |
| `docs/check_notebook_cache.py`                     | Validates and cleans jupyter cache state                               |
| `docs/source/conf.py`                              | Sphinx configuration (MyST-NB, cache settings)                         |
| `.gitlab/ci/docs.yml`                              | Documentation CI jobs                                                  |
| `.gitlab/ci/release.yml`                           | Release CI jobs                                                        |
| `.gitlab/scripts/releasegen.py`                    | Generates changelog and updates Colab links                            |
| `.gitlab/scripts/create_release.py`                | Orchestrates full releases                                             |
| `.gitlab/scripts/create_release_branch.py`         | Creates release branches on-demand from tags                           |
| `.gitlab/scripts/remove_docs_artifact_branches.py` | Cleans up stale artifact branches                                      |
