# Branching and Release Workflow

Development is done using trunk-based development strategy.  Feature development is done on short-lived branches against `main` branch and merged in to `main` as frequently as possible, with the idea that `main` is ready to ship at any interval.  If changes are not ready to be released, they should not be part of the public facing API and encapsulated in the `_prototype` subpackage.

As development nears completion, the code should move to the `_internal` subpackage and be exposed through our public facing API.  Docstrings, tutorials, how-tos, and other relevant deliverables should be prepared along-side.

After merge requests in to `main` are completed, additional validation in the form of functional tests are run.  On post-merge commit pipeline success, the build is tagged with `latest-known-good`. On a weekly cadence, a branch is created off of `main` to `releases/vX.X.X` and triggers release to PyPI and ReadTheDocs.

Hotfixes will be treated as fast track releases, where we trigger the release pipeline manually rather than waiting the scheduled weekly trigger.

## Branching Diagram
![image info](.gitlab/branching.png)
<!--- Code for mermaid gitGraph
%%{ init: { 'gitGraph': { 'mainBranchOrder': 1 } } }%%
gitGraph
    commit
    branch short_lived_dev_1 order: 2
    checkout short_lived_dev_1
    commit
    checkout main
    merge short_lived_dev_1
    commit id:"changelog_v0.1.0" tag:"v0.1.0"
    branch short_lived_dev_2 order: 3
    checkout short_lived_dev_2
    commit
    checkout main
    branch hotfix_1 order: 0
    checkout hotfix_1
    commit
    checkout main
    merge hotfix_1
    commit id:"changelog_v0.1.1" tag:"v0.1.1"
    branch short_lived_dev_3 order: 4
    checkout short_lived_dev_3
    commit
    checkout short_lived_dev_2
    commit
    checkout main
    merge short_lived_dev_2
    commit id:"changelog_v0.2.0" tag:"v0.2.0"
    checkout short_lived_dev_3
    commit
-->

## [Gitlab CI Pipelines](.gitlab-ci.yaml)
- On merge requests to `main` run the following jobs:
  - `build`: builds container images and verifies dependency lock file
  - `linting`: static analysis of code
  - `dependency tests`: tests against installations with no extras
  - `docs`: build and output artifacts for documentation
  - `unit tests`->`coverage`: run unit tests and generate code coverage report
- On commit to `main` for completed merge requests additionally run:
  - `functional tests`: superset of `unit tests` with additional slower functional tests using the GPU runner
  - `tag release candidate`: tags a successful run with `latest-known-good`
- On weekly scheduled release pipelines **only** run:
  - `create release`: updates `docs/.jupyter_cache/` and `CHANGELOG.md` and tags the commit with vNext

## [Github Actions](.github/workflows/publish.yml)
- This action publishes DAML to [PyPI](https://pypi.org/project/daml/) when new version tags are pushed

## [ReadTheDocs Pipeline](.readthedocs.yaml)
- This pipeline publishes documentation to [ReadTheDocs](https://daml.readthedocs.io/) on updates to `main` branch (latest) and new release tags (stable)
