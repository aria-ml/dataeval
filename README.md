# Documentation Artifacts: docs/cleanup

This orphan branch stores build artifacts for the `docs/cleanup` branch.

## Contents

- `.jupyter_cache/`: Cached Jupyter notebook execution results
- `notebooks/`: Generated .ipynb files for Google Colab

## Purpose

This branch is automatically managed by CI/CD pipelines to:
- Speed up documentation builds by reusing cached notebook executions
- Avoid committing large binary files to the main branch
- Persist cache across CI runs

## Last Updated

2026-02-13 19:49:02 UTC

Branch: docs/cleanup
Files: 23
Size: 35M

---
**Note:** This is an orphan branch with no history connection to main.
Do not merge this branch into main or other working branches.
