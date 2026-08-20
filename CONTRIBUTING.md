# Contributing

Thank you for your interest in being part of our community of active
developers!

## Development Setup

DataEval uses [uv](https://docs.astral.sh/uv/) for environment management and
[nox](https://nox.thea.codes/) as its task runner. `uv` is the only thing you need
installed up front — it fetches everything else on demand.

### Bootstrapping an environment

```bash
git clone https://github.com/aria-ml/dataeval.git
cd dataeval
uvx --with nox-uv nox -s dev
```

That builds `.venv` with DataEval and every development dependency. Run with no
arguments it prompts for the Python version and the PyTorch build; pass them as
arguments to skip the prompts:

```bash
uvx --with nox-uv nox -s dev -- --python 3.12 --device cu130
```

| Flag | Values | Default |
| ---- | ------ | ------- |
| `-p`, `--python` | `3.10` – `3.14` | `3.11` |
| `-d`, `--device` | `cpu`, `cu126`, `cu130` | `cu130` |
| `-n`, `--name` | any directory | `.venv` |

The chosen device is recorded in `.cuda-version`, which later sessions read so they
build against the same PyTorch variant.

Activate the result and you are ready to work:

```bash
source .venv/bin/activate
```

:warning: **Bootstrap with `uvx`, not `uv run`.** `uv run nox -s dev` would run nox
out of the very environment the session is about to delete and rebuild; the session
detects this and refuses to start. `uvx` fetches a throwaway nox instead, so there
is nothing to pull out from under. Adding `--with nox-uv` gives nox its uv-backed
session environments, which is what the remaining sessions expect.

### Running checks

Every task is a nox session — `uvx --with nox-uv nox -l` lists them. The common ones:

```bash
uv run nox -s test      # unit tests with coverage
uv run nox -s lint      # ruff, codespell, typos
uv run nox -s type      # pyright and type-completeness
uv run nox -s doctest   # docstring examples
uv run nox -s docs      # build the documentation
```

Once `.venv` exists, `uv run nox ...` is the convenient form for everything except
`dev` itself; `uvx --with nox-uv nox ...` works from anywhere and needs no project
environment at all.

## How Can I Contribute?

### Reporting Bugs

Bug reports can be submitted in several forms. Here are some general guidelines
to keep in mind when submitting a bug report for consideration.

#### Crafting a Bug Report

The bug report should be in the following format and contain as much detail as
possible.

```text
Steps to Reproduce:
 1. 
 2.
 3.
 ...

Expected Behavior:

Actual Behavior:

Frequency of Behavior:
```

#### Submitting a Bug Report

Bugs are tracked via issues in our internal GitLab repository, but issues can
be reported via GitHub or by emailing us at <dataeval@ariacoustics.com>. For
issues created in GitHub, please follow the above bug report template.

#### Making it Good(tm)

Bugs can be notoriously difficult to pin down and eliminate, but following the
tips below can help the maintainers do the best they can.

- Use a clear and descriptive title
- Describe the exact steps (before and during) which led to the issue
- Provide specific examples (such as data inputs or models used)
- Describe the behavior observed after following each step
- Explain what was the expected behavior compared to what was observed
- Include full callstacks and error messages when possible

### Suggestions for Improvement

We are always excited to hear back from our users for ideas for new features
and/or improvement of existing features and workflows.

Feel free to reach out to <dataeval@ariacoustics.com> as we would love to hear
from you.
