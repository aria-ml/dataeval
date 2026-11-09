# Working on DataEval: practices for agents

This file records how work is done in this repository, for coding agents and for the people who direct them. It covers
research, feature development, integration, API design, documentation and DevSecOps. Each practice comes from a real
decision or a real failure in this project.

The detailed rules live in their own files. This file points to them and does not repeat them:

- [CONTRIBUTING.md](CONTRIBUTING.md): environment setup and the nox sessions.
- [BRANCHING.md](BRANCHING.md): branches, commit prefixes, releases and backports.
- [docs/STYLE_GUIDE.md](docs/STYLE_GUIDE.md): how documentation, notebooks and docstrings are written.
- [docs/RELEASE.md](docs/RELEASE.md): notebooks, the docs build and the release jobs.

When the maintainer's instructions and this file disagree, follow the maintainer.

## Ground rules

- Ask before any action that is hard to reverse or that other people will see. That covers pushing, opening a merge
  request, filing an issue, posting a comment, deleting a branch or a worktree, and committing straight to `main`. A
  request for one of these covers that one action, not the next.
- Do not merge. The maintainer merges.
- Do not force-push unless asked. When asked, use `--force-with-lease`.
- Report what happened. If a test fails, say so and show the output. If you skipped a step, say which one.
- Never print a token or a secret, and never put one in a command line, a URL or a log.
- Never run a bare `git stash`. Other sessions share the stash. Set work aside with a temporary commit instead.
- Write US English everywhere, including code comments.

## Research

DataEval makes claims about data. A claim ships only with the evidence that backs it: the method or theorem, the scope
it holds in, its assumptions and its confidence.

- Argue a design from whether its result is valid at the stated confidence and scope. Do not argue it from the verdicts
  you expect it to give.
- Measure on real data first. Simulate only what the real data lacks, build the simulation from real images, and say
  in the result that it is simulated. A result that finds nothing on clean data teaches nothing, so add a known defect
  before you show a detector.
- Calibrate a detector before you claim it works. Measure its false alarms when nothing changed, on resampled or
  held-out data rather than one split. Then measure its power at the parameter that ships, not at the one that looked
  best.
- Say what the evidence does not cover. If a study used an oracle, such as true scene labels, name it. If the data has
  no example of a case, say the study cannot speak to that case.
- Fail closed. When a check cannot show that an estimate is safe, require the input from the user, or refuse, rather
  than guess.
- Do not state a cause you did not trace. Write what you measured.
- Treat a constant chosen on study data as provisional. When you retune one, record the sweep that chose it.
- Check a reviewer's finding against the code before you act on it. Reproduce a counterexample before you change a
  design around it.
- Check results that can depend on the machine or the versions: CPU against GPU, and the lowest supported dependencies.
- Keep a spike on a branch that never merges. Bring what it taught you back into the design.

### Validation records

A study that backs a public claim must be reproducible by someone else.

- Keep the study scripts, the saved outputs, the data checksums, a log of corrections and a table of where each
  constant came from. Regenerate every headline number from saved outputs with saved scripts, and run the study twice
  to check that it is deterministic.
- Store the record in the program meta repo under `DataEval/validation-studies/<date>-<study>/`, not in this
  repository. This repository is public. Its pages describe each study's data, method and results in their own words
  and never link to the meta repo.
- Requirements and their test cases live in [verification/registry.yaml](verification/registry.yaml). When a feature
  changes what a requirement covers, update its steps and run `nox -s verify`.

### Untrusted data

Downloaded footage, datasets and archives are untrusted.

- Put each download in its own new, empty directory, and keep your scripts in a different one.
- Run any Python that reads them with `python -I`, and pass their paths as arguments. An interpreter started inside a
  download directory can import a planted module.

## Feature development

### Design first

Anything larger than a small change starts with a design that the maintainer approves before any code is written.

- Write the design to `specs/<YYYY-MM-DD>-<topic>-design.md`. The `specs/` folder is ignored by git, and designs are
  never committed.
- Cover the intent, the goals, the non-goals and the anti-goals, the current state, the decisions, the acceptance
  tests, the delivery by merge request and the risks. Mark which decisions the maintainer made and which you made.
- Check every claim about the current code against the code, and cite the file and line.
- Have a fresh reader review the design before planning. Bring the questions only the maintainer can answer to the
  maintainer in one message.
- Before any work starts, run two agents against the design, each with its own brief. Give them the design and the
  docs it touches, and keep your own verdict out of the brief.
  - One challenges the design on simplicity and usability: whether a simpler shape serves the same users, and whether
    each flag, option, column and concept earns its place.
  - The other challenges the documentation the design plans: whether it is excessive, whether each piece sits at the
    right level, and whether each level fits its audience: how-to, tutorial, docstring and concept page, as
    [docs/STYLE_GUIDE.md](docs/STYLE_GUIDE.md) describes them. Concise and not over-explaining is a strength.
- Write the plan to `specs/` as well, as tasks that each carry their own tests.

### Build

- Work in a worktree on its own branch. The shared `.venv` imports the main checkout's source, so a worktree needs its
  own environment and its own ignore file for `specs/`:

  ```bash
  uv sync --frozen --extra cpu --extra opencv --extra onnx --group test-onnx --group lint --group type
  printf '**\n' > specs/.gitignore
  ```

- Write the failing test first, watch it fail for the right reason, then write the code.
- Fix the root cause in the shared function that every caller goes through, not in the one caller a report names.
- Make the smallest change that works. When you take a shortcut with a known limit, mark it with a `ponytail:` comment
  that names the limit and the upgrade path, as existing code does.
- A new option that is not passed must leave every result bit-identical to before. Test that.
- Make results deterministic. Define the order of inputs and how ties break before any step that depends on them, and
  test that reordering the input permutes the output and changes nothing else.
- Give CI synthetic fixtures with the properties the test needs, and state those properties. Run long studies on real
  data as offline validation runs, and report them in the merge request.
- Mark tests that need ffmpeg-generated content with `pytestmark = pytest.mark.ffmpeg`.
- Keep behavior that depends on a library version out of shared fixtures. Isolate it in its own test with a skip
  condition that probes the installed version.
- Split a large feature into merge requests that each carry their own acceptance tests. A new feature usually ships as
  experimental first (see API design).

### Verify

Run the sessions CI runs, one at a time. Several at once can run out of memory.

- `nox -s test` runs the unit and integration tests, including the ffmpeg tests, and skips the `cuda` ones.
- `nox -s type` runs pyright over the whole project, tests and notebooks included, on every supported Python, and
  fails if public type completeness falls to 90% or below. A clean `pyright src` is not enough.
- `nox -s deps` runs the unit tests on the oldest Python with the lowest direct dependencies, such as polars 1.0.0.
- `nox -s lint` runs ruff, codespell and typos. It also reformats files, so check `git status` afterwards.
- `nox -s doctest` runs the docstring examples and the documentation tests in `docs/tests`. `nox -s test` does not
  run those.
- `nox -s check` confirms the lock file and the exported requirement files match `pyproject.toml`.
- `nox -s docs` builds the documentation. Run it when you change a notebook.
- `npx markdownlint-cli2 '*.md' 'docs/**/*.md'` lints the Markdown, which `nox -s lint` does not.

## Integration

- Every branch reaches `main` through a GitLab merge request, unless the maintainer asks for a direct commit.
- Title the merge request `[tag] Summary`. CI rejects an unknown tag. The title becomes the squash commit and the
  changelog entry, so write it to stand alone as a release note. The tags are listed in [BRANCHING.md](BRANCHING.md).
- Give each merge request one purpose. Set it to squash and to remove the source branch.
- Write the description for the reviewer: what changed and why, under bold section headings, with bullets, and the
  checks you ran with their results. Leave out template filler and any attribution line.
- Before you change a description or a note that someone wrote, save a copy, and change it by exact replacement.
- Write each commit message as one line, `[tag] Description`, with no body and no trailers.
- If `git config commit.gpgsign` is `true`, check that the signing key is unlocked before you commit. A commit from a
  non-interactive session hangs when the passphrase is not cached. Test it with
  `echo test | timeout 10 gpg --batch --pinentry-mode error -bsau <key> >/dev/null`, using the key from
  `git config user.signingkey`. If it fails, ask the maintainer to unlock the key. Do not commit unsigned, and do not
  turn signing off, without their permission for that commit.
- Stack dependent merge requests, each targeting the one before it. After a squash merge, move the next branch onto
  `main` with `git rebase --onto origin/main <old predecessor head> <branch>`.
- A `[fix]` may need a backport. Cherry-pick only the fix onto the release branch, not the extensions around it.
- Watch the pipeline to the end. Tell a runner problem, such as a pod that never scheduled or pyright exiting with
  code 250 when it runs out of heap, from a real failure. Retry the first, and fix the second.
- A change to a notebook's code cells makes the CI docs job execute that notebook again. A heavy notebook can pass the
  one-hour limit. Build the docs locally, push the cache with `docs/push-docs-cache.sh`, then retry the job.
- When an API change affects dataeval-flow, open the flow merge request alongside this one.
- Before a release, check what the docs claim against what ships: the roadmap, the Stability status list, the
  migration notes and the "New in" page.

### Tracking work

Development is tracked in the GitLab group that holds this project. Epics represent programs and features, and issues
represent the work itself. Work is scheduled into program increments (PIs), which are GitLab milestones, and into
sprints, which are GitLab iterations.

- A program is an epic labeled `epic::program`. It has no parent. Its milestone is the PI it is committed to.
- A feature is an epic labeled `epic::feature`. It sits under one program and takes that program's PI. A feature never
  sits under another feature.
- An issue carries the `ARiA` label and sits under one feature, never directly under a program. Its milestone is
  always its feature's PI, whether it is in a sprint or in the backlog.
- An issue committed to a sprint has the iteration, a weight of 1, 2, 3 or 5, and an assignee. Split an issue of
  weight 8 or more before sprint planning.
- File a new issue on its own, with the `ARiA` label only, and no milestone, iteration or parent. The maintainer places
  it under a feature and schedules it during backlog grooming, and the issue then takes its PI from the feature.
- File issues only for committed or planned work, not for ideas or exploration. A design names the issues it needs,
  and they are filed when the build starts.
- Reference the issue in the merge request description with `Closes #<number>`, so that merging closes it.
- Record progress that spans several issues, such as a design decision or a validation result, as a note on the
  feature or program epic.
- Do not close an item you created by mistake to tidy it away. Tell the maintainer, since closed items stay in
  burndowns and reports.

## API design

### Two layers

- A core function in `dataeval.core` is stateless. It takes everything it uses as an explicit argument and returns raw
  values in a `*Result` typed dictionary.
- An evaluator is the opinionated layer. It can read what is attached to the `Metadata` it is given, behaves as before
  when nothing is attached, and lets an explicit argument win over attached values.
- Keep a shared output table general across tasks. Put task-specific values in a `details` field.
- DataEval reports what it finds about a dataset. It does not change datasets for training.

### Public surface

- A private module (`_name.py`) declares `__all__ = []`. Public names are exported only from the package's
  `__init__.py`, listed in alphabetical order. Some older modules break this rule. Do not copy them.
- When a researcher needs to replace one piece of the math, expose it as a small `runtime_checkable` protocol in
  `dataeval.protocols` with a default implementation, as `FeatureExtractor` and `SharpnessMeasure` do. Nothing after
  that point may depend on which implementation is used.
- Keep the terms of the method's source literature, and explain them in plain prose. Do not rename them.

### Stability

- Mark a new function or class `@experimental`. Mark a flag or an enum member with `warn_experimental_once`, since the
  decorator cannot mark it.
- Mark a typed dictionary or a protocol experimental with a `.. warning::` in its docstring. The decorator sets an
  attribute on the class, and on Python 3.11 a `runtime_checkable` protocol counts that attribute as a member, which
  breaks `isinstance`.
- List every experimental item in the Stability status section of
  [ValidationAndTrust.md](docs/source/concepts/ValidationAndTrust.md) and in the release's migration notes.
- Deprecate before removing, with `@deprecated(since=..., removal=..., alternative=...)`. A renamed function keeps its
  old name until the next major release.

### Behavior

- Fail closed. Raise a clear error rather than guess, and offer an explicit opt-in for the risky assumption. An unknown
  value is NaN, not 0.
- Make an error say how to fix it. A missing column names the flag that produces it.
- Measure, do not trust. Reported metadata, such as a video's frame rate, is a claim. Measure the value from the data
  and report any difference.
- Do not lose data silently. Move an overwritten column aside, record dropped columns, and warn.
- A result addressed by position, through `source_index`, is only valid for the source it was computed on. Code that
  applies a result to another view must check the alignment or raise.
- Compare values across scales with relative tolerances, not raw differences.
- Read the data once. Compute every selected statistic in one pass, and benchmark before choosing between threads and
  processes.

## Documentation and style

[docs/STYLE_GUIDE.md](docs/STYLE_GUIDE.md) is the rulebook, and `docs/tests/test_notebook_structure.py` enforces part of
it. The practices below are the ones agents most often get wrong.

- A how-to (`h2_*`) is a short recipe for one task. A tutorial (`tt_*`) teaches, with visuals. Drop a how-to that a
  tutorial already covers, unless it shows something the tutorial does not. Link to concepts rather than explaining
  them again.
- Write tutorial prose that a general undergraduate can follow. Translate a technical statement into plain words and
  keep its qualifier, as in "at least 95.6% right, and you can be 95% sure of that". Keep the technical terms in
  docstrings, concept pages and printouts.
- Quote a number only after you check it against executed output, and protect it with an assertion cell.
- Do not describe a behavior you did not run. Do not describe planned work as a feature. Do not drop a hedge when you
  edit for style.
- Published notebook outputs can be older than the library. The notebook cache keys only on code cells. When a number
  in the docs looks wrong, run it again against the current code before you change the prose.
- Use `*` and `**` for emphasis, never `_`. Wrap Markdown at 120 columns. Put identifiers in backticks.
- Cite stable sources, such as PubMed, arXiv or a publisher's abstract page, rather than a personal or course page.
  Open each new link before you commit it.
- Never link the public docs to an internal repository or to `specs/`.

## DevSecOps

- Read tokens from the environment or a local file at the moment you use them. Never echo one.
- CI scans every merge request for secrets, vulnerable dependencies and code patterns (secret detection, dependency
  scanning and Semgrep on GitLab; TruffleHog, dependency review and Semgrep on GitHub). Fix a finding or document why it
  does not apply. Do not add a path to `DS_EXCLUDED_PATHS` or `SAST_EXCLUDED_PATHS` in
  [.gitlab/ci/security.yml](.gitlab/ci/security.yml) to make a finding go away.
- Releases publish to PyPI through Trusted Publishing with PEP 740 attestations
  ([.github/workflows/publish.yml](.github/workflows/publish.yml)). Do not add a long-lived upload token.
- Install from the lock file with `uv sync --frozen`. Tag dependency changes `[deps]`, and confirm `nox -s check` and
  `nox -s deps` pass.
- Do not disable a check, a scan or commit signing to get a pipeline to pass.
- The shared CI components are maintained outside this repository. Change them there, through their own merge request.
- Scope `git filter-repo --message-callback` with `--refs`. Without it, the callback rewrites the whole history.
- Do not wait on a background job with `while pgrep -f <pattern>`. The pattern matches the loop's own command line, so
  the loop never ends. Poll the job's output file or its API status instead.
