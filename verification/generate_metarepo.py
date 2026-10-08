#!/usr/bin/env python3
"""Generate meta repo artifacts from the verification registry and test results.

Reads the registry (``registry.yaml``), the pytest report
(``output/verification_report.json``), and, when available, CI job results
(``output/ci_jobs.json``) to produce under ``output/metarepo/``:

  - requirements/<id>-<slug>.md   one per requirement, rendered from the registry
  - test-cases/test-case-<id>.md  one per test case, with each step's result
  - vcrm.md                       requirement to test case matrix with verification row
  - test-results.log              run log for the dated assessment folder (DR-1.1-H-4, DR-1.3-H-1)

A test case is a list of steps. Each step names the pytest tests (``alias::test``
or a full node id), CI jobs (``CI: <job>``), or planned tests (``NEW: <what>``)
that give its evidence. A step passes when every automated piece passes. A step
with a planned test, or with CI evidence that is not available, is pending.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import yaml

VERIFICATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VERIFICATION_DIR.parent
REGISTRY_PATH = VERIFICATION_DIR / "registry.yaml"
REPORT_PATH = PROJECT_ROOT / "output" / "verification_report.json"
CI_JOBS_PATH = PROJECT_ROOT / "output" / "ci_jobs.json"
OUTPUT_DIR = PROJECT_ROOT / "output" / "metarepo"
PRODUCT = "DataEval"
DISTRIBUTION = "dataeval"

DR_15 = "https://jatic.pages.jatic.net/internal-docs/standards/product/documentation/program-doc-requirements/#dr-15-product-requirements-definitions"

# Step and test case outcomes.
PASS, FAIL, SKIP, PENDING = "passed", "failed", "skipped", "pending"


def load_registry() -> dict:
    """Load the verification registry."""
    return yaml.safe_load(REGISTRY_PATH.read_text())


def load_json(path: Path) -> dict | None:
    """Load a JSON file if it exists."""
    return json.loads(path.read_text()) if path.exists() else None


def slug(text: str) -> str:
    """File-name slug for a requirement name."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def requirement_filename(req: dict) -> str:
    """Requirement file name: ``FR-<n>-<slug>.md`` or ``NFR-<n>-<slug>.md`` (DR-1.5-H-2)."""
    return f"{req['id']}-{slug(req['name'])}.md"


# ---------------------------------------------------------------------------
# Evidence resolution
# ---------------------------------------------------------------------------


def expand(ref: str, aliases: dict[str, str]) -> str:
    """Expand ``alias::test`` to a full pytest node id."""
    head, sep, rest = ref.partition("::")
    return f"{aliases[head]}::{rest}" if sep and head in aliases else ref


def evidence_status(ref: str, nodes: dict[str, str] | None, ci_jobs: dict[str, str] | None) -> str:
    """Outcome of one piece of evidence."""
    if ref.startswith("NEW:"):
        return PENDING
    if ref.startswith("CI:"):
        job = ref[3:].strip()
        if not ci_jobs or job not in ci_jobs:
            return PENDING
        return {"success": PASS, "failed": FAIL, "skipped": SKIP}.get(ci_jobs[job], PENDING)
    if nodes is None:
        return PENDING
    if ref in nodes:
        return {"passed": PASS, "skipped": SKIP}.get(nodes[ref], FAIL)
    # A parametrized test runs as `<ref>[<id>]` per case: the bare reference stands for all of them.
    variants = [nodes[k] for k in nodes if k.startswith(ref + "[")]
    if not variants:
        return PENDING
    return combine([{"passed": PASS, "skipped": SKIP}.get(s, FAIL) for s in variants])


def combine(statuses: list[str]) -> str:
    """Overall outcome: any failure fails; any pending is pending; all skipped is skipped."""
    if FAIL in statuses:
        return FAIL
    if PENDING in statuses:
        return PENDING
    if statuses and all(s == SKIP for s in statuses):
        return SKIP
    return PASS


def step_statuses(
    tc: dict, aliases: dict[str, str], nodes: dict | None, ci_jobs: dict | None
) -> list[tuple[str, list[tuple[str, str]]]]:
    """For each step: (outcome, [(evidence, outcome), ...])."""
    out = []
    for step in tc["steps"]:
        ev = [(expand(r, aliases), evidence_status(expand(r, aliases), nodes, ci_jobs)) for r in step["tests"]]
        out.append((combine([s for _, s in ev]), ev))
    return out


def tc_outcome(tc: dict, aliases: dict, nodes: dict | None, ci_jobs: dict | None) -> str:
    """Outcome of a whole test case."""
    return combine([s for s, _ in step_statuses(tc, aliases, nodes, ci_jobs)])


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

_MARK = {PASS: "P", FAIL: "F", SKIP: "S", PENDING: "—"}
_LABEL = {PASS: "passed", FAIL: "failed", SKIP: "skipped", PENDING: "pending"}


def requirement_md(req: dict) -> str:
    """Render one requirement in the DR-1.5-R-1 shape."""
    kind = "Non-Functional" if req["id"].startswith("N") else "Functional"
    out = [
        f"# {req['id']}: {req['name']}",
        "",
        f"## {kind} Requirements",
        "",
        f"- Requirement ID: {req['id']}",
        f"  - Name: {req['name']}",
        f"  - Description: {req['description']}",
        "  - Acceptance Criteria",
    ]
    out += [f"    - {c}" for c in req["criteria"]]
    if req.get("notes"):
        out += [f"  - {req.get('notes_title', 'Reference Measurements (informative; not acceptance criteria)')}"]
        out += [f"    - {n}" for n in req["notes"]]
    return "\n".join(out) + "\n"


def test_case_md(tc: dict, aliases: dict, nodes: dict | None, ci_jobs: dict | None, today: str) -> str:
    """Render one test case in the DR-1.6-H-4 template."""
    steps = step_statuses(tc, aliases, nodes, ci_jobs)
    n = len(steps)
    overall = combine([s for s, _ in steps])
    lines = [
        f"# {tc['name']}",
        "",
        "## Description",
        "",
        f"- Test Type: {tc['type']}",
        f"- Business Case: {tc['business']}",
        "",
        "**Initial Conditions:**",
        "",
    ]
    lines += [f"{i}. {c}" for i, c in enumerate(tc["conditions"], 1)]
    lines += ["", "## Test Steps", ""] + [f"{i}. {s['do']}" for i, s in enumerate(tc["steps"], 1)]
    lines += [f"{n + 1}. Confirm the Expected Results by validating all steps pass.", "", "**Expected Results**", ""]
    lines += [f"{i}. {s['expect']}" for i, s in enumerate(tc["steps"], 1)]
    lines += ["", "## Test Results", "", "| Test Step |  Result | Notes |", "|:----------|:-------:|:------|"]
    for i, (status, _) in enumerate(steps, 1):
        lines.append(f"|{i:<10}|    {_MARK[status]}    |  [^{i}] |")
    lines += [f"|{n + 1:<10}|    {_MARK[overall]}    |  [^{n + 1}] |", ""]
    for i, (_, ev) in enumerate(steps, 1):
        parts = []
        for ref, status in ev:
            if ref.startswith("NEW:"):
                parts.append(f"{ref[4:].strip()}: not yet automated")
            elif ref.startswith("CI:"):
                parts.append(
                    f"CI job `{ref[3:].strip()}`: {'result not available' if status == PENDING else _LABEL[status]}"
                )
            else:
                parts.append(f"`{ref}`: {'not run' if status == PENDING else _LABEL[status]}")
        lines.append(f"[^{i}]: " + "; ".join(parts))
    lines += [f"[^{n + 1}]: Overall verification: {_LABEL[overall]}", "", f"**Last Updated Date:** {today}"]
    return "\n".join(lines) + "\n"


def tc_sort_key(tc_id: str) -> list[int]:
    """Sort test case ids like ``1-1`` and ``21-1`` numerically."""
    return [int(p) for p in tc_id.split("-")]


def vcrm_md(registry: dict, nodes: dict | None, ci_jobs: dict | None, today: str) -> str:
    """Render the VCRM."""
    aliases = registry.get("aliases", {})
    tcs = sorted(registry["test_cases"], key=lambda t: tc_sort_key(t["id"]))
    ids = [t["id"] for t in tcs]
    header = (
        "| Requirement ID | Requirement Origin | Coverage | "
        + " | ".join(f"[TC-{i.replace('-', '.')}][{i}]" for i in ids)
        + " |"
    )
    sep = (
        "| "
        + " | ".join([":--------------", ":-------------------", ":--------:"] + [":-------------:"] * len(ids))
        + " |"
    )
    rows, origin_links, req_links = [], {}, []
    for req in registry["requirements"]:
        mine = {t["id"] for t in tcs if t["req"] == req["id"]}
        ref = req["id"].lower().replace("-", "")
        origin = req.get("origin", "DR-1.5")
        oref = origin.lower().replace("-", "").replace(".", "")
        origin_links[oref] = f"[{oref}]:{req.get('origin_link', DR_15)}"
        req_links.append(f"[{ref}]:requirements/{requirement_filename(req)}")
        cells = ["X" if i in mine else " " for i in ids]
        rows.append(
            "| " + " | ".join([f"[{req['id']}][{ref}]", f"[{origin}][{oref}]", "Yes" if mine else "No", *cells]) + " |"
        )
    label = {PASS: "Pass", FAIL: "Fail", SKIP: "Skipped", PENDING: "Pending"}
    verification = [label[tc_outcome(t, aliases, nodes, ci_jobs)] for t in tcs]
    parts = [
        f"# {registry['product']} Verification Cross-Reference Matrix (VCRM)",
        "",
        header,
        sep,
        *rows,
        "| **Verification** | | | " + " | ".join(verification) + " |",
        "",
        f"**Last Updated:** {today}",
        "",
        "<!-- Links for Test Cases -->",
        "",
        *[f"[{i}]:test-cases/test-case-{i}.md" for i in ids],
        "",
        "<!-- Links for Requirement IDs -->",
        "",
        *req_links,
        "",
        "<!-- Links for Requirement Origins -->",
        "",
        *sorted(origin_links.values()),
    ]
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Test results log
# ---------------------------------------------------------------------------

LOG_WIDTH = 100
_LOG_STATUS = {"passed": "PASS", "failed": "FAIL", "error": "ERROR", "skipped": "SKIP", "pending": "PENDING"}


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(PROJECT_ROOT), *args], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def _source_line() -> str:
    """Describe the verified source: project URL, ref, commit, and whether it had local changes."""
    url = os.environ.get("CI_PROJECT_URL")
    if not url:
        url = _git("remote", "get-url", "origin") or "unknown"
        if url.startswith("git@"):
            url = "https://" + url.removeprefix("git@").replace(":", "/", 1)
        url = re.sub(r"//[^/@]+@", "//", url).removesuffix(".git")
    ref = (
        os.environ.get("CI_COMMIT_TAG")
        or _git("describe", "--tags", "--exact-match", "HEAD")
        or os.environ.get("CI_COMMIT_REF_NAME")
        or _git("rev-parse", "--abbrev-ref", "HEAD")
        or "unknown"
    )
    commit = (os.environ.get("CI_COMMIT_SHA") or _git("rev-parse", "HEAD") or "unknown")[:8]
    dirty = " [uncommitted changes]" if _git("status", "--porcelain", "--untracked-files=no") else ""
    return f"{url} @ {ref} ({commit}){dirty}"


def _product_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"


def _label(status: str) -> str:
    return _LOG_STATUS.get(status, status.upper())


def _counts(statuses: list[str]) -> str:
    found = {s: statuses.count(s) for s in _LOG_STATUS if statuses.count(s)}
    return ", ".join(f"{n} {s}" for s, n in found.items()) or "none"


def _detail_lines(cases: dict, outcome: dict, tests: dict) -> tuple[list[str], set[str]]:
    """Per-step evidence for every test case, and the set of evidence references it cited."""
    lines: list[str] = []
    cited: set[str] = set()
    for i in sorted(cases, key=tc_sort_key):
        t, steps = cases[i]
        lines.append(f"[test-case-{i}] {_label(outcome[i])}  {t['name']}")
        for n, (status, evidence) in enumerate(steps, 1):
            lines.append(f"  step {n}: {_label(status)}  {t['steps'][n - 1]['do']}")
            for ref, st in evidence:
                cited.add(ref)
                lines.append(f"      {_label(st):<8}{ref}")
                msg = tests.get(ref, {}).get("message")
                if msg:
                    lines.append(f"              -> {msg}")
        lines.append("")
    return lines, cited


def _requirement_status(rid: str, cases: dict, outcome: dict) -> str:
    mine = [outcome[i] for i, (t, _) in cases.items() if t["req"] == rid]
    if not mine:
        return "UNMAPPED"
    if all(s == PASS for s in mine):
        return "VERIFIED"
    return "FAILED" if FAIL in mine else "PARTIAL"


def test_results_log(registry: dict, report: dict, ci_jobs: dict | None) -> str:
    """Render the run log: summary, test case results, requirement coverage, and per-step evidence."""
    aliases = registry.get("aliases", {})
    run = report.get("run", {})
    tests = report.get("tests", {})
    nodes = {k: v["status"] for k, v in tests.items()}
    cases = {t["id"]: (t, step_statuses(t, aliases, nodes, ci_jobs)) for t in registry["test_cases"]}
    outcome = {i: combine([s for s, _ in steps]) for i, (_, steps) in cases.items()}
    reqs = registry["requirements"]

    rstatus = {r["id"]: _requirement_status(r["id"], cases, outcome) for r in reqs}
    verified = list(rstatus.values()).count("VERIFIED")
    rule, thin = "=" * LOG_WIDTH, "-" * LOG_WIDTH
    lines = [
        rule,
        f"{registry['product']} - Verification Test Results",
        rule,
        f"Product version    : {_product_version(registry.get('distribution', slug(registry['product'])))}",
        f"Source             : {_source_line()}",
        f"Runner             : {os.environ.get('CI_JOB_URL') or 'local run (not CI)'}",
        f"Command            : {run.get('command', 'unknown')}",
        f"Python             : {run.get('python', 'unknown')} ({run.get('platform', 'unknown')})",
        f"Started (UTC)      : {run.get('started', 'unknown')}",
        f"Finished (UTC)     : {run.get('finished', 'unknown')}",
        f"pytest exit status : {run.get('exit_status', 'unknown')}",
        "Standard           : DR-1.1-H-4, DR-1.3-H-1 (JATIC internal-docs v1.2.0)",
        "",
        thin,
        "SUMMARY",
        thin,
        f"Test cases         : {len(cases)} total, {_counts(list(outcome.values()))}",
        f"Tests              : {len(tests)} total, {_counts([t['status'] for t in tests.values()])}",
        f"Requirements       : {len(reqs)} total, {verified} verified (every test case passed)",
        "",
        thin,
        "TEST CASE RESULTS",
        thin,
        f"{'TEST CASE':<10}{'STATUS':<9}{'STEPS PASSED':<14}{'REQUIREMENT':<13}NAME",
    ]
    for i in sorted(cases, key=tc_sort_key):
        t, steps = cases[i]
        done = sum(s == PASS for s, _ in steps)
        lines.append(f"{i:<10}{_label(outcome[i]):<9}{f'{done}/{len(steps)}':<14}{t['req']:<13}{t['name']}")
    lines += ["", thin, "REQUIREMENT COVERAGE", thin, f"{'REQUIREMENT':<13}{'STATUS':<10}{'TEST CASES':<20}NAME"]
    for r in reqs:
        mine = sorted((i for i, (t, _) in cases.items() if t["req"] == r["id"]), key=tc_sort_key)
        lines.append(f"{r['id']:<13}{rstatus[r['id']]:<10}{', '.join(mine) or '-':<20}{r['name']}")
    detail, cited = _detail_lines(cases, outcome, tests)
    lines += ["", thin, "TEST DETAIL", thin, *detail]
    lines += [thin, "TESTS NOT CITED BY A TEST CASE", thin]
    uncited = sorted(n for n in tests if n not in cited)
    for n in uncited:
        lines.append(f"  {_label(tests[n]['status']):<8}{n}")
    if not uncited:
        lines.append("  (none)")
    lines += ["", rule, "END OF REPORT", rule]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def check_registry(registry: dict) -> None:
    """Fail loudly on a registry that cannot render a consistent matrix."""
    req_ids = {r["id"] for r in registry["requirements"]}
    tc_ids = [t["id"] for t in registry["test_cases"]]
    if len(set(tc_ids)) != len(tc_ids):
        raise SystemExit("registry error: duplicate test case id")
    for t in registry["test_cases"]:
        if t["req"] not in req_ids:
            raise SystemExit(f"registry error: test case {t['id']} names unknown requirement {t['req']}")
        if not t["steps"]:
            raise SystemExit(f"registry error: test case {t['id']} has no steps")
    uncovered = req_ids - {t["req"] for t in registry["test_cases"]}
    if uncovered:
        raise SystemExit(f"registry error: requirements without a test case: {sorted(uncovered)}")


def main() -> None:
    """Write requirements, test cases, and the VCRM to ``output/metarepo``."""
    registry = load_registry()
    check_registry(registry)
    aliases = registry.get("aliases", {})
    report = load_json(REPORT_PATH)
    nodes = {k: v["status"] for k, v in report["tests"].items()} if report else None
    ci_jobs = load_json(CI_JOBS_PATH)
    today = datetime.now(tz=UTC).strftime("%m/%d/%Y")

    if nodes is None:
        print("No verification report with node results found: every pytest step will show as pending")
    for sub in ("requirements", "test-cases"):
        (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)
        for old in (OUTPUT_DIR / sub).glob("*.md"):
            old.unlink()
    for req in registry["requirements"]:
        (OUTPUT_DIR / "requirements" / requirement_filename(req)).write_text(requirement_md(req))
    for tc in registry["test_cases"]:
        out = OUTPUT_DIR / "test-cases" / f"test-case-{tc['id']}.md"
        out.write_text(test_case_md(tc, aliases, nodes, ci_jobs, today))
    (OUTPUT_DIR / "vcrm.md").write_text(vcrm_md(registry, nodes, ci_jobs, today))
    if report:
        (OUTPUT_DIR / "test-results.log").write_text(test_results_log(registry, report, ci_jobs))

    results = [tc_outcome(t, aliases, nodes, ci_jobs) for t in registry["test_cases"]]
    counts = {k: results.count(k) for k in (PASS, FAIL, SKIP, PENDING)}
    print(f"{len(registry['requirements'])} requirements, {len(results)} test cases: {counts}")
    print(f"Artifacts written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
