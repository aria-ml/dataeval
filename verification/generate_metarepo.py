#!/usr/bin/env python3
"""Generate meta repo artifacts from verification test results.

Reads the static registry (registry.yaml) and the dynamic verification report
(verification_report.json) to produce:

  - test-cases/test-case-{id}.md  — one per test case, with Test Results filled
  - vcrm.md                       — VCRM with Verification row filled from results
  - test-results.log              — per-test-case run log for the dated assessment folder

Output directory: verification/reports/metarepo/
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
OUTPUT_DIR = PROJECT_ROOT / "output" / "metarepo"
PRODUCT = "DataEval"
DISTRIBUTION = "dataeval"


def load_registry() -> dict:
    with open(REGISTRY_PATH) as f:
        return yaml.safe_load(f)


def load_report() -> dict | None:
    if REPORT_PATH.exists():
        with open(REPORT_PATH) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Test Case Markdown Generation
# ---------------------------------------------------------------------------


def _human_readable_step(nodeid: str) -> str:
    """Convert a pytest nodeid's test name into a human-readable step description."""
    # Extract the test function/method name (last :: segment)
    test_name = nodeid.rsplit("::", 1)[-1]
    # Strip 'test_' prefix, replace underscores with spaces, capitalize
    desc = re.sub(r"^test_", "", test_name).replace("_", " ").capitalize()
    return desc


def _result_char(status: str) -> str:
    """Map test status to P/F/S for the Test Results table."""
    if status == "passed":
        return "P"
    if status == "skipped":
        return "S"
    return "F"


def generate_test_case_md(tc_id: str, tc_meta: dict, report: dict | None) -> str:
    """Generate a single test case markdown file."""
    tc_key = f"test-case-{tc_id}"
    today = datetime.now(tz=UTC).strftime("%m/%d/%Y")

    # Look up dynamic results
    tc_report = None
    if report and tc_key in report.get("test_cases", {}):
        tc_report = report["test_cases"][tc_key]

    lines: list[str] = []

    # --- Header ---
    lines.append(f"# {tc_meta['name']}")
    lines.append("")

    # --- Description ---
    lines.append("## Description")
    lines.append("")
    lines.append(f"- Test Type: {tc_meta['test_type']}")
    lines.append(f"- Business Case: {tc_meta['business_case'].strip()}")
    lines.append("")
    lines.append("**Initial Conditions:**")
    lines.append("")
    for i, cond in enumerate(tc_meta["initial_conditions"], 1):
        lines.append(f"{i}. {cond}")
    lines.append("")

    # --- Test Steps ---
    lines.append("## Test Steps")
    lines.append("")

    if tc_report:
        tests = tc_report["tests"]
        for i, test in enumerate(tests, 1):
            lines.append(f"{i}. {_human_readable_step(test['test'])}")
        lines.append(f"{len(tests) + 1}. Confirm the Expected Results by validating all steps pass")
    else:
        # No automated results — use manual steps if defined, else placeholder
        manual_steps = tc_meta.get("test_steps", [])
        if manual_steps:
            for i, step in enumerate(manual_steps, 1):
                lines.append(f"{i}. {step}")
        else:
            for i, er in enumerate(tc_meta["expected_results"], 1):
                lines.append(f"{i}. Verify: {er}")
            lines.append(
                f"{len(tc_meta['expected_results']) + 1}. Confirm the Expected Results by validating all steps pass",
            )
    lines.append("")

    # --- Expected Results ---
    lines.append("**Expected Results**")
    lines.append("")
    for i, result in enumerate(tc_meta["expected_results"], 1):
        lines.append(f"{i}. {result}")
    lines.append("")

    # --- Test Results ---
    lines.append("## Test Results")
    lines.append("")
    lines.append("| Test Step |  Result | Notes |")
    lines.append("|:----------|:-------:|:------|")

    if tc_report:
        tests = tc_report["tests"]
        for i, test in enumerate(tests, 1):
            r = _result_char(test["status"])
            lines.append(f"|{i:<10}|    {r}    |  [^{i}] |")
        # Confirmation step
        overall = tc_report["status"]
        confirm = "P" if overall == "passed" else "F"
        n = len(tests) + 1
        lines.append(f"|{n:<10}|    {confirm}    |  [^{n}] |")
        lines.append("")

        # Footnotes
        for i, test in enumerate(tests, 1):
            lines.append(f"[^{i}]: `{test['test']}` — {test['status']}")
        lines.append(f"[^{n}]: Overall verification — {overall}")
    else:
        lines.append("|1         |   P/F   |  [^1] |")
        lines.append("")
        lines.append("[^1]: Awaiting automated test results")

    lines.append("")
    lines.append(f"**Last Updated Date:** {today}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# VCRM Markdown Generation
# ---------------------------------------------------------------------------


def _tc_sort_key(tc_id: str) -> list[int]:
    """Sort key for test case IDs like '1-1', '1-7', '2-3'."""
    return [int(p) for p in tc_id.split("-")]


def generate_vcrm(registry: dict, report: dict | None) -> str:
    """Generate the full VCRM markdown."""
    requirements = registry["requirements"]
    test_cases = registry["test_cases"]
    today = datetime.now(tz=UTC).strftime("%m/%d/%Y")

    # Sorted test case IDs
    all_tc_ids = sorted(test_cases.keys(), key=_tc_sort_key)

    # --- Header row ---
    tc_headers = []
    for tc_id in all_tc_ids:
        display = tc_id.replace("-", ".")
        tc_headers.append(f"[TC-{display}][{tc_id}]")
    header = "| Requirement ID | Requirement Origin | Coverage | " + " | ".join(tc_headers) + " |"

    # --- Separator row ---
    sep_parts = [":--------------", ":-------------------", ":--------:"]
    for _ in all_tc_ids:
        sep_parts.append(":-------------:")
    separator = "| " + " | ".join(sep_parts) + " |"

    # --- Requirement rows ---
    rows: list[str] = []
    for req_id, req_data in requirements.items():
        req_tcs = set(req_data.get("test_cases", []))
        coverage = "Yes" if req_tcs else "No"

        # Requirement ID link
        ref_key = req_id.lower().replace("-", "")
        req_cell = f"[{req_id}][{ref_key}]"

        # Origin link
        origin = req_data.get("origin", "")
        origin_ref = origin.lower().replace("-", "").replace(".", "")
        origin_cell = f"[{origin}][{origin_ref}]" if origin else ""

        # X marks
        tc_cells = ["X" if tc_id in req_tcs else " " for tc_id in all_tc_ids]

        row_parts = [req_cell, origin_cell, coverage] + tc_cells
        rows.append("| " + " | ".join(row_parts) + " |")

    # --- Verification row ---
    verification_cells: list[str] = []
    for tc_id in all_tc_ids:
        tc_key = f"test-case-{tc_id}"
        if report and tc_key in report.get("test_cases", {}):
            status = report["test_cases"][tc_key]["status"]
            verification_cells.append("Pass" if status == "passed" else "Fail")
        else:
            verification_cells.append("Pending")
    verification_row = "| **Verification** | | | " + " | ".join(verification_cells) + " |"

    # --- Reference links ---
    # Test case links
    tc_links = [f"[{tc_id}]:test-cases/test-case-{tc_id}.md" for tc_id in all_tc_ids]

    # Requirement ID links
    req_links = []
    for req_id, req_data in requirements.items():
        ref_key = req_id.lower().replace("-", "")
        filename = req_data.get("file", "#")
        req_links.append(f"[{ref_key}]:requirements/{filename}")

    # Origin links (deduplicated)
    origin_links: dict[str, str] = {}
    for req_data in requirements.values():
        origin = req_data.get("origin", "")
        origin_link = req_data.get("origin_link", "#")
        if origin:
            origin_ref = origin.lower().replace("-", "").replace(".", "")
            origin_links[origin_ref] = f"[{origin_ref}]:{origin_link}"

    # --- Assemble ---
    parts = [
        "# DataEval Verification Cross-Reference Matrix (VCRM)",
        "",
        header,
        separator,
        *rows,
        verification_row,
        "",
        f"**Last Updated:** {today}",
        "",
        "<!-- Links for Test Cases -->",
        "",
        *tc_links,
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
# Test Results Log Generation
# ---------------------------------------------------------------------------

LOG_WIDTH = 100
_LOG_STATUS = {"passed": "PASS", "failed": "FAIL", "error": "ERROR", "skipped": "SKIP", "xfailed": "XFAIL"}


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *args],
            capture_output=True,
            text=True,
            check=True,
        )
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


def _product_version() -> str:
    try:
        return metadata.version(DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return "unknown"


def _log_label(status: str) -> str:
    return _LOG_STATUS.get(status, status.upper())


def _log_counts(statuses: list[str]) -> str:
    found = {s: statuses.count(s) for s in _LOG_STATUS if statuses.count(s)}
    return ", ".join(f"{n} {s}" for s, n in found.items()) or "none"


def _log_requirement_status(tcs: list[str], tc_status: dict[str, str]) -> str:
    statuses = [tc_status.get(t) for t in tcs]
    if not tcs:
        return "UNMAPPED"
    if all(s == "passed" for s in statuses):
        return "VERIFIED"
    if any(s == "failed" for s in statuses):
        return "FAILED"
    return "PARTIAL"


def _log_test_case_rows(registry: dict, cases: dict, tc_status: dict[str, str]) -> list[str]:
    tc_to_reqs: dict[str, list[str]] = {}
    for req_id, req_data in registry["requirements"].items():
        for tc_id in req_data.get("test_cases", []):
            tc_to_reqs.setdefault(tc_id, []).append(req_id)

    rows = [f"{'TEST CASE':<10}{'STATUS':<8}{'PASSED/TESTS':<18}{'REQUIREMENTS':<16}NAME"]
    for tc_id in sorted(set(registry["test_cases"]) | set(tc_status), key=_tc_sort_key):
        tests = cases.get(f"test-case-{tc_id}", {}).get("tests", [])
        n_pass = sum(t["status"] == "passed" for t in tests)
        n_other = len(tests) - n_pass - sum(t["status"] in ("failed", "error") for t in tests)
        tally = f"{n_pass}/{len(tests)}" + (f" ({n_other} not run)" if n_other else "")
        status = _log_label(tc_status[tc_id]) if tc_id in tc_status else "NOT RUN"
        name = registry["test_cases"].get(tc_id, {}).get("name", "(not in registry)")
        rows.append(f"{tc_id:<10}{status:<8}{tally:<18}{', '.join(tc_to_reqs.get(tc_id, ['-'])):<16}{name}")
    return rows


def _log_detail_rows(cases: dict, all_tests: dict) -> list[str]:
    def test_line(nodeid: str, status: str) -> list[str]:
        message = all_tests.get(nodeid, {}).get("message")
        return [f"  {_log_label(status):<6}{nodeid}"] + ([f"        -> {message}"] if message else [])

    rows: list[str] = []
    for tc_key, case in sorted(cases.items(), key=lambda kv: _tc_sort_key(kv[0].removeprefix("test-case-"))):
        rows.append(f"[{tc_key}] {_log_label(case['status'])}")
        for test in case["tests"]:
            rows += test_line(test["test"], test["status"])
        rows.append("")

    mapped = {t["test"] for case in cases.values() for t in case["tests"]}
    unmapped = sorted(nodeid for nodeid in all_tests if nodeid not in mapped)
    rows += ["-" * LOG_WIDTH, "TESTS NOT MAPPED TO A TEST CASE", "-" * LOG_WIDTH]
    for nodeid in unmapped:
        rows += test_line(nodeid, all_tests[nodeid]["status"])
    if not unmapped:
        rows.append("  (none)")
    return rows


def generate_test_results_log(registry: dict, report: dict) -> str:
    """Generate the metarepo ``test-results.log`` (DR-1.1-H-4, DR-1.3-H-1) from a verification run."""
    run = report.get("run", {})
    all_tests = report.get("tests", {})
    cases = report["test_cases"]
    requirements = registry["requirements"]
    tc_status = {tc_key.removeprefix("test-case-"): c["status"] for tc_key, c in cases.items()}
    req_status = {r: _log_requirement_status(d.get("test_cases", []), tc_status) for r, d in requirements.items()}
    verified = list(req_status.values()).count("VERIFIED")
    rule, thin = "=" * LOG_WIDTH, "-" * LOG_WIDTH

    lines = [
        rule,
        f"{PRODUCT} - Verification Test Results",
        rule,
        f"Product version    : {_product_version()}",
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
        f"Test cases         : {len(cases)} total, {_log_counts(list(tc_status.values()))}",
        f"Tests              : {len(all_tests)} total, {_log_counts([t['status'] for t in all_tests.values()])}",
        f"Requirements       : {len(requirements)} total, {verified} verified (every mapped test case passed)",
        "",
        thin,
        "TEST CASE RESULTS",
        thin,
        *_log_test_case_rows(registry, cases, tc_status),
        "",
        thin,
        "REQUIREMENT COVERAGE",
        thin,
        f"{'REQUIREMENT':<13}{'STATUS':<10}{'TEST CASES':<20}NAME",
        *(
            f"{r:<13}{req_status[r]:<10}{', '.join(d.get('test_cases', [])) or '-':<20}{d['name']}"
            for r, d in requirements.items()
        ),
        "",
        thin,
        "TEST DETAIL",
        thin,
        *_log_detail_rows(cases, all_tests),
        "",
        rule,
        "END OF REPORT",
        rule,
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    registry = load_registry()
    report = load_report()

    if report:
        s = report["summary"]
        print(
            f"Loaded verification report: {s['total_test_cases']} test cases "
            f"({s['passed']} passed, {s['failed']} failed, {s['skipped']} skipped)",
        )
    else:
        print("No verification report found — generating templates only")

    tc_dir = OUTPUT_DIR / "test-cases"
    tc_dir.mkdir(parents=True, exist_ok=True)

    # Generate test case files
    for tc_id, tc_meta in registry["test_cases"].items():
        content = generate_test_case_md(tc_id, tc_meta, report)
        out_path = tc_dir / f"test-case-{tc_id}.md"
        out_path.write_text(content)
        print(f"  Generated {out_path.name}")

    # Generate VCRM
    vcrm_content = generate_vcrm(registry, report)
    vcrm_path = OUTPUT_DIR / "vcrm.md"
    vcrm_path.write_text(vcrm_content)
    print("  Generated vcrm.md")

    # Generate test results log
    if report:
        (OUTPUT_DIR / "test-results.log").write_text(generate_test_results_log(registry, report))
        print("  Generated test-results.log")

    print(f"\nAll artifacts written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
