"""Verification test configuration and report generation plugin.

Provides:
- ``test_case(*ids)`` marker linking tests to ``test-case-<id>.md`` in the meta repo
- JSON report generation mapping test case numbers to pass/fail results, plus
  run metadata and every test's status for the metarepo test-results log
- Terminal summary of verification results

A single test may map to multiple test cases (and therefore multiple FRs/NFRs
via the VCRM in the meta repo) by stacking markers::

    @pytest.mark.test_case(1)
    @pytest.mark.test_case(5)
    def test_something(): ...
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

VERIFICATION_DIR = Path(__file__).parent
OUTPUT_DIR = VERIFICATION_DIR.parent / "output"

# Ensure the project root is on sys.path so that ``from verification.helpers``
# imports work regardless of how pytest is invoked.
_PROJECT_ROOT = str(VERIFICATION_DIR.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def pytest_sessionstart(session):
    session.config._verification_started = datetime.now(timezone.utc)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "test_case(*ids): link test to one or more test-case-<id>.md files in the meta repo",
    )


# ---------------------------------------------------------------------------
# Collect per-phase reports so we can determine the final status of each item
# ---------------------------------------------------------------------------


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    reports = getattr(item, "_verification_reports", {})
    reports[call.when] = report
    item._verification_reports = reports


def _get_test_status(item):
    """Derive an overall status from the setup/call/teardown phase reports."""
    reports = getattr(item, "_verification_reports", {})

    for phase in ("setup", "teardown"):
        r = reports.get(phase)
        if r is not None and r.failed:
            return "error"

    call = reports.get("call")
    if call is not None:
        if call.passed:
            return "passed"
        if call.skipped:
            return "skipped"
        return "failed"

    setup = reports.get("setup")
    if setup is not None and setup.skipped:
        return "skipped"

    return "error"


def _get_test_message(item) -> str | None:
    """Return the first line of an item's skip, xfail, or failure reason, if any."""
    reports = getattr(item, "_verification_reports", {})
    for phase in ("setup", "call", "teardown"):
        r = reports.get(phase)
        if r is None:
            continue
        if r.skipped:
            reason = getattr(r, "wasxfail", None)
            if reason is None and isinstance(r.longrepr, tuple):
                reason = r.longrepr[2].removeprefix("Skipped: ")
            return reason.splitlines()[0][:200] if reason else None
        if r.failed:
            crash = getattr(r.longrepr, "reprcrash", None)
            message = crash.message if crash is not None else str(r.longrepr)
            return message.splitlines()[0][:200] if message else None
    return None


def _utc(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _tc_status(tests: list[dict]) -> str:
    """Determine overall test case status from individual test results.

    - "failed"  if any test failed or errored
    - "skipped" if all tests were skipped
    - "passed"  if no failures (passed tests + optional skips)
    """
    statuses = {t["status"] for t in tests}
    if statuses & {"failed", "error"}:
        return "failed"
    if statuses == {"skipped"}:
        return "skipped"
    return "passed"


def pytest_sessionfinish(session, exitstatus):
    """Write ``output/reports/verification_report.json``."""
    results: dict[str, list[dict]] = {}
    all_tests: dict[str, dict] = {}

    for item in session.items:
        status = _get_test_status(item)
        message = _get_test_message(item)
        all_tests[item.nodeid] = {"status": status, **({"message": message} if message else {})}
        for marker in item.iter_markers("test_case"):
            for tc_num in marker.args:
                tc_id = f"test-case-{tc_num}"
                results.setdefault(tc_id, []).append(
                    {
                        "test": item.nodeid,
                        "file": str(Path(item.fspath).relative_to(VERIFICATION_DIR)),
                        "status": status,
                    },
                )

    if not results:
        return

    tc_statuses = {tc_id: _tc_status(tests) for tc_id, tests in results.items()}
    passed = sum(1 for s in tc_statuses.values() if s == "passed")
    failed = sum(1 for s in tc_statuses.values() if s == "failed")
    skipped = sum(1 for s in tc_statuses.values() if s == "skipped")

    report = {
        "summary": {
            "total_test_cases": len(results),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
        },
        "test_cases": {
            tc_id: {
                "meta_repo_file": f"test-cases/{tc_id}.md",
                "status": _tc_status(tests),
                "tests": tests,
            }
            for tc_id, tests in sorted(results.items())
        },
        "run": {
            "started": _utc(session.config._verification_started),
            "finished": _utc(datetime.now(timezone.utc)),
            "exit_status": int(exitstatus),
            "command": " ".join(["pytest", *session.config.invocation_params.args]),
            "python": platform.python_version(),
            "platform": f"{platform.system().lower()} {platform.machine()}",
        },
        "tests": all_tests,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "verification_report.json"
    report_path.write_text(json.dumps(report, indent=2))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print a compact verification summary after the normal pytest output."""
    report_path = OUTPUT_DIR / "verification_report.json"
    if not report_path.exists():
        return

    report = json.loads(report_path.read_text())
    summary = report["summary"]

    terminalreporter.section("Verification Report")
    terminalreporter.write_line(
        f"Test Cases: {summary['total_test_cases']} total, "
        f"{summary['passed']} passed, "
        f"{summary['failed']} failed, "
        f"{summary['skipped']} skipped",
    )
    terminalreporter.write_line(f"Report: {report_path}")

    for tc_id, tc_data in report["test_cases"].items():
        if tc_data["status"] == "failed":
            terminalreporter.write_line(f"  FAILED: {tc_id} ({tc_data['meta_repo_file']})")
            for test in tc_data["tests"]:
                if test["status"] in ("failed", "error"):
                    terminalreporter.write_line(f"    - {test['test']}")
