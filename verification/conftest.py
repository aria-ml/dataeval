"""Verification test configuration and report generation plugin.

Provides:
- JSON report of every test's outcome, keyed by node id (read by generate_metarepo.py)
- Terminal summary of verification results
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

VERIFICATION_DIR = Path(__file__).parent
OUTPUT_DIR = VERIFICATION_DIR.parent / "output"

# Ensure the project root is on sys.path so that ``from verification.helpers``
# imports work regardless of how pytest is invoked.
_PROJECT_ROOT = str(VERIFICATION_DIR.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


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


def pytest_sessionstart(session):
    session.config._verification_started = datetime.now(UTC)


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


def pytest_sessionfinish(session, exitstatus):
    """Write ``output/verification_report.json``: run metadata and the outcome of every collected test.

    The registry (``verification/registry.yaml``) maps test cases and their steps to node ids, so
    ``generate_metarepo.py`` reads results from this map rather than from markers on the tests.
    """
    tests: dict[str, dict] = {}
    for item in session.items:
        message = _get_test_message(item)
        tests[item.nodeid] = {"status": _get_test_status(item), **({"message": message} if message else {})}
    if not tests:
        return

    statuses = [t["status"] for t in tests.values()]
    report = {
        "summary": {
            "total_tests": len(tests),
            **{s: statuses.count(s) for s in ("passed", "failed", "error", "skipped")},
        },
        "run": {
            "started": _utc(session.config._verification_started),
            "finished": _utc(datetime.now(UTC)),
            "exit_status": int(exitstatus),
            "command": " ".join(["pytest", *session.config.invocation_params.args]),
            "python": platform.python_version(),
            "platform": f"{platform.system().lower()} {platform.machine()}",
        },
        "tests": dict(sorted(tests.items())),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "verification_report.json").write_text(json.dumps(report, indent=2))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print a compact verification summary after the normal pytest output."""
    report_path = OUTPUT_DIR / "verification_report.json"
    if not report_path.exists():
        return

    report = json.loads(report_path.read_text())
    summary = report["summary"]

    terminalreporter.section("Verification Report")
    terminalreporter.write_line(
        f"Tests: {summary['total_tests']} total, {summary['passed']} passed, "
        f"{summary['failed']} failed, {summary['error']} errored, {summary['skipped']} skipped",
    )
    terminalreporter.write_line(f"Report: {report_path}")
    for node, result in report["tests"].items():
        if result["status"] in ("failed", "error"):
            terminalreporter.write_line(f"  {result['status'].upper()}: {node}")
