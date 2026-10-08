"""Unit tests for the meta repo artifact tooling (verification/generate_metarepo.py and push_verification.py)."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pytest.importorskip("yaml")
gen = _load("generate_metarepo", ROOT / "verification" / "generate_metarepo.py")

ALIASES = {"a": "pkg/test_a.py::TestA"}
NODES = {
    "pkg/test_a.py::TestA::test_one": "passed",
    "pkg/test_a.py::TestA::test_two": "failed",
    "pkg/test_a.py::TestA::test_skip": "skipped",
}


def _tc(*tests):
    return {
        "id": "1-1",
        "req": "FR-1",
        "name": "n",
        "type": "Core Functionality",
        "business": "b",
        "conditions": ["c"],
        "steps": [{"do": "d", "expect": "e", "tests": list(tests)}],
    }


class TestOutcomes:
    def test_passing_nodes_pass(self):
        assert gen.tc_outcome(_tc("a::test_one"), ALIASES, NODES, None) == "passed"

    def test_a_failing_node_fails_the_step(self):
        assert gen.tc_outcome(_tc("a::test_one", "a::test_two"), ALIASES, NODES, None) == "failed"

    def test_planned_tests_are_pending_not_passed(self):
        assert gen.tc_outcome(_tc("a::test_one", "NEW: something"), ALIASES, NODES, None) == "pending"

    def test_a_node_missing_from_the_report_is_pending(self):
        assert gen.tc_outcome(_tc("a::test_missing"), ALIASES, NODES, None) == "pending"

    def test_all_skipped_is_skipped_not_failed(self):
        assert gen.tc_outcome(_tc("a::test_skip"), ALIASES, NODES, None) == "skipped"

    def test_ci_job_results_are_read(self):
        assert gen.tc_outcome(_tc("CI: lint"), ALIASES, None, {"lint": "success"}) == "passed"
        assert gen.tc_outcome(_tc("CI: lint"), ALIASES, None, {"lint": "failed"}) == "failed"
        assert gen.tc_outcome(_tc("CI: lint"), ALIASES, None, None) == "pending"


class TestRendering:
    def test_requirement_file_name_follows_dr_1_5(self):
        assert (
            gen.requirement_filename({"id": "NFR-4", "name": "Config & Reproducibility"})
            == "NFR-4-config-reproducibility.md"
        )

    def test_each_step_has_one_expected_result_and_a_confirm_step(self):
        md = gen.test_case_md(_tc("a::test_one"), ALIASES, NODES, None, "01/01/2026")
        assert "2. Confirm the Expected Results" in md
        assert md.count("\n1. ") == 3  # step, expected result, initial-conditions-free numbering check

    def test_registry_without_coverage_is_rejected(self):
        reg = {"requirements": [{"id": "FR-1"}, {"id": "FR-2"}], "test_cases": [_tc("a::test_one")]}
        with pytest.raises(SystemExit):
            gen.check_registry(reg)

    def test_registry_with_unknown_requirement_is_rejected(self):
        tc = _tc("a::test_one")
        tc["req"] = "FR-9"
        with pytest.raises(SystemExit):
            gen.check_registry({"requirements": [{"id": "FR-1"}], "test_cases": [tc]})


def test_shipped_registry_is_consistent():
    gen.check_registry(gen.load_registry())


class TestStalePruning:
    def test_only_managed_kinds_directly_in_their_directories_are_stale(self, monkeypatch):
        pytest.importorskip("requests")
        monkeypatch.syspath_prepend(str(ROOT / ".gitlab" / "scripts"))
        push = _load("push_verification", ROOT / ".gitlab" / "scripts" / "push_verification.py")
        existing = {
            "P/requirements/FR-1-old.md",
            "P/requirements/NFR-2-old.md",
            "P/test-cases/test-case-9-9.md",
            "P/archive/2026-10-07/requirements/FR-1-old.md",
            "P/proposed/requirements/FR-1-x.md",
            "P/requirements/notes.md",
            "P/vcrm.md",
            "P/requirements/FR-3-kept.md",
        }
        stale = push.find_stale(existing, {"P/requirements/FR-3-kept.md", "P/vcrm.md"}, "P")
        assert stale == ["P/requirements/FR-1-old.md", "P/requirements/NFR-2-old.md", "P/test-cases/test-case-9-9.md"]


class TestDefaultBranchGuard:
    @pytest.fixture
    def push(self, monkeypatch):
        pytest.importorskip("requests")
        monkeypatch.syspath_prepend(str(ROOT / ".gitlab" / "scripts"))
        for var in ("CI_COMMIT_BRANCH", "CI_DEFAULT_BRANCH", "CI_COMMIT_TAG"):
            monkeypatch.delenv(var, raising=False)
        return _load("push_verification", ROOT / ".gitlab" / "scripts" / "push_verification.py")

    def test_default_branch_is_refused(self, push, monkeypatch):
        monkeypatch.setenv("CI_COMMIT_BRANCH", "main")
        monkeypatch.setenv("CI_DEFAULT_BRANCH", "main")
        with pytest.raises(SystemExit, match="default branch"):
            push.refuse_default_branch()

    def test_main_is_refused_even_if_the_default_differs(self, push, monkeypatch):
        monkeypatch.setenv("CI_COMMIT_BRANCH", "main")
        monkeypatch.setenv("CI_DEFAULT_BRANCH", "develop")
        with pytest.raises(SystemExit):
            push.refuse_default_branch()

    def test_release_branch_and_tag_are_allowed(self, push, monkeypatch):
        monkeypatch.setenv("CI_DEFAULT_BRANCH", "main")
        monkeypatch.setenv("CI_COMMIT_BRANCH", "release/v1.1")
        push.refuse_default_branch()
        monkeypatch.delenv("CI_COMMIT_BRANCH")
        monkeypatch.setenv("CI_COMMIT_TAG", "v1.1.5")
        push.refuse_default_branch()


class TestResultsLog:
    def test_log_reports_a_failed_step_and_uncited_tests(self):
        registry = {
            "product": "P",
            "distribution": "no-such-distribution",
            "aliases": ALIASES,
            "requirements": [{"id": "FR-1", "name": "r"}],
            "test_cases": [_tc("a::test_one", "a::test_two")],
        }
        report = {"run": {"command": "pytest x"}, "tests": {k: {"status": v} for k, v in NODES.items()}}
        log = gen.test_results_log(registry, report, None)
        assert "FAIL" in log
        assert "pkg/test_a.py::TestA::test_two" in log
        assert "FAILED" in log  # the requirement is FAILED
        assert "pkg/test_a.py::TestA::test_skip" in log.split("TESTS NOT CITED")[1]
