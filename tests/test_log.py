import logging
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import dataeval
from dataeval._log import _NAMESPACES, get_logger
from dataeval.types import Output, set_metadata

# Matched against source text, so both patterns anchor on the assignment. A bare
# substring search would also hit the prose in ``_log.py``'s own docstrings.
DERIVED_LOGGER = re.compile(r"=\s*logging\.getLogger\(__name__\)")
CURATED_LOGGER = re.compile(r"=\s*get_logger\(__name__\)")


@pytest.mark.required
@patch.object(logging.StreamHandler, "emit")
def test_dateval_log_default(mock_emit):
    dataeval.log()
    assert mock_emit.called


@pytest.mark.required
def test_dataeval_log_custom():
    mock_handler = logging.StreamHandler()
    mock_handler.emit = MagicMock()
    dataeval.log(logging.DEBUG, mock_handler)
    assert mock_handler.emit.called


@pytest.mark.required
def test_dataeval_log_idempotent():
    """Calling log twice with the same handler does not attach it twice (68->70)."""
    logger = logging.getLogger("dataeval")
    handler = logging.StreamHandler()
    try:
        dataeval.log(logging.DEBUG, handler)
        dataeval.log(logging.DEBUG, handler)
        assert logger.handlers.count(handler) == 1
    finally:
        logger.removeHandler(handler)


class TestGetLogger:
    """Resolution of a module path to its curated logging namespace.

    Logger names are a public configuration surface: users filter and route on
    them. Deriving them from ``__name__`` ties that surface to the private module
    layout, so an internal file move renames a logger users may have configured.
    ``get_logger`` maps module paths onto a curated namespace instead.
    """

    @pytest.mark.required
    def test_subpackage_module_resolves_to_subpackage_namespace(self):
        assert get_logger("dataeval.core._ber").name == "dataeval.core"

    @pytest.mark.required
    def test_private_top_level_module_resolves_without_underscore(self):
        assert get_logger("dataeval._embeddings").name == "dataeval.embeddings"

    @pytest.mark.required
    def test_module_moved_into_a_package_keeps_its_namespace(self):
        """``_metadata.py`` becoming ``_metadata/`` must not rename the logger."""
        assert get_logger("dataeval._metadata").name == "dataeval.metadata"
        assert get_logger("dataeval._metadata._metadata").name == "dataeval.metadata"

    @pytest.mark.required
    def test_deeply_nested_module_resolves_to_the_subtree_root(self):
        assert get_logger("dataeval._metadata._structurers._tracking").name == "dataeval.metadata"

    @pytest.mark.required
    def test_src_prefix_is_stripped(self):
        """``fn.__module__`` is ``src.dataeval...`` when running against the source tree."""
        assert get_logger("src.dataeval.core._ber").name == "dataeval.core"

    @pytest.mark.required
    def test_unregistered_module_falls_back_to_the_root_logger(self):
        """Never raise at import time; the coverage test is what catches gaps."""
        assert get_logger("dataeval.unregistered._thing").name == "dataeval"

    @pytest.mark.required
    def test_curated_namespace_propagates_to_the_root_handler(self):
        """Configuring ``dataeval`` alone must still capture every subsystem."""
        handler = logging.Handler()
        handler.emit = MagicMock()
        root = logging.getLogger("dataeval")
        previous = root.level
        try:
            root.addHandler(handler)
            root.setLevel(logging.DEBUG)
            get_logger("dataeval.core._ber").debug("emitted")
            assert handler.emit.called
        finally:
            root.removeHandler(handler)
            root.setLevel(previous)


class TestNamespaceCoverage:
    """Structural enforcement that the curated namespace stays curated.

    A table of prefixes only stays authoritative if a new module cannot quietly
    bypass it. These walk the source tree rather than the import graph so a module
    with an unmet optional dependency is still covered.
    """

    @staticmethod
    def _source_modules():
        """Yield ``(module_path, source)`` for every module under ``src/dataeval``."""
        package = Path(dataeval.__file__).parent
        for path in sorted(package.rglob("*.py")):
            parts = path.relative_to(package).with_suffix("").parts
            parts = parts[:-1] if parts[-1] == "__init__" else parts
            yield ".".join(("dataeval", *parts)), path.read_text(encoding="utf-8")

    @pytest.mark.required
    def test_no_module_derives_its_logger_name_from_dunder_name(self):
        """``logging.getLogger(__name__)`` reintroduces the private module path."""
        offenders = [module for module, source in self._source_modules() if DERIVED_LOGGER.search(source)]
        assert offenders == [], (
            f"{offenders} derive a logger name from __name__; use dataeval._log.get_logger(__name__)"
        )

    @pytest.mark.required
    def test_every_logging_module_resolves_to_a_curated_namespace(self):
        """A module that logs must map to a subsystem, not fall back to the root."""
        uncurated = [
            module
            for module, source in self._source_modules()
            if CURATED_LOGGER.search(source) and get_logger(module).name == "dataeval"
        ]
        assert uncurated == [], f"{uncurated} fall back to the root logger; add a prefix to _log._NAMESPACES"

    @pytest.mark.required
    def test_no_namespace_prefix_is_stale(self):
        """A prefix left behind by a refactor curates nothing and misleads the next reader."""
        modules = {module for module, _ in self._source_modules()}
        stale = sorted(
            prefix for prefix in _NAMESPACES if not any(m == prefix or m.startswith(f"{prefix}.") for m in modules)
        )
        assert stale == [], f"{stale} match no module under src/dataeval; drop them from _log._NAMESPACES"


class _TracedResult(Output): ...


def evaluate(self) -> _TracedResult:
    return _TracedResult()


# Declared before decorating: ``set_metadata`` closes over the function it is given, so
# setting ``__module__`` on the wrapper afterwards would not reach it. Named ``evaluate``
# because the trace reports the *defining* function, and this stands in for a real one.
evaluate.__module__ = "dataeval.bias._balance"


class _Evaluator:
    evaluate = set_metadata(evaluate)


class _Subclass(_Evaluator):
    """An evaluator a caller inherited into their own package."""

    __module__ = "acme.evaluators"


class TestSetMetadataNamespace:
    """``set_metadata`` resolves its logger at call time from the decorated function.

    The execution trace belongs to the subsystem that ran, not to ``dataeval.types``
    where the decorator lives, so the module path is resolved per call. Curating it
    keeps that behaviour while dropping the private path from the logger name.
    """

    @staticmethod
    def _decorated(module: str):
        class _Result(Output): ...

        def evaluate() -> _Result:
            return _Result()

        evaluate.__module__ = module
        return set_metadata(evaluate)

    @pytest.mark.required
    def test_execution_trace_uses_the_curated_namespace(self, caplog):
        caplog.set_level(logging.INFO, logger="dataeval")
        self._decorated("dataeval.bias._balance")()
        assert {record.name for record in caplog.records} == {"dataeval.bias"}

    @pytest.mark.required
    def test_execution_trace_names_the_defining_module_in_the_message(self, caplog):
        """Curating the logger must not cost provenance: the message keeps the path."""
        caplog.set_level(logging.INFO, logger="dataeval")
        self._decorated("dataeval.bias._balance")()
        assert "dataeval.bias._balance.evaluate" in caplog.records[0].message

    @pytest.mark.required
    def test_a_subclass_traces_to_the_namespace_of_the_code_it_inherited(self, caplog):
        """A caller's subclass lives in a module DataEval has no namespace for.

        The namespace names the subsystem whose code ran, so inheriting an evaluator
        must not move its trace off ``dataeval.bias`` and onto the root -- a handler
        scoped to the subsystem would stop seeing it.
        """
        caplog.set_level(logging.INFO, logger="dataeval")
        _Subclass().evaluate()

        assert {record.name for record in caplog.records} == {"dataeval.bias"}
        # And the message still names what actually ran, which is the subclass.
        assert "acme.evaluators._Subclass.evaluate" in caplog.records[0].message
