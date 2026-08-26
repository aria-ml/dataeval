"""Enforce that python-domain cross-references in prose point at symbols that exist.

A cross-reference to a renamed or deleted symbol does not fail the docs build. Sphinx
resolves what it can and silently renders the rest as plain text, so the page still
builds and the link is simply gone. That is how references to the pre-class API
(``imagestats``, ``labelstats``, ``balance``, ``diversity``, ``coverage``, ``ber``,
``divergence``, ``calculate``) survived the rename unnoticed.

Sphinx's own ``nitpicky`` mode reports these, but it cannot be used as a gate here:
``autodoc_typehints = "description"`` and numpydoc type lines turn every annotation
into a cross-reference, and ~1280 of those have nothing to bind to (a bare ``NDArray``,
the word ``optional``, each value of a ``Literal[...]`` split on its commas). None of
that is fixable from the docstring side, and it is indistinguishable by name from the
prose references that matter.

So this checks the prose directly, against the imported package rather than a docs
build: it runs in seconds and needs no Sphinx. It covers ``src/`` docstrings as well as
the hand-written pages, which is where the two references this test was written against
turned out to be broken.

A second failure mode is checked alongside it: a reference whose *markup* is malformed, so
the role never fires at all and the whole thing renders as literal text. This is the same
silent loss from the other direction -- the symbol is right, the link still does not exist --
and it is invisible to the resolution check below, which reads the role out of the source
without caring whether Sphinx would have parsed it.

Only the two reference forms whose resolution does not depend on context are checked:

* absolute -- ``:class:`~dataeval.data.Embeddings```
* project-relative -- ``{func}`.compute_stats```, the leading dot being Sphinx's
  "search the whole project" marker, and the dominant form in these docs.

A bare ``:class:`Ontology``` is skipped. Sphinx resolves those against the surrounding
module first and only then globally, and reproducing that faithfully would mean
reimplementing the python domain; a bare name is also indistinguishable from a
reference to a third-party type such as ``ArrayLike``. Spell a reference with a leading
dot or a full path and it is covered.
"""

import importlib
import inspect
import pkgutil
import re
from pathlib import Path

import pytest

import dataeval

ROOT = Path(__file__).resolve().parent.parent

# Both role spellings that appear in these sources: reST ``:class:`X``` in ``src/``
# docstrings and ``.rst`` pages, MyST ``{class}`X``` in the ``.md`` guides and notebooks.
ROLE = re.compile(r"[:{](?:py:)?(?P<role>func|class|meth|obj|attr|data|mod|exc)[}:]`(?P<body>[^`]+)`")
# The same marker without its body, for the malformed-markup checks below.
MARKER = r"[:{](?:py:)?(?:func|class|meth|obj|attr|data|mod|exc)[}:]"
# ``` `{meth}`.Metadata.agg` ``` -- a backtick before the role makes it a code span holding
# the word "meth", and the target beside it plain text. Renders as ``{meth}.Metadata.agg` ``.
WRAPPED_ROLE = re.compile(rf"`{MARKER}`[^`\n]+`")
# ``` {func} `.label_errors` ``` -- a role and its target are one token; a space between them
# leaves the role as literal text and the target as an ordinary code span.
SPACED_ROLE = re.compile(rf"{MARKER}[ \t]+`")
# Both are deliberately narrow. A bare ``{exc}`` with no backtick after it is *not* flagged:
# it is far more often an f-string placeholder in ``src/`` than a broken role.
# ``{meth}`display text <.Ontology.is_a>``` -- the target is what is in the brackets.
ANGLE = re.compile(r"<(?P<target>[^<>]+)>\s*$")

ATTRIBUTES_SECTION = re.compile(r"^Attributes\s*\n-+\s*$", re.MULTILINE)
NEXT_SECTION = re.compile(r"\n[A-Z][A-Za-z ]+\n-+\n")
ATTRIBUTE_ENTRY = re.compile(r"^(?P<name>[A-Za-z_]\w*)\s*(?::.*)?$", re.MULTILINE)


def _iter_modules():
    """Yield ``dataeval`` and every importable public submodule."""
    yield dataeval
    for info in pkgutil.walk_packages(dataeval.__path__, "dataeval."):
        if any(part.startswith("_") for part in info.name.split(".")):
            continue
        try:
            yield importlib.import_module(info.name)
        except Exception:  # pragma: no cover - skip modules with unmet optional deps
            continue


def _documented_attributes(obj) -> set[str]:
    """Names from a numpydoc ``Attributes`` section.

    napoleon renders each entry as an ``.. attribute::`` directive, so these are real
    anchors even when the attribute is assigned at runtime and never appears in
    ``dir()`` -- which is true of every ``DataFrameOutput`` subclass.
    """
    doc = inspect.getdoc(obj) or ""
    heading = ATTRIBUTES_SECTION.search(doc)
    if not heading:
        return set()
    body = doc[heading.end() :]
    following = NEXT_SECTION.search(body)
    if following:
        body = body[: following.start()]
    # Entries start in column 0; their descriptions are indented beneath them.
    return {m.group("name") for m in ATTRIBUTE_ENTRY.finditer(body)}


def _class_members(obj) -> set[str]:
    names: set[str] = set()
    for klass in inspect.getmro(obj):
        names.update(vars(klass))
        names.update(getattr(klass, "__annotations__", {}))  # dataclass fields
        names.update(_documented_attributes(klass))
    # Keep public names and dunders (``__getitem__`` is documented and referenced);
    # drop single-underscore internals.
    return {n for n in names if not n.startswith("_") or (n.startswith("__") and n.endswith("__"))}


def _build_symbols() -> set[str]:
    """Every dotted name a cross-reference could legitimately resolve to."""
    symbols: set[str] = set()
    for module in _iter_modules():
        symbols.add(module.__name__)
        for name, obj in vars(module).items():
            if name.startswith("_"):
                continue
            # Without this, ``import torch`` inside a module would register "torch"
            # as a dataeval symbol and make ``torch.device`` look locally defined.
            if inspect.ismodule(obj) and not getattr(obj, "__name__", "").startswith("dataeval"):
                continue
            qualified = f"{module.__name__}.{name}"
            symbols.add(qualified)
            if inspect.isclass(obj):
                symbols.update(f"{qualified}.{member}" for member in _class_members(obj))
    return symbols


SYMBOLS = _build_symbols()
# Every trailing fragment of every symbol, which is what a leading-dot reference
# matches: ``.Balance`` resolves because "Balance" ends ``dataeval.bias.Balance``.
SUFFIXES = {".".join(parts[i:]) for s in SYMBOLS for parts in [s.split(".")] for i in range(len(parts))}


def _iter_source_files():
    for pattern in ("docs/source/**/*.md", "docs/source/**/*.rst", "docs/source/**/*.py"):
        for path in ROOT.glob(pattern):
            if ".jupyter_cache" not in path.parts:
                yield path
    yield from (ROOT / "src").rglob("*.py")


def _collect_references():
    """Return ``[(relative_path, target), ...]`` for every checkable reference."""
    found = []
    for path in sorted(_iter_source_files()):
        for match in ROLE.finditer(path.read_text(errors="replace")):
            body = match.group("body").strip()
            angle = ANGLE.search(body)
            target = (angle.group("target") if angle else body).strip().lstrip("~")
            if target.startswith("dataeval.") or target.startswith("."):
                found.append((path.relative_to(ROOT).as_posix(), target))
    return found


REFERENCES = _collect_references()


@pytest.mark.required
def test_collector_finds_references():
    """Guard against the extractor silently matching nothing (e.g. a regex regression)."""
    assert len(REFERENCES) > 500, f"only found {len(REFERENCES)} references; the extractor is likely broken"


@pytest.mark.required
def test_symbol_table_is_populated():
    """Guard against an import regression emptying the table and passing everything."""
    assert "dataeval.bias.Balance" in SYMBOLS
    assert "Balance" in SUFFIXES


@pytest.mark.required
@pytest.mark.parametrize(("source", "target"), REFERENCES, ids=[f"{s}::{t}" for s, t in REFERENCES])
def test_cross_reference_resolves(source: str, target: str):
    """Every absolute or project-relative cross-reference names a symbol that exists."""
    resolved = target in SYMBOLS if target.startswith("dataeval.") else target.lstrip(".") in SUFFIXES
    assert resolved, (
        f"{source} references '{target}', which does not exist in the dataeval package. "
        "Sphinx renders an unresolved reference as plain text, so this link is silently "
        "dead. Update it to the symbol's current name, or drop the role if the target is "
        "no longer public."
    )


def _collect_malformed():
    """Return ``[(relative_path, line, text, why), ...]`` for every broken role marker."""
    found = []
    checks = (
        (WRAPPED_ROLE, "the role is wrapped in backticks, which makes it a code span"),
        (SPACED_ROLE, "there is whitespace between the role and its target"),
    )
    for path in sorted(_iter_source_files()):
        text = path.read_text(errors="replace")
        for pattern, why in checks:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                found.append((path.relative_to(ROOT).as_posix(), line, match.group(0), why))
    return found


MALFORMED = _collect_malformed()


@pytest.mark.required
def test_malformed_detector_is_wired_up():
    """Guard against the patterns silently matching nothing, as the collector above does."""
    sample = "See {meth}`.Metadata.agg` and `{meth}`.Metadata.at` and {func} `.compute_stats`."
    assert len(WRAPPED_ROLE.findall(sample)) == 1, "the wrapped-role pattern stopped matching"
    assert len(SPACED_ROLE.findall(sample)) == 1, "the spaced-role pattern stopped matching"
    assert not WRAPPED_ROLE.search("{meth}`.Metadata.agg`"), "a well-formed role must not match"
    assert not SPACED_ROLE.search("{meth}`.Metadata.agg`"), "a well-formed role must not match"
    assert not SPACED_ROLE.search('f"{exc} raised"'), "an f-string placeholder must not match"


@pytest.mark.required
def test_no_malformed_role_markup():
    """Every role marker is spelled so that Sphinx actually parses it as one.

    Not parametrized over the offenders, as ``test_cross_reference_resolves`` is over its
    references: the intended steady state here is *zero* rows, and a parametrized test with
    nothing to parametrize reports as skipped rather than passed. One assertion listing
    every offender always runs, and shows them all at once rather than one per run.
    """
    assert not MALFORMED, (
        "Malformed cross-reference markup:\n"
        + "\n".join(f"  {source}:{line} contains {text!r} -- {why}" for source, line, text, why in MALFORMED)
        + (
            "\nThe role never fires, so Sphinx renders it as literal text and the link is "
            "silently dead even though the symbol exists. Spell it {role}`.Target`: no backtick "
            "before the role, no whitespace between it and its target."
        )
    )
