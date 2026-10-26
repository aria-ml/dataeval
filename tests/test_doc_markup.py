"""Enforce that the Markdown in the guides and notebooks is spelled so that Sphinx parses it.

Two spellings build cleanly and render wrongly, so neither ``--fail-on-warning`` nor the
cross-reference test sees them:

* A reST role (``:class:`X```) in Markdown. MyST reads ``{class}`X```, so the reST form renders as
  literal text. ``test_doc_cross_references`` accepts both spellings because ``src/`` docstrings are
  reST, which means it cannot reject the wrong one in a guide.
* A code fence that closes the wrong block. A four-backtick directive holding a three-backtick example
  that is then closed with four backticks leaves the fences balanced, so a count of openers and closers
  passes, yet the example is swallowed and an empty block follows.

Only Markdown is checked: the ``.md`` guides and the markdown cells of the ``py:percent`` notebooks.
Code cells and ``src/`` docstrings are not.
"""

import re
from collections.abc import Iterator
from pathlib import Path

import pytest

pytestmark = pytest.mark.infrastructure

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs" / "source"

# The role names the cross-reference test checks, in their reST spelling.
RST_ROLE = re.compile(r"(?<![\w{`]):(?:py:)?(?:func|class|meth|obj|attr|data|mod|exc):`[^`\n]+`")
FENCE = re.compile(r"^\s*(`{3,})(.*)$")
INNER_NEVER_CLOSED = "a code block opened inside this fence is never closed before the fence ends"


def _markdown_blocks(path: Path) -> Iterator[tuple[int, list[str]]]:
    """Yield ``(first line number, lines)`` for each stretch of Markdown in a docs file."""
    lines = path.read_text(errors="replace").split("\n")
    if path.suffix == ".md":
        yield 1, lines
        return
    start: int | None = None
    block: list[str] = []
    for number, line in enumerate(lines, 1):
        if line.startswith("# %%"):
            if block and start is not None:
                yield start, block
            block, start = [], (number + 1 if "[markdown]" in line else None)
        elif start is not None:
            block.append(line[2:] if line.startswith("# ") else ("" if line == "#" else line))
    if block and start is not None:
        yield start, block


def _source_files() -> Iterator[Path]:
    for path in sorted(DOCS.rglob("*.md")):
        if ".jupyter_cache" not in path.parts:
            yield path
    yield from sorted((DOCS / "notebooks").glob("*.py"))


def _fence_problems(lines: list[str]) -> list[tuple[int, str]]:
    """Return ``(index, why)`` for each fence that does not do what it was written to do."""
    problems: list[tuple[int, str]] = []
    outer: tuple[int, int] | None = None  # (backticks, index) of the fence that is open
    outer_info = ""
    inner: tuple[int, int] | None = None  # a shorter fence opened inside it
    for index, line in enumerate(lines):
        match = FENCE.match(line)
        if not match:
            continue
        ticks, rest = len(match.group(1)), match.group(2).strip()
        if outer is None:
            outer, outer_info = (ticks, index), rest
        elif ticks >= outer[0] and not rest:
            if inner is not None:
                problems.append((inner[1], INNER_NEVER_CLOSED))
            if not outer_info.startswith("{") and not any(text.strip() for text in lines[outer[1] + 1 : index]):
                problems.append((outer[1], "empty code block"))
            outer = inner = None
        elif ticks < outer[0]:
            if inner is None:
                inner = (ticks, index)
            elif ticks == inner[0] and not rest:
                inner = None
    if outer is not None:
        problems.append((outer[1], "code fence is never closed"))
    return problems


def _collect() -> tuple[list[str], list[str]]:
    roles: list[str] = []
    fences: list[str] = []
    for path in _source_files():
        name = path.relative_to(ROOT).as_posix()
        for start, lines in _markdown_blocks(path):
            outside: list[str] = []
            open_ticks = 0
            for line in lines:
                match = FENCE.match(line)
                if match:
                    ticks, rest = len(match.group(1)), match.group(2).strip()
                    if open_ticks == 0:
                        open_ticks = ticks
                    elif ticks >= open_ticks and not rest:
                        open_ticks = 0
                    outside.append("")
                else:
                    outside.append("" if open_ticks else line)
            for offset, line in enumerate(outside):
                for match in RST_ROLE.finditer(line):
                    roles.append(f"{name}:{start + offset} {match.group(0)}")
            fences.extend(f"{name}:{start + index} {why}" for index, why in _fence_problems(lines))
    return roles, fences


RST_ROLES, BAD_FENCES = _collect()


@pytest.mark.required
def test_the_collector_reads_the_docs():
    """Guard against the file walk silently reading nothing, which would pass every check below."""
    files = list(_source_files())
    assert len(files) > 50, f"only found {len(files)} Markdown sources; the walk is likely broken"
    assert any(start > 1 for path in files if path.suffix == ".py" for start, _ in _markdown_blocks(path))


@pytest.mark.required
def test_the_checks_are_wired_up():
    """Guard against either pattern silently matching nothing."""
    assert RST_ROLE.search("see :class:`~dataeval.quality.Duplicates` here")
    assert RST_ROLE.search("see :meth:`.Metadata.agg`")
    assert not RST_ROLE.search("see {class}`~dataeval.quality.Duplicates` here"), "the MyST spelling must pass"
    assert not RST_ROLE.search("write `:class:` in a docstring"), "a role named inside a code span must pass"

    wrong_close = ["````{note}", "text", "```python", "import logging", "````", "", "```", "", "```"]
    assert {why for _, why in _fence_problems(wrong_close)} == {INNER_NEVER_CLOSED, "empty code block"}
    assert _fence_problems(["````{note}", "text", "```python", "import logging", "```", "````"]) == []
    assert _fence_problems(["```python", "x = 1", "```"]) == []
    directive = ["```{include} ../CHANGELOG.md", "```"]
    assert _fence_problems(directive) == [], "a body-less MyST directive is not an empty code block"
    assert [why for _, why in _fence_problems(["```python", "x = 1"])] == ["code fence is never closed"]


@pytest.mark.required
def test_no_rst_roles_in_markdown():
    """Every role in a guide or notebook cell is spelled ``{role}`Target```, which MyST parses.

    One assertion listing every offender, as ``test_no_malformed_role_markup`` does: the intended
    state is zero offenders, and a parametrized test with nothing to parametrize reports as skipped.
    """
    assert not RST_ROLES, (
        "reST role spelling in Markdown:\n"
        + "\n".join(f"  {entry}" for entry in RST_ROLES)
        + "\nMyST does not parse :class:`X`, so it renders as literal text and the link is silently dead. "
        "Spell it {class}`X`."
    )


@pytest.mark.required
def test_no_malformed_code_fences():
    """Every fence closes the block it opened, including a short fence nested in a longer one."""
    assert not BAD_FENCES, (
        "Malformed code fences:\n"
        + "\n".join(f"  {entry}" for entry in BAD_FENCES)
        + "\nA fence of N backticks is closed only by N or more. Close an inner example with its own "
        "backticks first, then close the outer fence."
    )
