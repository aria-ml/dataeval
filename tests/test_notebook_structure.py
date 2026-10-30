"""Enforce the structure that ``docs/STYLE_GUIDE.md`` sets for the tutorials and how-tos.

The notebooks in ``docs/source/notebooks`` are ``py:percent`` scripts. ``tt_*`` files are tutorials and ``h2_*``
files are how-tos. These checks read their Markdown cells and fail on structure that a reviewer would otherwise
catch by eye:

* Headings: one H1 and no skipped levels, sentence case, no numbering, no bold, em dash or trailing colon, and
  the fixed template strings.
* Titles: an imperative phrase for tutorials and how-tos.
* Tutorials: the opening and three metadata lines, and the template sections in order.
* How-tos: no boilerplate or teaching headings, at most five steps and a lead of at most 120 words.
* Next steps: the entry form, the counts the guide allows and a link text that is the title of its target.
* Voice: no em dash outside a Next steps entry, contraction, first person, "corpus", stress word or italics, no bold
  label or bold table row label, and sentence case in table column headings.
* Width: no Markdown line over 120 columns, counting the ``# `` of the file. A table row, a metadata line of a
  tutorial and a line that is one unbreakable token, such as a URL, pass.

A how-to over 600 words warns and does not fail, because the guide leaves that limit to review.

The voice checks skip code spans, roles, link targets, math and fenced code. How plain a sentence is, whether bold marks
the first mention of a term, descriptions and what a section teaches stay with review.
"""

import functools
import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

pytestmark = pytest.mark.infrastructure

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "docs" / "source" / "notebooks"

MAX_LEAD_WORDS = 120
MAX_STEPS = 5
MAX_HOWTO_WORDS = 600
NEXT_STEPS_RANGE = {"tutorial": (2, 4), "how-to": (2, 3)}
MAX_CONCEPT_LINKS = 2
MAX_NOTEBOOK_LINKS = 2
MAX_LINE_COLUMNS = 120

# The tutorial sections in the order the guide gives them. "What you learned" is optional.
TUTORIAL_SECTIONS = [
    "What you will do",
    "What you will learn",
    "What you will need",
    "Introduction",
    "Setup",
    "Conclusion",
    "What you learned",
    "Next steps",
    "On your own",
]
TUTORIAL_OPTIONAL = {"What you learned"}
# Spellings that stand for a template heading, mapped to the heading to use.
RENAMED_HEADINGS = {
    "What you'll do": "What you will do",
    "What you'll learn": "What you will learn",
    "What you'll need": "What you will need",
    "Background": "Introduction",
    "Getting started": "Setup",
}
# The lead replaces these in a how-to.
HOWTO_BOILERPLATE = {"Problem statement", "When to use", "What you will need", "Getting started"}
HOWTO_TEACHING = {"Introduction", "Setup", "Background", "Conclusion", "Summary", "What you learned", "Key takeaways"}

SMALL_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "for",
    "nor",
    "of",
    "in",
    "on",
    "at",
    "to",
    "by",
    "vs",
    "vs.",
    "with",
    "from",
    "as",
    "into",
    "is",
    "are",
    "be",
}
WORD = re.compile(r"[A-Za-z0-9`'\-]+")
LINK_TARGET = re.compile(r"\]\([^)]*\)")
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
NUMBERED = re.compile(r"^(?:(?:Step|Question|Part|Phase) \d+|\d+\.\s)")
GERUND_TITLE = re.compile(r"^(?:[A-Za-z]{3,}ing\b|Introduction to\b)")
METADATA_LINES = {
    "estimated time": re.compile(r"^Estimated time to complete: \d+ minutes?$"),
    "ML stages": re.compile(r"^Relevant ML stages: \S"),
    "personas": re.compile(r"^Relevant personas: \S"),
}
METADATA_PREFIXES = ("Estimated time", "Relevant ML stages", "Relevant personas")
ENTRY = re.compile(r"^- \[(?P<title>[^\]]+)\]\((?P<target>[^)]+)\) — (?P<description>\S.*)$")

# The voice rules read prose, so a line loses what is not prose before they look at it.
NOT_PROSE = [
    re.compile(p) for p in (r"https?://\S+", r"\]\([^)]*\)", r"\{[a-z]+\}`[^`]*`", r"`[^`\n]*`", r"\$[^$\n]+\$")
]
ENTRY_SEPARATOR = re.compile(r"(?<=\)) —")
VOICE_RULES = [
    ("em dash, the only one allowed is the separator of a Next steps entry", re.compile("—")),
    (
        "contraction, write it out",
        re.compile(
            r"\b(?:\w+n[’']t|\w+[’'](?:ll|re|ve|d|m)|(?:it|that|there|here|what|who|he|she|let|how)[’']s)\b", re.I
        ),
    ),
    ("first person, address the reader as 'you'", re.compile(r"\b(?:[Ww]e|us|[Oo]urs?|[Oo]urselves)\b")),
    ("'corpus', say dataset", re.compile(r"\bcorpus\b|\bcorpora\b", re.I)),
    ("stress word, delete it", re.compile(r"\b(?:simply|genuinely|quietly|exactly)\b|\bon your behalf\b", re.I)),
    (
        "italics, delete the markup",
        re.compile(
            r"(?<![\w*\\])\*(?![\s*])[^*\n]+?(?<![\s*\\])\*(?![\w*])|(?<![\w_])_(?![\s_])[^_\n]+?(?<![\s_])_(?![\w_])"
        ),
    ),
    (
        "bold label, bold only a term at its first mention",
        re.compile(r"^\s*(?:(?:[-*+]|\d+\.)\s+)?\*\*[^*]+?(?::\*\*|\*\*\s*(?::|—))"),
    ),
    ("bold row label in a table", re.compile(r"^\s*\|\s*\*\*")),
]
UNBREAKABLE = re.compile(r"^\s*(?:(?:[-*+]|\d+[.)])\s+)?\S+$")
TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$")


@dataclass(frozen=True)
class Line:
    """One line of a Markdown cell. ``code_before`` marks the first line after a code cell."""

    text: str
    code_before: bool = False


@dataclass(frozen=True)
class Notebook:
    """The Markdown of a notebook, as lines in order, with its headings."""

    name: str
    kind: str
    lines: tuple[Line, ...]
    headings: tuple[tuple[int, str], ...]

    @property
    def h1(self) -> str | None:
        return next((text for level, text in self.headings if level == 1), None)

    def sections(self, level: int) -> list[str]:
        return [text for lvl, text in self.headings if lvl == level]


def parse(name: str, text: str) -> Notebook:
    """Read the Markdown cells of a ``py:percent`` script. The tutorial or how-to kind comes from the name."""
    lines: list[Line] = []
    headings: list[tuple[int, str]] = []
    code_before = False
    for cell in re.split(r"^# %%", text, flags=re.MULTILINE)[1:]:
        header, _, body = cell.partition("\n")
        if "[markdown]" not in header:
            code_before = True
            continue
        fenced = False
        for raw in body.split("\n"):
            line = raw[2:] if raw.startswith("# ") else ("" if raw == "#" else raw)
            lines.append(Line(line, code_before))
            code_before = False
            if line.lstrip().startswith("```"):
                fenced = not fenced
            match = None if fenced else HEADING.match(line)
            if match:
                headings.append((len(match.group(1)), match.group(2).replace("’", "'")))
    return Notebook(name, "tutorial" if name.startswith("tt_") else "how-to", tuple(lines), tuple(headings))


def _load() -> list[Notebook]:
    paths = sorted(p for p in NOTEBOOK_DIR.glob("*.py") if p.name.startswith(("tt_", "h2_")))
    return [parse(p.name, p.read_text()) for p in paths]


NOTEBOOKS = _load()


def _words(line: str) -> int:
    return len(WORD.findall(LINK_TARGET.sub("]", line)))


def _title_cased(heading: str) -> bool:
    """Whether a heading of three or more words capitalizes most of the words after the first."""
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", re.sub(r"`[^`]*`", "", heading))
    if len(words) < 3:
        return False
    rest = [
        w for w in words[1:] if w.lower() not in SMALL_WORDS and not w.isupper() and not re.search(r"[a-z][A-Z]", w)
    ]
    return bool(rest) and sum(w[0].isupper() for w in rest) / len(rest) >= 0.6


def _section(nb: Notebook, heading: str, level: int = 2) -> list[str]:
    """The text lines under a heading, up to the next heading or the next code cell."""
    out: list[str] = []
    inside = False
    for line in nb.lines:
        match = HEADING.match(line.text)
        if inside and (match or line.code_before):
            break
        if match and (len(match.group(1)), match.group(2)) == (level, heading):
            inside = True
        elif inside:
            out.append(line.text)
    return out


def lead_words(nb: Notebook) -> int:
    """Words between the H1 and the first heading or code cell after it."""
    words, seen = 0, False
    for line in nb.lines:
        if seen and (line.code_before or HEADING.match(line.text)):
            break
        match = HEADING.match(line.text)
        if match and len(match.group(1)) == 1:
            seen = True
        elif seen and line.text.strip() and not line.text.startswith(("(", "```", ":::")):
            words += _words(line.text)
    return words


def prose_words(nb: Notebook) -> int:
    """Words of Markdown outside headings and the Next steps list."""
    words, skipping = 0, False
    for line in nb.lines:
        match = HEADING.match(line.text)
        if match and len(match.group(1)) == 2:
            skipping = match.group(2) == "Next steps"
        if match or skipping:
            continue
        words += _words(line.text)
    return words


def heading_problems(nb: Notebook) -> list[str]:
    problems: list[str] = []
    if len(nb.sections(1)) != 1:
        problems.append(f"{len(nb.sections(1))} H1 headings, expected one")
    previous = 0
    for level, text in nb.headings:
        if previous and level > previous + 1:
            problems.append(f"heading level skips from H{previous} to H{level}: {text}")
        previous = level
        if _title_cased(text):
            problems.append(f"not sentence case: {text}")
        if NUMBERED.match(text):
            problems.append(f"numbered heading, name the action and let position carry the order: {text}")
        if "**" in text or "—" in text or text.endswith(":"):
            problems.append(f"bold, em dash or trailing colon in a heading: {text}")
        if nb.kind == "tutorial" and level == 2 and text in RENAMED_HEADINGS:
            problems.append(f"use the template heading '{RENAMED_HEADINGS[text]}' instead of '{text}'")
    titles = {t.lower(): t for t in TUTORIAL_SECTIONS}
    for text in nb.sections(2):
        canonical = titles.get(text.lower())
        if nb.kind == "tutorial" and canonical and canonical != text:
            problems.append(f"template heading is spelled '{canonical}', not '{text}'")
    title = nb.h1 or ""
    if GERUND_TITLE.match(title):
        problems.append(f"a {nb.kind} title is an imperative phrase, not a gerund: {title}")
    return problems


def tutorial_problems(nb: Notebook) -> list[str]:
    problems: list[str] = []
    present = nb.sections(2)
    missing = [s for s in TUTORIAL_SECTIONS if s not in present and s not in TUTORIAL_OPTIONAL]
    problems.extend(f"missing section: {s}" for s in missing)
    positions = [TUTORIAL_SECTIONS.index(s) for s in present if s in TUTORIAL_SECTIONS]
    if positions != sorted(positions):
        order = ", ".join(s for s in present if s in TUTORIAL_SECTIONS)
        problems.append(f"template sections are out of order: {order}")
    opening: list[str] = []
    found: set[str] = set()
    after_h1 = metadata_started = False
    for line in nb.lines:
        match = HEADING.match(line.text)
        if match:
            if len(match.group(1)) == 1:
                after_h1 = True
                continue
            if after_h1:
                break
        if not after_h1 or not line.text.strip():
            continue
        if line.text.startswith(METADATA_PREFIXES):
            found.update(key for key, pattern in METADATA_LINES.items() if pattern.match(line.text))
        elif not found and not metadata_started:
            opening.append(line.text)
        metadata_started = metadata_started or line.text.startswith(METADATA_PREFIXES)
    if not opening:
        problems.append("no opening sentence between the title and the metadata lines")
    problems.extend(f"metadata line missing or malformed: {key}" for key in METADATA_LINES if key not in found)
    return problems


def howto_problems(nb: Notebook) -> list[str]:
    problems: list[str] = []
    for level, text in nb.headings:
        if level in (2, 3) and text in HOWTO_BOILERPLATE:
            problems.append(f"boilerplate heading, the lead replaces it: {text}")
        elif level == 2 and text in HOWTO_TEACHING:
            problems.append(f"tutorial section in a how-to: {text}")
    steps = [t for t in nb.sections(2) if t != "Next steps"]
    if len(steps) > MAX_STEPS:
        problems.append(f"{len(steps)} steps, at most {MAX_STEPS}: a longer how-to is two how-tos")
    lead = lead_words(nb)
    if lead > MAX_LEAD_WORDS:
        problems.append(f"lead is {lead} words, at most {MAX_LEAD_WORDS}")
    return problems


def _slug(heading: str) -> str:
    """The anchor MyST gives a heading: lower case, punctuation dropped, each space a hyphen."""
    return re.sub(r"\s", "-", re.sub(r"[^\w\s-]", "", heading.replace("`", "").lower()))


@functools.cache
def _headings_of(path: Path) -> tuple[tuple[int, str], ...]:
    if path.suffix == ".py":
        return parse(path.name, path.read_text()).headings
    headings: list[tuple[int, str]] = []
    fence = 0
    for line in path.read_text().split("\n"):
        ticks = re.match(r"^\s*(`{3,})(.*)$", line)
        if fence:
            fence = 0 if ticks and len(ticks[1]) >= fence and not ticks[2].strip() else fence
        elif ticks:
            fence = len(ticks[1])
        elif match := HEADING.match(line):
            headings.append((len(match[1]), match[2].replace("’", "'")))
    return tuple(headings)


def expected_title(target: str) -> str | None:
    """The link text for a Next steps target: the heading an anchor names, else the H1 of the page.

    Only a local notebook or Markdown page has a title to check. Any other target returns None.
    """
    page, _, anchor = target.partition("#")
    path = (NOTEBOOK_DIR / page).resolve()
    if not page.endswith((".py", ".md")) or not path.is_file():
        return None
    headings = _headings_of(path)
    named = next((text for _, text in headings if anchor and _slug(text) == anchor), None)
    return named or next((text for level, text in headings if level == 1), None)


def next_steps_problems(nb: Notebook) -> list[str]:
    if "Next steps" not in nb.sections(2):
        return ["missing section: Next steps"]
    entries: list[str] = []
    for text in _section(nb, "Next steps"):
        if text.startswith("- "):
            entries.append(text.rstrip())
        elif text.startswith(" ") and entries:
            entries[-1] += " " + text.strip()
    problems: list[str] = []
    low, high = NEXT_STEPS_RANGE[nb.kind]
    if not low <= len(entries) <= high:
        problems.append(f"{len(entries)} entries, a {nb.kind} lists {low} to {high}")
    concepts = notebooks = 0
    for entry in entries:
        match = ENTRY.match(entry)
        if not match:
            problems.append(f"entry is not '- [Title](target) — Description.': {entry[:80]}")
            continue
        if not match["description"].endswith("."):
            problems.append(f"description is not a sentence ending in a period: {entry[:80]}")
        title = expected_title(match["target"])
        if title and match["title"] != title:
            problems.append(f"link text '{match['title']}' is not the title '{title}' of {match['target']}")
        target = match["target"].split("#")[0]
        concepts += "/concepts/" in target
        notebooks += target.endswith(".py")
    if concepts > MAX_CONCEPT_LINKS:
        problems.append(f"{concepts} concept pages, at most {MAX_CONCEPT_LINKS}")
    if notebooks > MAX_NOTEBOOK_LINKS:
        problems.append(f"{notebooks} notebooks, at most {MAX_NOTEBOOK_LINKS}")
    return problems


def _markdown_lines(nb: Notebook) -> list[tuple[str, str]]:
    """Each Markdown line outside a code fence with the H2 section it sits in."""
    out: list[tuple[str, str]] = []
    section, fenced = "", False
    for line in nb.lines:
        text = line.text
        if text.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced:
            match = HEADING.match(text)
            section = match.group(2) if match and len(match.group(1)) == 2 else section
            out.append((text, section))
    return out


def _prose(text: str, section: str) -> str:
    """A line without the code, roles, link targets and math that the voice rules do not cover."""
    text = ENTRY_SEPARATOR.sub("", text, count=1) if section == "Next steps" else text
    for pattern in NOT_PROSE:
        text = pattern.sub("", text)
    return text


def voice_problems(nb: Notebook) -> list[str]:
    """Voice and emphasis rules of the guide that a pattern can check, and table headings in title case."""
    lines = _markdown_lines(nb)
    problems: list[str] = []
    for text, section in lines:
        if HEADING.match(text):
            continue
        prose = _prose(text, section)
        problems.extend(f"{rule}: {text.strip()[:90]}" for rule, pattern in VOICE_RULES if pattern.search(prose))
    for (head, _), (separator, _) in zip(lines, lines[1:], strict=False):
        if head.lstrip().startswith("|") and TABLE_SEPARATOR.match(separator):
            cells = [cell.strip() for cell in head.strip().strip("|").split("|")]
            problems.extend(f"table column heading is not sentence case: {c}" for c in cells if _title_cased(c))
    return problems


def width_problems(nb: Notebook) -> list[str]:
    """Markdown lines over 120 columns in the file, which counts the ``# `` that ``Line.text`` leaves out.

    A table row passes, as it does for markdownlint, and so does a line that is one unbreakable token, such as a URL.
    The three metadata lines of a tutorial stay on one line each, so they pass too.
    """
    problems: list[str] = []
    for line in nb.lines:
        raw = f"# {line.text}" if line.text else "#"
        exempt = line.text.startswith(("|", *METADATA_PREFIXES)) or UNBREAKABLE.match(line.text)
        if len(raw) > MAX_LINE_COLUMNS and not exempt:
            problems.append(f"{len(raw)} columns, wrap at {MAX_LINE_COLUMNS}: {line.text.strip()[:70]}")
    return problems


def _report(check: str, problems: dict[str, list[str]]) -> str:
    lines = [f"{check}:"]
    for name, items in problems.items():
        lines.extend(f"  {name}: {item}" for item in items)
    return "\n".join(lines) + "\nSee docs/STYLE_GUIDE.md for the structure and the heading rules."


def _failures(check: Callable[[Notebook], list[str]], kind: str | None = None) -> dict[str, list[str]]:
    results = {nb.name: check(nb) for nb in NOTEBOOKS if kind in (None, nb.kind)}
    return {name: items for name, items in results.items() if items}


@pytest.mark.required
def test_the_collector_reads_the_notebooks():
    """Guard against the file walk silently reading nothing, which would pass every check below."""
    kinds = {nb.kind for nb in NOTEBOOKS}
    assert len(NOTEBOOKS) > 30, f"only found {len(NOTEBOOKS)} notebooks; the walk is likely broken"
    assert kinds == {"tutorial", "how-to"}
    assert all(nb.h1 and nb.headings and nb.lines for nb in NOTEBOOKS)


@pytest.mark.required
def test_the_checks_are_wired_up():
    """Guard against a check silently matching nothing, with a notebook built to break each rule."""
    assert _title_cased("Marginal Class Probability Shift")
    assert not _title_cased("Marginal class probability shift")
    assert not _title_cased("Compute the `MMD` statistic")
    assert not _title_cased("Build a MetadataLike object from a DataFrame")
    assert not _title_cased("Respond to a StatsInvalidatedWarning")
    assert _title_cased("Build a MetadataLike Object From a DataFrame")
    for name in ("h2_example.py", "tt_example.py"):
        assert heading_problems(parse(name, "# %% [markdown]\n# # Measure things\n")) == []
    assert NUMBERED.match("Question 1: the vehicle type")
    assert NUMBERED.match("Step 3 Choose the level")
    assert not NUMBERED.match("Choose your level with `at()`")
    assert METADATA_LINES["estimated time"].match("Estimated time to complete: 15 minutes")
    assert not METADATA_LINES["estimated time"].match("Estimated time to complete: 10-15 minutes")

    steps = "".join(f"\n# %% [markdown]\n# ## Run Every Step {i}\n" for i in range(6))
    entries = "".join(f"# - [{n}](./h2_{n}.py) — Entry {n}.\n" for n in "abc")
    entries += "# - [C](../concepts/C.md) — Concept.\n# - [D](../concepts/D.md) — Concept.\n"
    entries += "# - [E](../concepts/E.md) — Concept.\n# - [F](../concepts/F.md) Missing the dash\n"
    bad_howto = parse(
        "h2_example.py",
        f"# %% [markdown]\n# # Measuring data\n#\n# {' '.join(['word'] * 130)}\n#\n# ## When to use\n#\n# ## Summary\n"
        f"{steps}\n# %% [markdown]\n# ## Next steps\n#\n{entries}",
    )
    found = heading_problems(bad_howto) + howto_problems(bad_howto) + next_steps_problems(bad_howto)
    for expected in (
        "a how-to title is an imperative phrase, not a gerund",
        "not sentence case: Run Every Step 0",
        "boilerplate heading, the lead replaces it: When to use",
        "tutorial section in a how-to: Summary",
        "8 steps, at most 5",
        "lead is 130 words, at most 120",
        "7 entries, a how-to lists 2 to 3",
        "3 concept pages, at most 2",
        "3 notebooks, at most 2",
        "entry is not '- [Title](target) — Description.': - [F]",
    ):
        assert any(expected in problem for problem in found), f"{expected!r} not reported in {found}"

    bad_tutorial = parse(
        "tt_example.py",
        "# %% [markdown]\n# # Introduction to things\n\n# %% [markdown]\n# ## Conclusion\n\n"
        "# %% [markdown]\n# ## Introduction\n\n# %% [markdown]\n# ## What you'll do\n",
    )
    found = heading_problems(bad_tutorial) + tutorial_problems(bad_tutorial)
    for expected in (
        "a tutorial title is an imperative phrase, not a gerund",
        "use the template heading 'What you will do' instead of 'What you'll do'",
        "missing section: Setup",
        "template sections are out of order: Conclusion, Introduction",
        "no opening sentence",
        "metadata line missing or malformed: personas",
    ):
        assert any(expected in problem for problem in found), f"{expected!r} not reported in {found}"

    assert _slug("Feature extractors: a dependency of both layers") == "feature-extractors-a-dependency-of-both-layers"
    assert _slug("Bayes error rate — the irreducible floor") == "bayes-error-rate--the-irreducible-floor"
    assert expected_title("https://example.com/page.md") is None
    real = NOTEBOOKS[0]
    section = real.headings[1][1]
    assert expected_title(f"./{real.name}") == real.h1
    assert expected_title(f"./{real.name}#{_slug(section)}") == section, "an anchor names its heading"
    assert expected_title(f"./{real.name}#no-such-heading") == real.h1, "an unknown anchor falls back to the page title"
    link = "# %% [markdown]\n# # Link pages\n\n# %% [markdown]\n# ## Next steps\n#\n# - [{text}](./{name}) — Entry.\n"
    wrong = parse("h2_link.py", link.format(text="Wrong text", name=real.name))
    assert f"link text 'Wrong text' is not the title '{real.h1}' of ./{real.name}" in next_steps_problems(wrong)
    right = parse("h2_link.py", link.format(text=real.h1, name=real.name))
    assert not any("link text" in problem for problem in next_steps_problems(right))


@pytest.mark.required
def test_the_voice_checks_are_wired_up():
    """Guard against a voice rule silently matching nothing, and against flagging what is not prose."""
    bad = parse(
        "h2_voice.py",
        "# %% [markdown]\n# # Sound right\n#\n"
        "# We don't write this way — it is *loud* and _also loud_, so simply stop. A corpus is not a dataset.\n"
        "# - **Label:** a bold list label\n"
        "#\n# | **Row label** | Used With Python |\n# | --- | --- |\n# | a | b |\n",
    )
    found = voice_problems(bad)
    for expected in (
        "em dash",
        "contraction, write it out: We don't",
        "first person",
        "'corpus', say dataset",
        "stress word",
        "italics, delete the markup",
        "bold label",
        "bold row label in a table",
        "table column heading is not sentence case: Used With Python",
    ):
        assert any(expected in problem for problem in found), f"{expected!r} not reported in {found}"
    assert sum("italics" in problem for problem in found) == 1, "one line holds both markups"

    good = parse(
        "h2_voice.py",
        "# %% [markdown]\n# # Sound right\n#\n"
        "# The dataset's page has `don't`, {class}`.Coverage` and [a link](./a_b_c.py) to read.\n"
        "# A **codebook** is a term that the page defines, and 2 * 3 * 4 is a product. Read https://example.com/a_b_c.\n"
        "#\n# ```text\n# We don't — *loud*\n# ```\n"
        "#\n# | Column heading | Another heading |\n# | --- | --- |\n# | a | b |\n"
        "#\n# ## Next steps\n#\n# - [Title](./x.py) — Description.\n",
    )
    assert voice_problems(good) == []


@pytest.mark.required
def test_the_width_check_is_wired_up():
    """Guard against the width check matching nothing, or flagging a table row, a lone URL or a line at the limit."""
    words = " ".join(["word"] * 30)
    url_line = f"word https://example.com/{'a' * 110}"
    text = (
        "# %% [markdown]\n# # Wrap lines\n#\n"
        f"# {'x' * 118}\n"  # 120 columns with the '# ' is allowed
        f"# {words}\n"
        f"# | {words} | y |\n"
        f"# https://example.com/{'a' * 130}\n"
        f"# - https://example.com/{'a' * 130}\n"
        f"# Relevant personas: {words}\n"
        f"# {url_line}\n"
    )
    assert width_problems(parse("h2_wrap.py", text)) == [
        f"{len(words) + 2} columns, wrap at 120: {words[:70]}",
        f"{len(url_line) + 2} columns, wrap at 120: {url_line[:70]}",
    ]


@pytest.mark.required
def test_markdown_cells_wrap_at_120_columns():
    """No Markdown line of a tutorial or how-to is over 120 columns, as for the Markdown pages."""
    failures = _failures(width_problems)
    assert not failures, _report("Lines over 120 columns", failures)


@pytest.mark.required
def test_voice_follows_the_style_guide():
    """Prose has no em dash, contraction, first person, 'corpus', stress word or italics, and uses bold plainly."""
    failures = _failures(voice_problems)
    assert not failures, _report("Voice problems", failures)


@pytest.mark.required
def test_headings_follow_the_style_guide():
    """Every heading is sentence case, unnumbered and plain, and the template headings use the fixed strings."""
    failures = _failures(heading_problems)
    assert not failures, _report("Heading problems", failures)


@pytest.mark.required
def test_tutorials_follow_the_template():
    """Every tutorial has its opening, three metadata lines and the template sections in order."""
    failures = _failures(tutorial_problems, "tutorial")
    assert not failures, _report("Tutorial template problems", failures)


@pytest.mark.required
def test_howtos_follow_the_template():
    """Every how-to has a lead of at most 120 words, at most five steps and none of the old boilerplate."""
    failures = _failures(howto_problems, "how-to")
    assert not failures, _report("How-to template problems", failures)


@pytest.mark.required
def test_next_steps_follow_the_rubric():
    """Every notebook ends with Next steps, in the entry form and within the counts the guide allows."""
    failures = _failures(next_steps_problems)
    assert not failures, _report("Next steps problems", failures)


@pytest.mark.required
def test_howtos_stay_near_the_length_target():
    """A how-to over 600 words warns, because the guide leaves the limit to review."""
    counts = {nb.name: prose_words(nb) for nb in NOTEBOOKS if nb.kind == "how-to"}
    long = {name: [f"{words} words"] for name, words in counts.items() if words > MAX_HOWTO_WORDS}
    if long:
        warnings.warn(_report(f"How-tos over {MAX_HOWTO_WORDS} words of prose", long), UserWarning, stacklevel=2)
