"""Enforce that literature-backed public API objects cite their sources.

DataEval's documentation makes claims about *why* a result can be trusted, and
those claims rest on published derivations. A citation that lives only in a
concept page drifts away from the code it describes; a citation in the
docstring travels with the implementation and shows up in the API reference
automatically.

This test pins the set of objects that are known to implement a published
method. Each must carry a numpydoc ``References`` section. It does not require
citations from everything — several DataEval functions are original or are thin
geometric primitives with no single source — so the list below is explicit
rather than derived.

To extend coverage, add a symbol to ``REQUIRES_REFERENCES``. That is the
intended direction of travel: the "Known gaps" section of
``docs/source/concepts/ValidationAndTrust.md`` records that most public symbols
still lack a citation, and this list is how that gap gets closed without
silently regressing.
"""

import importlib
import inspect
import re

import pytest

# (module suffix, symbol) pairs whose docstrings must carry a References section.
REQUIRES_REFERENCES: list[tuple[str, str]] = [
    ("bias", "Balance"),
    ("bias", "Diversity"),
    ("bias", "Parity"),
    ("core", "ber_knn"),
    ("core", "ber_mst"),
    ("core", "cluster"),
    ("core", "completeness"),
    ("core", "coverage_adaptive"),
    ("core", "coverage_naive"),
    ("core", "dhash"),
    ("core", "dhash_d4"),
    ("core", "divergence_fnn"),
    ("core", "divergence_mst"),
    ("core", "label_parity"),
    ("core", "mutual_info"),
    ("core", "parity"),
    ("core", "phash"),
    ("core", "phash_d4"),
    ("core", "rank_hdbscan_complexity"),
    ("core", "rank_hdbscan_distance"),
    ("core", "rank_kmeans_complexity"),
    ("core", "rank_kmeans_distance"),
    ("core", "rank_knn"),
    ("core", "rank_result_stratified"),
    ("core", "uap"),
    ("performance", "Sufficiency"),
    ("quality", "Duplicates"),
    ("quality", "Outliers"),
    ("scope", "Coverage"),
    ("shift", "DriftMMD"),
    ("shift", "DriftReconstruction"),
    ("shift", "DriftUnivariate"),
    ("shift", "OODReconstruction"),
]

REFERENCES_HEADING = re.compile(r"^References\s*\n-+\s*$", re.MULTILINE)
# A citation is only useful if it identifies something: a URL, a year, or a DOI.
CITATION_CONTENT = re.compile(r"https?://|\(\d{4}\)|doi:", re.IGNORECASE)


def _resolve(module_suffix: str, symbol: str) -> object:
    module = importlib.import_module(f"dataeval.{module_suffix}")
    assert hasattr(module, symbol), f"dataeval.{module_suffix} has no attribute '{symbol}'"
    return getattr(module, symbol)


@pytest.mark.required
@pytest.mark.parametrize(("module_suffix", "symbol"), REQUIRES_REFERENCES)
def test_documents_its_sources(module_suffix: str, symbol: str) -> None:
    """Every listed object exposes a References section naming its sources."""
    obj = _resolve(module_suffix, symbol)
    doc = inspect.getdoc(obj) or ""

    assert doc, f"dataeval.{module_suffix}.{symbol} has no docstring"

    match = REFERENCES_HEADING.search(doc)
    assert match, (
        f"dataeval.{module_suffix}.{symbol} implements a published method but its "
        f"docstring has no 'References' section. Add one, or remove the symbol from "
        f"REQUIRES_REFERENCES in {__file__} if it is not literature-backed."
    )

    body = doc[match.end() :]
    # Stop at the next numpydoc section heading, if any.
    next_section = re.search(r"\n[A-Z][A-Za-z ]+\n-+\n", body)
    if next_section:
        body = body[: next_section.start()]

    assert body.strip(), f"dataeval.{module_suffix}.{symbol} has an empty References section"
    assert CITATION_CONTENT.search(body), (
        f"dataeval.{module_suffix}.{symbol} has a References section with no "
        f"identifiable citation (expected a URL, a (year), or a doi)"
    )


@pytest.mark.required
def test_reference_list_has_no_duplicates() -> None:
    """The enforcement list is a set, so a symbol cannot be listed twice."""
    assert len(REQUIRES_REFERENCES) == len(set(REQUIRES_REFERENCES))


@pytest.mark.required
def test_reference_list_is_sorted() -> None:
    """Keep the list ordered so additions produce readable diffs."""
    assert sorted(REQUIRES_REFERENCES) == REQUIRES_REFERENCES, "REQUIRES_REFERENCES should stay sorted"
