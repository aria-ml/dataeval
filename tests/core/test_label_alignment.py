from collections.abc import Iterable

import pytest
from pydantic import ValidationError

from dataeval import Ontology
from dataeval.core import label_alignment, label_reconciliation
from dataeval.protocols import Matcher
from dataeval.types import Correspondence, OntologyConcept


def target_ontology() -> Ontology:
    """Reference ontology with two roots and two levels of granularity.

    vehicle
    ├── car
    │   ├── sedan
    │   └── suv
    └── truck
    animal
    ├── dog
    └── cat
    """
    return Ontology.from_hierarchy({
        "vehicle": {"car": {"sedan": None, "suv": None}, "truck": None},
        "animal": {"dog": None, "cat": None},
    })


def relations_of(result, source: str) -> dict[str, str]:
    """Map ``target -> relation`` for the correspondences of one source concept."""
    return {c.target: c.relation for c in result["correspondences"] if c.source == source}


class StubMatcher:
    """A dependency-free :class:`Matcher` emitting fixed equivalence proposals."""

    def __init__(self, *proposals: tuple[str, str, float]) -> None:
        self._proposals = proposals

    def __call__(self, source: Iterable[OntologyConcept], target: Iterable[OntologyConcept]) -> list[Correspondence]:
        return [
            Correspondence(source=s, target=t, relation="equivalent", confidence=c, matcher="stub")
            for s, t, c in self._proposals
        ]


@pytest.mark.required
class TestCorrespondence:
    def test_frozen(self):
        c = Correspondence(source="a", target="b", relation="equivalent")
        with pytest.raises(ValidationError):
            c.confidence = 0.5

    def test_defaults(self):
        c = Correspondence(source="a", target="b", relation="narrower")
        assert c.confidence == 1.0
        assert c.matcher == ""

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            Correspondence(source="a", target="b", relation="equivalent", confidence=1.5)

    def test_relation_must_be_known(self):
        with pytest.raises(ValidationError):
            Correspondence(source="a", target="b", relation="sameas")  # type: ignore[arg-type]


@pytest.mark.required
class TestAlign:
    def test_exact_equivalence_anchoring(self):
        result = label_alignment(["car", "dog"], target_ontology())
        assert relations_of(result, "car")["car"] == "equivalent"
        assert relations_of(result, "dog")["dog"] == "equivalent"
        equiv = next(c for c in result["correspondences"] if c.source == "car" and c.relation == "equivalent")
        assert equiv.confidence == 1.0
        assert equiv.matcher == "exact"

    def test_narrower_propagation_to_nearest_anchored_ancestor(self):
        source = Ontology.from_hierarchy({"car": {"hatchback": None}})
        result = label_alignment(source, target_ontology())
        narrower = next(c for c in result["correspondences"] if c.source == "hatchback")
        assert (narrower.relation, narrower.target, narrower.matcher) == ("narrower", "car", "structural")
        assert result["class_remap"]["hatchback"] == "car"

    def test_broader_diagnostics_for_uncovered_target_children(self):
        # source has only "car"; target's children sedan/suv are finer and uncovered
        result = label_alignment(["car"], target_ontology())
        assert relations_of(result, "car") == {"car": "equivalent", "sedan": "broader", "suv": "broader"}

    def test_broader_excluded_from_class_remap(self):
        result = label_alignment(["car"], target_ontology())
        assert dict(result["class_remap"]) == {"car": "car"}
        broader_targets = {c.target for c in result["correspondences"] if c.relation == "broader"}
        assert broader_targets.isdisjoint(result["class_remap"].values())

    def test_covered_sibling_suppresses_broader(self):
        # sedan is carried over, so only the *uncovered* sibling suv is flagged broader
        result = label_alignment(["car", "sedan"], target_ontology())
        assert relations_of(result, "car") == {"car": "equivalent", "suv": "broader"}

    def test_unaligned_sets_are_open_world(self):
        result = label_alignment(["car", "spaceship"], target_ontology())
        assert result["unaligned_source"] == ("spaceship",)
        assert "vehicle" in result["unaligned_target"]
        assert "dog" in result["unaligned_target"]
        assert "car" not in result["unaligned_target"]  # referenced

    def test_mergeability_lossless(self):
        result = label_alignment(["car", "dog"], target_ontology())
        assert result["mergeability"] == "lossless"

    def test_mergeability_lossy_on_collapse(self):
        # two distinct source concepts coarsen to the same target -> specificity lost
        source = Ontology.from_hierarchy({"car": {"a": None, "b": None}})
        result = label_alignment(source, target_ontology())
        assert dict(result["class_remap"]) == {"car": "car", "a": "car", "b": "car"}
        assert result["unaligned_source"] == ()
        assert result["mergeability"] == "lossy"

    def test_mergeability_partial_on_unaligned(self):
        result = label_alignment(["car", "spaceship"], target_ontology())
        assert result["mergeability"] == "partial"

    def test_bare_list_source_matches_reconciliation(self):
        names = ["sedan", "truck", "spaceship"]
        target = target_ontology()
        result = label_alignment(names, target)
        rec = label_reconciliation(names, target)
        # On a structureless source, the equivalence class_remap is exactly the matched set
        assert dict(result["class_remap"]) == dict(rec["matched"])
        assert set(result["unaligned_source"]) == set(rec["unmatched"])

    def test_single_string_source_rejected(self):
        with pytest.raises(TypeError, match="sequence of class names"):
            label_alignment("car", target_ontology())  # type: ignore[arg-type]

    def test_ambiguous_source_left_unanchored(self):
        # boat and plane share the synonym "Craft" -> ambiguous, no correspondence
        target = Ontology([
            OntologyConcept(id="boat", label="Boat", synonyms=("Craft",)),
            OntologyConcept(id="plane", label="Plane", synonyms=("Craft",)),
        ])
        result = label_alignment(["Craft"], target)
        assert result["unaligned_source"] == ("Craft",)
        assert all(c.source != "Craft" for c in result["correspondences"])


@pytest.mark.required
class TestMatcherIntegration:
    def test_stub_satisfies_matcher_protocol(self):
        assert isinstance(StubMatcher(), Matcher)

    def test_proposal_accepted_above_threshold(self):
        result = label_alignment(["x"], target_ontology(), matchers=[StubMatcher(("x", "car", 0.8))], threshold=0.5)
        match = next(c for c in result["correspondences"] if c.source == "x")
        assert (match.target, match.relation, match.matcher) == ("car", "equivalent", "stub")
        assert result["class_remap"]["x"] == "car"

    def test_proposal_below_threshold_abstains(self):
        result = label_alignment(["x"], target_ontology(), matchers=[StubMatcher(("x", "car", 0.4))], threshold=0.9)
        assert result["unaligned_source"] == ("x",)

    def test_tie_abstains(self):
        # two distinct targets at the same best score -> ambiguous -> withheld
        result = label_alignment(
            ["x"], target_ontology(), matchers=[StubMatcher(("x", "car", 0.8), ("x", "truck", 0.8))]
        )
        assert result["unaligned_source"] == ("x",)

    def test_best_proposal_wins(self):
        result = label_alignment(
            ["x"], target_ontology(), matchers=[StubMatcher(("x", "car", 0.6), ("x", "truck", 0.9))]
        )
        assert result["class_remap"]["x"] == "truck"

    def test_matcher_not_consulted_for_exact_anchored(self):
        # "car" exact-anchors to car; a conflicting matcher proposal must be ignored
        result = label_alignment(["car"], target_ontology(), matchers=[StubMatcher(("car", "truck", 1.0))])
        assert result["class_remap"]["car"] == "car"
