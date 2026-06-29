import pytest

from dataeval import Ontology
from dataeval.core import ontology_validation
from dataeval.types import OntologyConcept

SNAKE_CASE = r"[a-z0-9]+(_[a-z0-9]+)*"


def clean_ontology() -> Ontology:
    """A well-formed, snake_case taxonomy with no structural or naming smells.

    vehicle
    ├── car
    │   ├── sedan
    │   └── coupe
    └── truck
    """
    return Ontology.from_hierarchy({"vehicle": {"car": ["sedan", "coupe"], "truck": None}})


@pytest.mark.required
class TestCleanOntology:
    def test_no_findings(self):
        res = ontology_validation(clean_ontology(), label_pattern=SNAKE_CASE)
        assert res["isolated"] == []
        assert res["external_ancestors"] == {}
        assert res["redundant_edges"] == []
        assert res["ancestor_siblings"] == []
        assert res["unary_parents"] == []
        assert res["label_collisions"] == {}
        assert res["nonconforming_labels"] == {}

    def test_roots_and_leaves(self):
        res = ontology_validation(clean_ontology())
        assert set(res["roots"]) == {"vehicle"}
        assert set(res["leaves"]) == {"sedan", "coupe", "truck"}

    def test_shape_profile(self):
        res = ontology_validation(clean_ontology())
        assert res["depth"] == {"vehicle": 0, "car": 1, "sedan": 2, "coupe": 2, "truck": 1}
        assert res["fan_out"] == {"vehicle": 2, "car": 2, "sedan": 0, "coupe": 0, "truck": 0}
        assert res["parent_count"] == {"vehicle": 0, "car": 1, "sedan": 1, "coupe": 1, "truck": 1}


@pytest.mark.required
class TestConnectivity:
    def test_isolated_concept(self):
        # 'orphan' has neither parents nor children
        onto = Ontology.from_hierarchy({"vehicle": ["car"], "orphan": None})
        res = ontology_validation(onto)
        assert res["isolated"] == ["orphan"]
        # a populated root (vehicle) is not isolated despite having no parents
        assert "vehicle" not in res["isolated"]

    def test_external_ancestors(self):
        # 'truck' hangs off an undefined external parent; the truncation propagates
        # to its descendant 'pickup'
        onto = Ontology([
            OntologyConcept(id="truck", label="truck", parents=("ext:heavy",)),
            OntologyConcept(id="pickup", label="pickup", parents=("truck",)),
            OntologyConcept(id="car", label="car"),
        ])
        res = ontology_validation(onto)
        assert res["external_ancestors"] == {"truck": ["ext:heavy"], "pickup": ["ext:heavy"]}
        assert "car" not in res["external_ancestors"]


@pytest.mark.required
class TestRedundancyAndContradiction:
    def test_redundant_edge(self):
        # sedan is declared a direct child of both car and vehicle, but car is
        # already a child of vehicle -> the sedan->vehicle edge is redundant
        onto = Ontology([
            OntologyConcept(id="vehicle", label="vehicle"),
            OntologyConcept(id="car", label="car", parents=("vehicle",)),
            OntologyConcept(id="sedan", label="sedan", parents=("car", "vehicle")),
        ])
        res = ontology_validation(onto)
        assert res["redundant_edges"] == [("vehicle", "sedan")]

    def test_ancestor_sibling(self):
        # 'car' is placed under 'root' alongside 'vehicle', yet vehicle is car's
        # ancestor -> car shares a parent with its own ancestor (and the direct
        # root->car edge is redundant)
        onto = Ontology([
            OntologyConcept(id="root", label="root"),
            OntologyConcept(id="vehicle", label="vehicle", parents=("root",)),
            OntologyConcept(id="car", label="car", parents=("root", "vehicle")),
        ])
        res = ontology_validation(onto)
        assert res["ancestor_siblings"] == [("car", "vehicle")]
        assert res["redundant_edges"] == [("root", "car")]

    def test_unary_parents(self):
        # a -> b -> c is a unary chain: a and b each have exactly one child
        onto = Ontology.from_hierarchy({"a": {"b": {"c": None}}})
        res = ontology_validation(onto)
        assert set(res["unary_parents"]) == {"a", "b"}


@pytest.mark.required
class TestNaming:
    def test_label_collisions(self):
        # boat and plane share the synonym 'craft' (case-insensitively)
        onto = Ontology([
            OntologyConcept(id="water", label="water"),
            OntologyConcept(id="boat", label="boat", synonyms=("craft",), parents=("water",)),
            OntologyConcept(id="plane", label="plane", synonyms=("Craft",)),
        ])
        res = ontology_validation(onto)
        assert set(res["label_collisions"]) == {"craft"}
        assert set(res["label_collisions"]["craft"]) == {"boat", "plane"}

    def test_collision_ignores_self_reuse(self):
        # a concept whose synonym equals its own label is not a collision
        onto = Ontology([OntologyConcept(id="car", label="car", synonyms=("Car",))])
        assert ontology_validation(onto)["label_collisions"] == {}

    def test_nonconforming_labels_opt_in(self):
        onto = Ontology.from_hierarchy({"Land Vehicle": ["pickup_truck"]})
        flagged = ontology_validation(onto, label_pattern=SNAKE_CASE)["nonconforming_labels"]
        assert flagged == {"Land Vehicle": "Land Vehicle"}  # caps + space; pickup_truck is clean

    def test_nonconforming_disabled_by_default(self):
        onto = Ontology.from_hierarchy({"Land Vehicle": ["Pickup Truck"]})
        assert ontology_validation(onto)["nonconforming_labels"] == {}


@pytest.mark.required
class TestEdgeCases:
    def test_empty_ontology(self):
        onto = Ontology([])
        res = ontology_validation(onto, label_pattern=SNAKE_CASE)
        assert res["roots"] == []
        assert res["leaves"] == []
        assert res["depth"] == {}
        assert res["label_collisions"] == {}

    def test_multi_parent_is_not_redundant(self):
        # legitimate multiple inheritance (amphibious is-a land and water) must
        # not be flagged: neither parent subsumes the other
        onto = Ontology.from_hierarchy({"land": ["amphibious"], "water": ["amphibious"]})
        res = ontology_validation(onto)
        assert res["redundant_edges"] == []
        assert res["parent_count"]["amphibious"] == 2
