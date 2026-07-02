import pytest

from dataeval import Ontology
from dataeval.core._label_coverage import label_coverage
from dataeval.types import OntologyConcept


def animal_ontology() -> Ontology:
    """A small taxonomy with two internal concepts and four leaf species.

    animal
    ├── mammal
    │   ├── cat
    │   └── dog
    └── bird
        ├── owl
        └── hawk
    """
    return Ontology.from_hierarchy({"animal": {"mammal": ["cat", "dog"], "bird": ["owl", "hawk"]}})


@pytest.mark.required
class TestBasicCoverage:
    def test_resolution_and_direct_counts(self):
        res = label_coverage({"cat": 8, "dog": 2, "owl": 1}, animal_ontology())
        assert res["matched"] == {"cat": "cat", "dog": "dog", "owl": "owl"}
        assert res["unmatched"] == {}
        assert res["ambiguous"] == {}
        # direct mass lands exactly on the matched leaf; unlabeled concepts are 0
        assert res["direct_count"]["cat"] == 8
        assert res["direct_count"]["hawk"] == 0
        assert res["direct_count"]["animal"] == 0

    def test_subtree_counts_roll_up(self):
        res = label_coverage({"cat": 8, "dog": 2, "owl": 1}, animal_ontology())
        assert res["subtree_count"]["mammal"] == 10
        assert res["subtree_count"]["bird"] == 1
        assert res["subtree_count"]["animal"] == 11

    def test_covered_leaves(self):
        res = label_coverage({"cat": 8, "dog": 2, "owl": 1}, animal_ontology())
        assert res["covered_leaves"]["mammal"] == (2, 2)
        assert res["covered_leaves"]["bird"] == (1, 2)
        assert res["covered_leaves"]["animal"] == (3, 4)
        # a leaf's own subtree is just itself
        assert res["covered_leaves"]["cat"] == (1, 1)
        assert res["covered_leaves"]["hawk"] == (0, 1)

    def test_covered_children(self):
        res = label_coverage({"cat": 8, "dog": 2, "owl": 1}, animal_ontology())
        assert res["covered_children"]["animal"] == (2, 2)
        assert res["covered_children"]["bird"] == (1, 2)
        # leaves report (0, 0)
        assert res["covered_children"]["cat"] == (0, 0)

    def test_coverage_by_depth(self):
        res = label_coverage({"cat": 8, "dog": 2, "owl": 1}, animal_ontology())
        assert res["coverage_by_depth"] == {0: (1, 1), 1: (2, 2), 2: (3, 4)}

    def test_leaf_coverage_and_distribution(self):
        res = label_coverage({"cat": 8, "dog": 2, "owl": 1}, animal_ontology())
        assert res["leaf_coverage"] == 0.75
        assert res["leaf_distribution"]["cat"] == pytest.approx(8 / 11)
        assert res["leaf_distribution"]["owl"] == pytest.approx(1 / 11)
        assert res["leaf_distribution"]["hawk"] == 0.0


@pytest.mark.required
class TestResolutionEdges:
    def test_unmatched_mass_reported(self):
        res = label_coverage({"cat": 5, "unicorn": 3}, animal_ontology())
        assert res["unmatched"] == {"unicorn": 3}
        # unmatched mass is not attributed to any concept
        assert res["subtree_count"]["animal"] == 5

    def test_ambiguous_mass_excluded_from_tallies(self):
        # 'cat' and 'car' both carry the synonym 'jaguar', so 'jaguar' is ambiguous
        onto = Ontology([
            OntologyConcept(id="animal", label="animal"),
            OntologyConcept(id="cat_animal", label="cat", parents=("animal",), synonyms=("jaguar",)),
            OntologyConcept(id="car_model", label="car", parents=("animal",), synonyms=("jaguar",)),
        ])
        res = label_coverage({"jaguar": 5, "cat": 2}, onto)
        assert res["ambiguous"] == {"jaguar": ["cat_animal", "car_model"]}
        assert res["matched"] == {"cat": "cat_animal"}
        # only the unambiguous 'cat' mass is attributed; 'jaguar' mass is dropped
        assert res["direct_count"]["cat_animal"] == 2
        assert res["subtree_count"]["animal"] == 2


@pytest.mark.required
class TestGraphShapes:
    def test_multi_parent_mass_rolls_up_once_per_ancestor(self):
        # 'amphibious' has two parents; its mass reaches each distinct ancestor once
        onto = Ontology([
            OntologyConcept(id="vehicle", label="vehicle"),
            OntologyConcept(id="car", label="car", parents=("vehicle",)),
            OntologyConcept(id="boat", label="boat", parents=("vehicle",)),
            OntologyConcept(id="amphibious", label="amphibious", parents=("car", "boat")),
        ])
        res = label_coverage({"amphibious": 5}, onto)
        assert res["subtree_count"]["car"] == 5
        assert res["subtree_count"]["boat"] == 5
        assert res["subtree_count"]["vehicle"] == 5
        assert res["direct_count"]["amphibious"] == 5

    def test_external_ancestor_not_counted(self):
        # 'ext:heavy' is an undefined external parent, so mass does not propagate to it
        onto = Ontology([OntologyConcept(id="truck", label="truck", parents=("ext:heavy",))])
        res = label_coverage({"truck": 3}, onto)
        assert res["subtree_count"] == {"truck": 3}
        assert "ext:heavy" not in res["subtree_count"]


@pytest.mark.required
class TestNoLabeledLeaves:
    def test_mass_only_on_internal_concept(self):
        # all mass lands on an internal concept; no leaf is populated
        res = label_coverage({"animal": 4}, animal_ontology())
        assert res["direct_count"]["animal"] == 4
        assert res["leaf_coverage"] == 0.0
        assert res["leaf_distribution"] == {"cat": 0.0, "dog": 0.0, "owl": 0.0, "hawk": 0.0}

    def test_empty_ontology_has_no_leaves(self):
        # an empty ontology is the only genuinely leaf-free graph: the leaf_coverage
        # scalar falls back to 0.0 rather than dividing by zero
        res = label_coverage({}, Ontology([]))
        assert res["leaf_coverage"] == 0.0
        assert res["leaf_distribution"] == {}
        assert res["coverage_by_depth"] == {}
        assert res["direct_count"] == {}
