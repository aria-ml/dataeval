import numpy as np
import pytest

from dataeval import Ontology
from dataeval.core._label_coverage import label_coverage
from dataeval.protocols import DatasetMetadata
from dataeval.scope._representation import Representation, RepresentationOutput
from dataeval.types import OntologyConcept


class ClassificationDataset:
    """A minimal image-classification dataset with one-hot targets.

    ``index2label`` is carried on ``metadata`` so class indices resolve to the
    ontology's leaf names.
    """

    def __init__(self, counts: dict[str, int], leaves: list[str]) -> None:
        targets = []
        for name, count in counts.items():
            onehot = np.zeros(len(leaves))
            onehot[leaves.index(name)] = 1
            targets += [onehot.copy() for _ in range(count)]
        self._targets = targets
        self._data = np.ones((len(targets), 3, 3))
        self.metadata = DatasetMetadata(id="d", index2label=dict(enumerate(leaves)))

    def __getitem__(self, i):
        return self._data[i], self._targets[i], {"id": i}

    def __len__(self) -> int:
        return len(self._targets)


def animal_ontology() -> Ontology:
    """animal → {mammal → [cat, dog], bird → [owl, hawk]}."""
    return Ontology.from_hierarchy({"animal": {"mammal": ["cat", "dog"], "bird": ["owl", "hawk"]}})


ANIMAL_LEAVES = ["cat", "dog", "owl", "hawk"]


@pytest.mark.required
class TestEvaluate:
    def test_worklist_ranks_by_deficit(self):
        ds = ClassificationDataset({"cat": 8, "dog": 2, "owl": 1}, ANIMAL_LEAVES)
        res = Representation(animal_ontology()).evaluate(ds)

        assert isinstance(res, RepresentationOutput)
        assert res.columns == ["concept", "label", "parent", "action", "count", "target", "deficit"]
        rows = res.data().to_dicts()
        # total 11, uniform target round(11/4) = 3; sorted by -deficit
        assert [r["concept"] for r in rows] == ["hawk", "owl", "dog"]
        assert rows[0] == {
            "concept": "hawk",
            "label": "hawk",
            "parent": "bird",
            "action": "acquire",  # count == 0
            "count": 0,
            "target": 3,
            "deficit": 3,
        }
        assert rows[1]["action"] == "augment"  # owl has count 1
        assert res.total_deficit == 6
        assert res.leaf_coverage == 0.75

    def test_dark_branch_reported(self):
        # nothing under 'bird' is labeled, so the whole branch is dark
        ds = ClassificationDataset({"cat": 5, "dog": 5}, ANIMAL_LEAVES)
        res = Representation(animal_ontology()).evaluate(ds)

        dark = res.dark_branches.to_dicts()
        assert dark == [{"concept": "bird", "label": "bird", "leaves": 2}]

    def test_expected_floor_sets_target_and_flags_violation(self):
        # owl is asserted to need a 50% share but is badly under-represented
        ds = ClassificationDataset({"cat": 8, "dog": 2, "owl": 1}, ANIMAL_LEAVES)
        res = Representation(animal_ontology(), expected={"owl": 0.5}).evaluate(ds)

        viol = res.violations.to_dicts()
        assert len(viol) == 1
        assert viol[0]["concept"] == "owl"
        assert viol[0]["floor"] == 0.5
        assert viol[0]["actual"] == pytest.approx(1 / 11)
        assert viol[0]["shortfall"] == 5  # round(0.5 * 11) - 1

        # the asserted floor also raises owl's worklist target above the uniform 3
        owl_row = next(r for r in res.data().to_dicts() if r["concept"] == "owl")
        assert owl_row["target"] == 6  # round(0.5 * 11)


@pytest.mark.required
class TestDarkBranchMaximality:
    def test_only_topmost_dark_branch_reported(self):
        # bird → raptor → [owl, hawk]; nothing under bird is labeled, so both bird
        # and raptor are dark but only the maximal 'bird' is reported
        onto = Ontology.from_hierarchy({"animal": {"mammal": ["cat", "dog"], "bird": {"raptor": ["owl", "hawk"]}}})
        cov = label_coverage({"cat": 5, "dog": 5}, onto)
        assert Representation(onto)._dark_branches(cov) == [{"concept": "bird", "label": "bird", "leaves": 2}]


@pytest.mark.required
class TestExpectedResolution:
    def test_unresolvable_expected_name_is_ignored(self):
        # a name matching no concept resolves to 0 ids and is dropped with a warning
        r = Representation(animal_ontology(), expected={"owl": 0.5, "unicorn": 0.3})
        assert r._expected_by_concept() == {"owl": 0.5}

    def test_no_expected_yields_empty_floors(self):
        assert Representation(animal_ontology())._expected_by_concept() == {}


@pytest.mark.required
class TestInternalEdges:
    def test_external_parent_shown_as_id(self):
        # 'truck' hangs off an undefined external parent; the id is used as its label
        onto = Ontology([OntologyConcept(id="truck", label="truck", parents=("ext:heavy",))])
        rep = Representation(onto)
        assert rep._label("ext:heavy") == "ext:heavy"
        rows = rep._worklist({"truck": 0}, {}, total=10)
        assert rows[0]["parent"] == "ext:heavy"
        assert rows[0]["action"] == "acquire"

    def test_empty_ontology_has_no_worklist(self):
        # no leaves → uniform share is 0.0 and the worklist is empty
        assert Representation(Ontology([]))._worklist({}, {}, total=0) == []

    def test_satisfied_floor_is_not_a_violation(self):
        # owl's observed share (1/11) comfortably clears a 5% floor, so no row is emitted
        onto = animal_ontology()
        direct = label_coverage({"cat": 8, "dog": 2, "owl": 1}, onto)["direct_count"]
        assert Representation(onto)._violations(direct, {"owl": 0.05}, total=11) == []

    def test_violations_with_zero_total(self):
        # a zero-size dataset makes every observed share 0.0, so any floor is violated
        onto = animal_ontology()
        direct = label_coverage({}, onto)["direct_count"]
        rows = Representation(onto)._violations(direct, {"owl": 0.5}, total=0)
        assert len(rows) == 1
        assert rows[0]["actual"] == 0.0
        assert rows[0]["concept"] == "owl"
