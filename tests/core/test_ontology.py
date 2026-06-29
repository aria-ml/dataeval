import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from dataeval import Ontology
from dataeval.core import label_reconciliation
from dataeval.exceptions import OntologyCycleError, OntologyError
from dataeval.types import OntologyConcept


def build_ontology() -> Ontology:
    """A small DAG with synonyms, a multi-parent node, and a dangling parent.

    vehicle
    ├── land_vehicle ("Ground Vehicle")
    │   ├── car ("Automobile")
    │   └── amphibious  ── also child of water_vehicle (DAG)
    └── water_vehicle
        └── amphibious
    truck ── parent "ext:heavy" (not loaded; external boundary)
    boat / plane ── share synonym "Craft" (ambiguous)
    """
    return Ontology([
        OntologyConcept(id="vehicle", label="Vehicle"),
        OntologyConcept(id="land_vehicle", label="Land Vehicle", synonyms=("Ground Vehicle",), parents=("vehicle",)),
        OntologyConcept(id="water_vehicle", label="Water Vehicle", parents=("vehicle",)),
        OntologyConcept(id="car", label="Car", synonyms=("Automobile",), parents=("land_vehicle",)),
        OntologyConcept(id="amphibious", label="Amphibious Vehicle", parents=("land_vehicle", "water_vehicle")),
        OntologyConcept(id="truck", label="Truck", parents=("ext:heavy",)),
        OntologyConcept(id="boat", label="Boat", synonyms=("Craft",), parents=("water_vehicle",)),
        OntologyConcept(id="plane", label="Plane", synonyms=("Craft",), parents=("vehicle",)),
    ])


@pytest.mark.required
class TestOntologyModel:
    def test_len_contains_getitem(self):
        onto = build_ontology()
        assert len(onto) == 8
        assert "car" in onto
        assert "ext:heavy" not in onto
        assert onto["car"].label == "Car"
        assert onto.concept("car") == onto["car"]

    def test_roots(self):
        # only 'vehicle' has no parents at all; truck has an (external) parent
        assert set(build_ontology().roots) == {"vehicle"}

    def test_find_by_label_synonym_and_case(self):
        onto = build_ontology()
        assert onto.find("Car") == ("car",)
        assert onto.find("automobile") == ("car",)  # synonym, case-insensitive
        assert onto.find("GROUND VEHICLE") == ("land_vehicle",)
        assert onto.find("nope") == ()

    def test_find_by_exact_id(self):
        assert build_ontology().find("car") == ("car",)

    def test_find_ambiguous(self):
        assert set(build_ontology().find("Craft")) == {"boat", "plane"}

    def test_ancestors_breadth_first(self):
        onto = build_ontology()
        assert onto.ancestors("car") == ("land_vehicle", "vehicle")
        # multi-parent node visits both parents before the shared grandparent
        assert onto.ancestors("amphibious") == ("land_vehicle", "water_vehicle", "vehicle")

    def test_ancestors_unknown_raises(self):
        with pytest.raises(KeyError):
            build_ontology().ancestors("ext:heavy")

    def test_descendants(self):
        onto = build_ontology()
        assert set(onto.descendants("vehicle")) == {
            "land_vehicle",
            "water_vehicle",
            "car",
            "amphibious",
            "boat",
            "plane",
        }
        assert onto.descendants("car") == ()

    def test_is_a(self):
        onto = build_ontology()
        assert onto.is_a("car", "vehicle")
        assert onto.is_a("amphibious", "water_vehicle")
        assert not onto.is_a("vehicle", "car")
        assert not onto.is_a("car", "water_vehicle")

    def test_dangling_parent_is_external_boundary(self):
        # references a parent not present in the ontology: kept, not an error
        onto = build_ontology()
        assert onto.ancestors("truck") == ("ext:heavy",)
        assert "ext:heavy" not in onto

    def test_external_ids(self):
        assert build_ontology().external_ids == ("ext:heavy",)

    def test_label_collisions(self):
        # boat and plane share the synonym "Craft"; nothing else collides
        assert build_ontology().label_collisions == {"craft": ("boat", "plane")}

    def test_label_collisions_dedupes_self_reuse(self):
        # a concept whose synonym casefold-equals its own label is not a collision
        onto = Ontology([OntologyConcept(id="car", label="car", synonyms=("Car",))])
        assert onto.label_collisions == {}

    def test_lowest_common_ancestor(self):
        onto = build_ontology()
        assert onto.lowest_common_ancestor("car", "amphibious") == "land_vehicle"
        assert onto.lowest_common_ancestor("car", "boat") == "vehicle"
        # a concept and its descendant: the ancestor itself
        assert onto.lowest_common_ancestor("car", "vehicle") == "vehicle"

    def test_lca_none_when_disjoint(self):
        onto = Ontology([OntologyConcept(id="a", label="A"), OntologyConcept(id="b", label="B")])
        assert onto.lowest_common_ancestor("a", "b") is None
        assert onto.lowest_common_ancestors("a", "b") == ()

    def test_lowest_common_ancestors_dag_returns_all(self):
        # x and y each inherit from both p1 and p2: two incomparable LCAs
        onto = Ontology.from_hierarchy({"p1": ["x", "y"], "p2": ["x", "y"]})
        assert onto.lowest_common_ancestors("x", "y") == ("p1", "p2")
        # the singular projection collapses deterministically to one of them
        assert onto.lowest_common_ancestor("x", "y") in {"p1", "p2"}

    def test_lca_picks_deepest_among_incomparable(self):
        # both p (deep) and q (shallow) are common ancestors of x and y, but
        # only the lower pair are LCAs; q sits above p so it must be excluded
        onto = Ontology.from_hierarchy({"q": {"p": ["x", "y"]}})
        assert onto.lowest_common_ancestors("x", "y") == ("p",)
        assert onto.lowest_common_ancestor("x", "y") == "p"

    def test_lca_can_be_external_reference(self):
        # the only shared ancestor is an undefined (external) parent
        onto = Ontology([
            OntologyConcept(id="a", label="A", parents=("ext:top",)),
            OntologyConcept(id="b", label="B", parents=("ext:top",)),
        ])
        assert onto.lowest_common_ancestors("a", "b") == ("ext:top",)
        assert onto.lowest_common_ancestor("a", "b") == "ext:top"

    def test_lca_unknown_concept_raises(self):
        with pytest.raises(KeyError):
            build_ontology().lowest_common_ancestor("nope", "car")
        with pytest.raises(KeyError):
            build_ontology().lowest_common_ancestors("car", "ext:heavy")

    def test_leaves(self):
        # everything that is never a parent: car, amphibious, truck, boat, plane
        assert set(build_ontology().leaves) == {"car", "amphibious", "truck", "boat", "plane"}

    def test_siblings(self):
        onto = build_ontology()
        # car and amphibious both have land_vehicle as a parent
        assert set(onto.siblings("car")) == {"amphibious"}
        # land_vehicle and water_vehicle share parent vehicle (with plane)
        assert set(onto.siblings("land_vehicle")) == {"water_vehicle", "plane"}

    def test_siblings_unknown_raises(self):
        with pytest.raises(KeyError):
            build_ontology().siblings("ext:heavy")

    def test_depth_of(self):
        onto = build_ontology()
        assert onto.depth_of("vehicle") == 0
        assert onto.depth_of("land_vehicle") == 1
        assert onto.depth_of("car") == 2
        # amphibious: max over land_vehicle(1)/water_vehicle(1) + 1 = 2
        assert onto.depth_of("amphibious") == 2
        # only parent is external (ext:heavy) -> depth 1
        assert onto.depth_of("truck") == 1

    def test_subtree_ids(self):
        onto = build_ontology()
        assert onto.subtree_ids("land_vehicle") == frozenset({"land_vehicle", "car", "amphibious"})
        assert onto.subtree_ids("car") == frozenset({"car"})  # leaf: just itself
        with pytest.raises(KeyError):
            onto.subtree_ids("ext:heavy")

    def test_subtree(self):
        sub = build_ontology().subtree("land_vehicle")
        assert set(sub.ids) == {"land_vehicle", "car", "amphibious"}
        # land_vehicle's parent (vehicle) is pruned -> it becomes a root
        assert sub.roots == ("land_vehicle",)
        # amphibious keeps only the in-subtree parent (water_vehicle pruned)
        assert sub.concept("amphibious").parents == ("land_vehicle",)

    def test_duplicate_id_raises(self):
        with pytest.raises(OntologyError, match="Duplicate concept id"):
            Ontology([OntologyConcept(id="x", label="X"), OntologyConcept(id="x", label="X2")])

    def test_cycle_raises(self):
        with pytest.raises(OntologyCycleError, match="cycle"):
            Ontology([
                OntologyConcept(id="a", label="A", parents=("b",)),
                OntologyConcept(id="b", label="B", parents=("a",)),
            ])

    def test_typed_exceptions_subclass_valueerror(self):
        # back-compat: existing `except ValueError` still catches these
        assert issubclass(OntologyError, ValueError)
        assert issubclass(OntologyCycleError, OntologyError)

    def test_concept_is_frozen(self):
        concept = OntologyConcept(id="a", label="A")
        with pytest.raises(ValidationError):
            concept.label = "B"


@pytest.mark.required
class TestValidateLabels:
    def test_match_report(self):
        res = label_reconciliation(["Car", "Automobile", "Craft", "Nonexistent"], build_ontology())
        assert res["matched"] == {"Car": "car", "Automobile": "car"}
        assert res["unmatched"] == ["Nonexistent"]
        assert set(res["ambiguous"]["Craft"]) == {"boat", "plane"}

    def test_ancestor_paths(self):
        res = label_reconciliation(["Car", "Amphibious Vehicle"], build_ontology())
        assert res["ancestor_paths"]["Car"] == ["land_vehicle", "vehicle"]
        assert res["ancestor_paths"]["Amphibious Vehicle"] == ["land_vehicle", "water_vehicle", "vehicle"]

    def test_induced_edges_transitive_reduction(self):
        # vehicle is matched too: car/amphibious should attach to their nearest
        # matched ancestor (land_vehicle), not to the further vehicle
        res = label_reconciliation(["Vehicle", "Land Vehicle", "Car", "Amphibious Vehicle"], build_ontology())
        edges = set(res["induced_edges"])
        assert ("Land Vehicle", "Car") in edges
        assert ("Land Vehicle", "Amphibious Vehicle") in edges
        assert ("Vehicle", "Land Vehicle") in edges
        assert ("Vehicle", "Car") not in edges  # collapsed via land_vehicle

    def test_relations(self):
        res = label_reconciliation(["Car", "Land Vehicle", "Amphibious Vehicle"], build_ontology())
        rel = res["relations"]
        assert rel[("Car", "Land Vehicle")] == "descendant"
        assert rel[("Land Vehicle", "Car")] == "ancestor"
        assert rel[("Car", "Amphibious Vehicle")] == "sibling"

    def test_unrelated_relation(self):
        onto = Ontology([
            OntologyConcept(id="a", label="A"),
            OntologyConcept(id="b", label="B"),
        ])
        res = label_reconciliation(["A", "B"], onto)
        assert res["relations"][("A", "B")] == "unrelated"

    def test_ambiguous_excluded_from_hierarchy(self):
        res = label_reconciliation(["Craft"], build_ontology())
        assert res["ancestor_paths"] == {}
        assert res["induced_edges"] == []

    def test_external_ancestors_flags_truncated_hierarchy(self):
        # 'Truck' resolves but its only ancestor is the undefined 'ext:heavy'
        res = label_reconciliation(["Truck", "Car"], build_ontology())
        assert res["external_ancestors"] == {"Truck": ["ext:heavy"]}
        # 'Car' is fully rooted, so it is absent from external_ancestors
        assert "Car" not in res["external_ancestors"]


@pytest.mark.required
class TestFromHierarchy:
    """Dependency-free construction from plain Python hierarchies (no rdflib)."""

    def test_flat_list(self):
        onto = Ontology.from_hierarchy(["car", "dog", "bird"])
        assert set(onto.ids) == {"car", "dog", "bird"}
        assert set(onto.roots) == {"car", "dog", "bird"}

    def test_one_level_mapping(self):
        onto = Ontology.from_hierarchy({"car": ["sedan", "SUV"], "dog": None})
        assert onto.concept("sedan").parents == ("car",)
        assert onto.concept("dog").parents == ()
        assert set(onto.descendants("car")) == {"sedan", "SUV"}

    def test_nested_mapping(self):
        onto = Ontology.from_hierarchy({"vehicle": {"car": {"sedan": None}}})
        assert [onto.concept(a).label for a in onto.ancestors("sedan")] == ["car", "vehicle"]
        # labels double as ids and labels
        assert onto.concept("sedan").id == "sedan"

    def test_shared_child_becomes_dag(self):
        onto = Ontology.from_hierarchy({"land": ["amphibious"], "water": ["amphibious"]})
        assert set(onto.concept("amphibious").parents) == {"land", "water"}

    def test_cycle_raises(self):
        with pytest.raises(OntologyCycleError):
            Ontology.from_hierarchy({"a": {"b": {"a": None}}})

    def test_non_string_label_raises(self):
        with pytest.raises(OntologyError, match="Unexpected hierarchy node"):
            Ontology.from_hierarchy({"car": [123]})


TURTLE = """
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix ex: <http://example.org/> .

ex:Animal a owl:Class ; rdfs:label "Animal" .
ex:Dog a owl:Class ;
    skos:prefLabel "Dog" ;
    skos:altLabel "Canine" ;
    rdfs:subClassOf ex:Animal ;
    skos:definition "A domestic dog." .
"""

# Mirrors the JHU/APL example: @context shortcuts, language-tagged labels, and
# altLabel as both a scalar (Dog) and a list (Cat).
JSONLD = """
{
  "@context": {
    "subClassOf": {"@id": "http://www.w3.org/2000/01/rdf-schema#subClassOf", "@type": "@id"},
    "label": {"@id": "http://www.w3.org/2000/01/rdf-schema#label"},
    "prefLabel": {"@id": "http://www.w3.org/2004/02/skos/core#prefLabel"},
    "altLabel": {"@id": "http://www.w3.org/2004/02/skos/core#altLabel"},
    "owl": "http://www.w3.org/2002/07/owl#"
  },
  "@graph": [
    {"@id": "ex:Animal", "@type": "owl:Class", "label": {"@language": "en", "@value": "Animal"}},
    {"@id": "ex:Dog", "@type": "owl:Class", "subClassOf": "ex:Animal",
     "prefLabel": {"@language": "en", "@value": "Dog"},
     "altLabel": {"@language": "en", "@value": "Canine"}},
    {"@id": "ex:Cat", "@type": "owl:Class", "subClassOf": "ex:Animal",
     "label": {"@language": "en", "@value": "Cat"},
     "altLabel": [
        {"@language": "en", "@value": "Feline"},
        {"@language": "en", "@value": "Kitty"}
     ]}
  ]
}
"""


@pytest.mark.required
class TestOptionalDependency:
    """Graceful behavior when the optional 'rdflib' dependency is absent.

    These run in the base (rdflib-free) suite: building and querying an Ontology
    in memory must never require rdflib, and the RDF constructors must fail with
    an actionable message rather than a bare ImportError.
    """

    def test_in_memory_build_needs_no_rdflib(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "rdflib", None)  # make `import rdflib` fail
        onto = Ontology([OntologyConcept(id="a", label="A"), OntologyConcept(id="b", label="B", parents=("a",))])
        assert onto.ancestors("b") == ("a",)

    def test_from_rdf_without_rdflib_raises_helpful_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "rdflib", None)  # make `import rdflib` fail
        with pytest.raises(ImportError, match=r"dataeval\[ontology\]"):
            Ontology.from_rdf(TURTLE, format="turtle")


@pytest.mark.optional
class TestRdfAdapters:
    def test_from_turtle(self):
        onto = Ontology.from_rdf(TURTLE, format="turtle")
        dog = onto.concept("http://example.org/Dog")
        assert dog.label == "Dog"
        assert "Canine" in dog.synonyms
        assert dog.definition == "A domestic dog."
        assert dog.parents == ("http://example.org/Animal",)
        assert onto.find("Canine") == ("http://example.org/Dog",)

    def test_from_jsonld_scalar_and_list_altlabel(self):
        onto = Ontology.from_rdf(JSONLD, format="json-ld")
        assert len(onto) == 3
        cat = onto.concept("ex:Cat")
        assert cat.label == "Cat"
        assert set(cat.synonyms) == {"Feline", "Kitty"}  # list form
        dog = onto.concept("ex:Dog")
        assert dog.synonyms == ("Canine",)  # scalar form
        assert onto.is_a("ex:Dog", "ex:Animal")

    def test_from_rdflib_graph(self):
        rdflib = pytest.importorskip("rdflib")
        graph = rdflib.Graph()
        graph.parse(data=TURTLE, format="turtle")
        onto = Ontology.from_rdflib(graph)
        assert onto.is_a("http://example.org/Dog", "http://example.org/Animal")


CV = "http://example.org/cv-ontology#"
VEHICLE_ONTOLOGY = Path(__file__).parent / "vehicle_ontology.jsonld"


@pytest.mark.optional
class TestVehicleOntologyFixture:
    """End-to-end against the committed public sample ontology (a CUI-free stand-in
    for a real-world OWL/JSON-LD ontology, including an intentional dangling parent)."""

    @pytest.fixture
    def onto(self):
        return Ontology.from_rdf(VEHICLE_ONTOLOGY.read_bytes(), format="json-ld")

    def test_loads_full_hierarchy(self, onto):
        assert len(onto) == 22
        assert {onto.concept(r).label for r in onto.roots} == {"Aircraft", "Land Vehicle", "Water Vessel"}
        assert [onto.concept(a).label for a in onto.ancestors(f"{CV}ToyotaCorolla")] == ["Sedan", "Land Vehicle"]
        assert [onto.concept(a).label for a in onto.ancestors(f"{CV}Boeing737")] == ["Commercial Airliner", "Aircraft"]

    def test_synonym_matching(self, onto):
        assert onto.find("B737") == (f"{CV}Boeing737",)
        assert onto.find("Predator B") == (f"{CV}MQ9Reaper",)

    def test_dangling_parent_surfaced_as_external(self, onto):
        # Submarine's parent 'UnderseaVessel' is intentionally undefined in the file
        assert onto.external_ids == (f"{CV}UnderseaVessel",)
        res = label_reconciliation(["Submarine", "Toyota Corolla"], onto)
        assert res["matched"]["Submarine"] == f"{CV}Submarine"
        assert res["external_ancestors"] == {"Submarine": [f"{CV}UnderseaVessel"]}
        assert "Toyota Corolla" not in res["external_ancestors"]
