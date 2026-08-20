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
class TestOntologyConcept:
    def test_equivalent_to_defaults_empty(self):
        assert OntologyConcept(id="a", label="A").equivalent_to == ()

    def test_equivalent_to_round_trips(self):
        concept = OntologyConcept(id="a", label="A", equivalent_to=("b", "c"))
        assert concept.equivalent_to == ("b", "c")
        assert OntologyConcept(**concept.model_dump()) == concept


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

    def test_is_a_is_reflexive(self):
        # subsumption is reflexive under RDFS/OWL, and the container already
        # relies on that when it drops a self-referential parent as trivial.
        # Without this, every caller filtering by concept writes `a == b or ...`.
        onto = build_ontology()
        assert onto.is_a("car", "car")
        assert onto.is_a("vehicle", "vehicle")

    def test_is_a_reflexive_without_a_declared_self_parent(self):
        # the reflexive answer does not depend on the user having written the
        # trivial edge (which the container drops on build anyway)
        onto = Ontology([OntologyConcept(id="a", label="A", parents=("a",))])
        assert onto.concept("a").parents == ()
        assert onto.is_a("a", "a")

    def test_is_a_reflexivity_requires_a_defined_concept(self):
        # external ids and unknown ids are not concepts; no entailment for them
        onto = build_ontology()
        with pytest.raises(KeyError):
            onto.is_a("ext:heavy", "ext:heavy")
        with pytest.raises(KeyError):
            onto.is_a("nope", "nope")

    def test_is_a_agrees_with_lowest_common_ancestor_on_identity(self):
        # LCA already counts a concept as its own ancestor; is_a must not disagree
        onto = build_ontology()
        assert onto.lowest_common_ancestor("car", "car") == "car"
        assert onto.is_a("car", "car")

    def test_ancestors_and_descendants_stay_proper(self):
        # reflexivity belongs to the subsumption predicate only: self-inclusion
        # in these would break roots, depth, siblings, and subtree listings
        onto = build_ontology()
        assert "car" not in onto.ancestors("car")
        assert "vehicle" not in onto.descendants("vehicle")

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
        # three-node cycle: no mutual pair, so nothing licenses reading it as
        # equivalence (a two-node cycle is `owl:equivalentClass` and merges)
        with pytest.raises(OntologyCycleError, match="cycle"):
            Ontology([
                OntologyConcept(id="a", label="A", parents=("b",)),
                OntologyConcept(id="b", label="B", parents=("c",)),
                OntologyConcept(id="c", label="C", parents=("a",)),
            ])

    def test_cycle_error_names_the_cycle_not_its_descendants(self):
        # "x" and "y" are merely downstream of the a->b->c cycle; naming them
        # sends the user looking at concepts whose own edges are fine.
        with pytest.raises(OntologyCycleError) as exc_info:
            Ontology([
                OntologyConcept(id="a", label="A", parents=("b",)),
                OntologyConcept(id="b", label="B", parents=("c",)),
                OntologyConcept(id="c", label="C", parents=("a",)),
                OntologyConcept(id="x", label="X", parents=("a",)),
                OntologyConcept(id="y", label="Y", parents=("x",)),
            ])
        message = str(exc_info.value)
        assert "'a'" in message
        assert "'b'" in message
        assert "'c'" in message
        assert "'x'" not in message
        assert "'y'" not in message

    def test_self_parent_is_not_a_cycle(self):
        # X is-a X is trivially true (and materialized by RDFS/OWL reasoners);
        # it carries no information, so it is dropped rather than rejected.
        onto = Ontology([
            OntologyConcept(id="a", label="A", parents=("a",)),
            OntologyConcept(id="b", label="B", parents=("a", "b")),
        ])
        assert onto.concept("a").parents == ()
        assert onto.roots == ("a",)
        assert onto.children("a") == ("b",)  # not ("a", "b")
        assert onto.descendants("a") == ("b",)
        assert onto.ancestors("b") == ("a",)

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

    def test_node_repeated_under_itself_is_not_a_cycle(self):
        # annotation schemas often list a category among its own choices (the
        # "unspecified" option): vehicle -> [vehicle, aircraft -> [aircraft, ...]]
        onto = Ontology.from_hierarchy({"vehicle": ["vehicle", {"aircraft": ["aircraft", "helicopter"]}]})
        assert onto.roots == ("vehicle",)
        assert onto.children("vehicle") == ("aircraft",)
        assert onto.children("aircraft") == ("helicopter",)
        assert onto.ancestors("helicopter") == ("aircraft", "vehicle")

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
    @pytest.fixture(autouse=True)
    def _require_rdflib(self):
        """Every case here goes through an RDF constructor, which needs the optional dep."""
        pytest.importorskip("rdflib")

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

    def test_reflexive_subclassof_loads(self):
        # rdfs:subClassOf is reflexive under RDFS/OWL entailment, so materialized
        # ontologies commonly ship `X rdfs:subClassOf X`.
        reflexive = TURTLE + "\nex:Animal rdfs:subClassOf ex:Animal .\n"
        onto = Ontology.from_rdf(reflexive, format="turtle")
        assert onto.concept("http://example.org/Animal").parents == ()
        assert onto.children("http://example.org/Animal") == ("http://example.org/Dog",)

    def test_from_rdflib_graph(self):
        rdflib = pytest.importorskip("rdflib")
        graph = rdflib.Graph()
        graph.parse(data=TURTLE, format="turtle")
        onto = Ontology.from_rdflib(graph)
        assert onto.is_a("http://example.org/Dog", "http://example.org/Animal")

    def test_equivalent_class_merges_concepts(self):
        source = TURTLE + '\nex:Canid a owl:Class ; rdfs:label "Canid" ; owl:equivalentClass ex:Dog .\n'
        onto = Ontology.from_rdf(source, format="turtle")
        # "Canid" sorts before "Dog", so it wins the canonical election
        assert onto.canonical("http://example.org/Dog") == "http://example.org/Canid"
        assert onto.aliases("http://example.org/Canid") == ("http://example.org/Dog",)
        assert onto.find("Dog") == ("http://example.org/Canid",)
        # the absorbed concept's parent survives the merge
        assert onto.concept("http://example.org/Dog").parents == ("http://example.org/Animal",)

    def test_broader_transitive_builds_the_hierarchy(self):
        # legal SKOS; without reading it a scheme that uses only this predicate
        # loads flat, with every concept a root and no is-a edges at all.
        # Here nothing is materialized, so it is the sole source of hierarchy.
        source = """
        @prefix skos: <http://www.w3.org/2004/02/skos/core#> .
        @prefix ex: <http://example.org/> .
        ex:Animal a skos:Concept ; skos:prefLabel "Animal" .
        ex:Dog a skos:Concept ; skos:prefLabel "Dog" ; skos:broaderTransitive ex:Animal .
        """
        onto = Ontology.from_rdf(source, format="turtle")
        assert onto.concept("http://example.org/Dog").parents == ("http://example.org/Animal",)
        assert onto.is_a("http://example.org/Dog", "http://example.org/Animal")

    def test_concept_order_is_independent_of_set_iteration(self):
        # subjects are collected into a set, so without sorting the concept
        # order (and therefore Relabel's integer class indices) varied per process
        onto = Ontology.from_rdf(TURTLE, format="turtle")
        assert onto.ids == ("http://example.org/Animal", "http://example.org/Dog")

    def test_broader_transitive_is_a_fallback_not_a_direct_edge(self):
        # a materialized closure states Dog broaderTransitive Mammal AND Animal.
        # Reading both as direct parents made Animal a direct parent of Dog, so
        # siblings(Dog) reported Dog's own parent Mammal as its sibling.
        source = """
        @prefix skos: <http://www.w3.org/2004/02/skos/core#> .
        @prefix ex: <http://example.org/> .
        ex:Animal a skos:Concept ; skos:prefLabel "Animal" .
        ex:Mammal a skos:Concept ; skos:prefLabel "Mammal" ; skos:broader ex:Animal .
        ex:Dog a skos:Concept ; skos:prefLabel "Dog" ; skos:broader ex:Mammal ;
            skos:broaderTransitive ex:Mammal, ex:Animal .
        """
        onto = Ontology.from_rdf(source, format="turtle")
        assert onto.concept("http://example.org/Dog").parents == ("http://example.org/Mammal",)
        assert onto.children("http://example.org/Animal") == ("http://example.org/Mammal",)
        assert onto.siblings("http://example.org/Dog") == ()

    def test_exact_match_does_not_merge(self):
        # skos:exactMatch is a cross-scheme mapping property; intra-ontology
        # identity is owl:equivalentClass. Merging on it would conflate the two.
        source = TURTLE + "\nex:Dog skos:exactMatch <http://other.org/dog> .\n"
        onto = Ontology.from_rdf(source, format="turtle")
        assert onto.aliases("http://example.org/Dog") == ()


CV = "http://example.org/cv-ontology#"
VEHICLE_ONTOLOGY = Path(__file__).parent / "vehicle_ontology.jsonld"


@pytest.mark.optional
class TestVehicleOntologyFixture:
    """End-to-end against the committed public sample ontology (a CUI-free stand-in
    for a real-world OWL/JSON-LD ontology, including an intentional dangling parent)."""

    @pytest.fixture(autouse=True)
    def _require_rdflib(self):
        """Every case here goes through an RDF constructor, which needs the optional dep."""
        pytest.importorskip("rdflib")

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


@pytest.mark.required
class TestEquivalence:
    def test_explicit_equivalence_merges_to_lexicographic_min(self):
        onto = Ontology([
            OntologyConcept(id="ex:Car", label="Car", equivalent_to=("ex:Auto",)),
            OntologyConcept(id="ex:Auto", label="Auto"),
        ])
        assert onto.ids == ("ex:Auto",)
        assert len(onto) == 1
        assert onto.concept("ex:Auto").equivalent_to == ("ex:Car",)

    def test_mutual_subsumption_is_equivalence_not_a_cycle(self):
        # a reasoner materializing EquivalentClasses(A B) emits exactly this
        onto = Ontology([
            OntologyConcept(id="a", label="A", parents=("b",)),
            OntologyConcept(id="b", label="B", parents=("a",)),
        ])
        assert onto.ids == ("a",)
        assert onto.concept("a").parents == ()
        assert onto.roots == ("a",)

    def test_canonical_election_is_order_independent(self):
        # from_rdflib builds from a set, so input order is nondeterministic
        forward = Ontology([
            OntologyConcept(id="ex:Car", label="Car", equivalent_to=("ex:Auto",)),
            OntologyConcept(id="ex:Auto", label="Auto"),
        ])
        reverse = Ontology([
            OntologyConcept(id="ex:Auto", label="Auto"),
            OntologyConcept(id="ex:Car", label="Car", equivalent_to=("ex:Auto",)),
        ])
        assert forward.ids == reverse.ids == ("ex:Auto",)

    def test_grouping_is_transitive(self):
        onto = Ontology([
            OntologyConcept(id="a", label="A", equivalent_to=("b",)),
            OntologyConcept(id="b", label="B", equivalent_to=("c",)),
            OntologyConcept(id="c", label="C"),
        ])
        assert onto.ids == ("a",)
        assert onto.concept("a").equivalent_to == ("b", "c")

    def test_shared_undefined_equivalent_merges_both_concepts(self):
        # A = X and C = X entails A = C
        onto = Ontology([
            OntologyConcept(id="a", label="A", equivalent_to=("x",)),
            OntologyConcept(id="c", label="C", equivalent_to=("x",)),
        ])
        assert onto.ids == ("a",)
        assert onto.concept("a").equivalent_to == ("c", "x")

    def test_undefined_equivalent_becomes_alias_not_external(self):
        onto = Ontology([OntologyConcept(id="a", label="A", equivalent_to=("x",))])
        assert onto.concept("a").equivalent_to == ("x",)
        assert onto.external_ids == ()

    def test_self_equivalence_is_dropped_as_trivial(self):
        onto = Ontology([OntologyConcept(id="a", label="A", equivalent_to=("a",))])
        assert onto.ids == ("a",)
        assert onto.concept("a").equivalent_to == ()

    def test_merge_unions_labels_and_synonyms_for_find(self):
        onto = Ontology([
            OntologyConcept(id="ex:Car", label="Car", synonyms=("Motorcar",), equivalent_to=("ex:Auto",)),
            OntologyConcept(id="ex:Auto", label="Auto", synonyms=("Automobile",)),
        ])
        assert onto.find("Automobile") == ("ex:Auto",)
        assert onto.find("Motorcar") == ("ex:Auto",)
        assert onto.find("Car") == ("ex:Auto",)
        assert "Auto" not in onto.concept("ex:Auto").synonyms  # the surviving label

    def test_merge_falls_back_to_a_members_definition(self):
        onto = Ontology([
            OntologyConcept(id="a", label="A", equivalent_to=("b",)),
            OntologyConcept(id="b", label="B", definition="A road vehicle."),
        ])
        assert onto.concept("a").definition == "A road vehicle."

    def test_merge_prefers_the_canonical_members_definition(self):
        onto = Ontology([
            OntologyConcept(id="a", label="A", definition="Canonical.", equivalent_to=("b",)),
            OntologyConcept(id="b", label="B", definition="Alias."),
        ])
        assert onto.concept("a").definition == "Canonical."

    def test_merge_unions_parents_and_drops_intra_group_edges(self):
        onto = Ontology([
            OntologyConcept(id="a", label="A", parents=("top",), equivalent_to=("b",)),
            OntologyConcept(id="b", label="B", parents=("a", "side")),
        ])
        assert onto.concept("a").parents == ("top", "side")

    def test_non_member_parent_pointing_at_an_alias_is_redirected(self):
        # without this the child is lost from children(canonical) and the alias
        # degrades into a phantom external reference
        onto = Ontology([
            OntologyConcept(id="ex:Car", label="Car", equivalent_to=("ex:Auto",)),
            OntologyConcept(id="ex:Auto", label="Auto"),
            OntologyConcept(id="ex:Sedan", label="Sedan", parents=("ex:Car",)),
        ])
        assert onto.concept("ex:Sedan").parents == ("ex:Auto",)
        assert onto.children("ex:Auto") == ("ex:Sedan",)
        assert onto.external_ids == ()

    def test_unlicensed_cycle_still_raises(self):
        # no mutual pair and no equivalent_to: this is a typo, not equivalence
        with pytest.raises(OntologyCycleError) as exc_info:
            Ontology([
                OntologyConcept(id="a", label="A", parents=("b",)),
                OntologyConcept(id="b", label="B", parents=("c",)),
                OntologyConcept(id="c", label="C", parents=("a",)),
            ])
        assert "'a'" in str(exc_info.value)

    def test_from_hierarchy_cycle_still_raises(self):
        with pytest.raises(OntologyCycleError):
            Ontology.from_hierarchy({"a": {"b": {"a": None}}})

    def test_ontology_without_equivalences_is_untouched(self):
        onto = build_ontology()
        assert onto.concept("car").parents == ("land_vehicle",)
        assert onto.concept("car").equivalent_to == ()
        assert onto.external_ids == ("ext:heavy",)

    def test_canonical_resolves_aliases_and_canonicals(self):
        onto = Ontology([
            OntologyConcept(id="ex:Car", label="Car", equivalent_to=("ex:Auto",)),
            OntologyConcept(id="ex:Auto", label="Auto"),
        ])
        assert onto.canonical("ex:Car") == "ex:Auto"
        assert onto.canonical("ex:Auto") == "ex:Auto"
        with pytest.raises(KeyError):
            onto.canonical("ex:Nope")

    def test_aliases_lists_the_absorbed_ids(self):
        onto = Ontology([
            OntologyConcept(id="ex:Car", label="Car", equivalent_to=("ex:Auto",)),
            OntologyConcept(id="ex:Auto", label="Auto"),
        ])
        assert onto.aliases("ex:Auto") == ("ex:Car",)
        assert onto.aliases("ex:Car") == ("ex:Car",)  # resolves through the alias first

    def test_alias_ids_are_transparent_to_mapping_access(self):
        onto = Ontology([
            OntologyConcept(id="ex:Car", label="Car", equivalent_to=("ex:Auto",)),
            OntologyConcept(id="ex:Auto", label="Auto"),
        ])
        assert "ex:Car" in onto
        assert onto["ex:Car"] is onto["ex:Auto"]
        assert onto.concept("ex:Car").id == "ex:Auto"

    def test_iteration_yields_canonicals_only(self):
        onto = Ontology([
            OntologyConcept(id="ex:Car", label="Car", equivalent_to=("ex:Auto",)),
            OntologyConcept(id="ex:Auto", label="Auto"),
        ])
        assert [c.id for c in onto] == ["ex:Auto"]
        assert onto.ids == ("ex:Auto",)
        assert onto.roots == ("ex:Auto",)
        assert onto.leaves == ("ex:Auto",)

    def test_find_resolves_an_alias_id_to_its_canonical(self):
        onto = Ontology([
            OntologyConcept(id="ex:Car", label="Car", equivalent_to=("ex:Auto",)),
            OntologyConcept(id="ex:Auto", label="Auto"),
        ])
        assert onto.find("ex:Car") == ("ex:Auto",)

    def test_equivalent_labels_are_no_longer_a_collision(self):
        # ex:Car and ex:Auto are not ambiguous - they are the same class
        onto = Ontology([
            OntologyConcept(id="ex:Car", label="Car", equivalent_to=("ex:Auto",)),
            OntologyConcept(id="ex:Auto", label="Car"),
        ])
        assert onto.label_collisions == {}

    @staticmethod
    def merged_ontology():
        # ex:Car is absorbed into ex:Auto; ex:Sedan is a child, ex:Vehicle a parent
        return Ontology([
            OntologyConcept(id="ex:Car", label="Car", parents=("ex:Vehicle",), equivalent_to=("ex:Auto",)),
            OntologyConcept(id="ex:Auto", label="Auto"),
            OntologyConcept(id="ex:Vehicle", label="Vehicle"),
            OntologyConcept(id="ex:Sedan", label="Sedan", parents=("ex:Car",)),
        ])

    def test_is_a_is_symmetric_across_equivalents(self):
        onto = self.merged_ontology()
        assert onto.is_a("ex:Car", "ex:Auto")
        assert onto.is_a("ex:Auto", "ex:Car")

    def test_is_a_accepts_alias_on_either_side(self):
        onto = self.merged_ontology()
        assert onto.is_a("ex:Sedan", "ex:Car")  # alias as the superclass
        assert onto.is_a("ex:Car", "ex:Vehicle")  # alias as the subclass

    def test_traversals_accept_an_alias_and_emit_canonicals(self):
        onto = self.merged_ontology()
        assert onto.ancestors("ex:Car") == ("ex:Vehicle",)
        assert onto.descendants("ex:Car") == ("ex:Sedan",)
        assert onto.children("ex:Car") == ("ex:Sedan",)
        assert onto.depth_of("ex:Car") == 1
        assert onto.subtree_ids("ex:Car") == frozenset({"ex:Auto", "ex:Sedan"})

    def test_lowest_common_ancestor_accepts_an_alias(self):
        onto = self.merged_ontology()
        assert onto.lowest_common_ancestor("ex:Sedan", "ex:Car") == "ex:Auto"

    def test_siblings_accepts_an_alias(self):
        onto = Ontology([
            OntologyConcept(id="ex:Car", label="Car", parents=("ex:Vehicle",), equivalent_to=("ex:Auto",)),
            OntologyConcept(id="ex:Auto", label="Auto"),
            OntologyConcept(id="ex:Vehicle", label="Vehicle"),
            OntologyConcept(id="ex:Boat", label="Boat", parents=("ex:Vehicle",)),
        ])
        assert onto.siblings("ex:Car") == ("ex:Boat",)

    def test_subtree_concept_order_follows_the_parent_ontology(self):
        # subtree_ids() is a frozenset, so building from it directly made the
        # subtree's ids/roots/leaves/iteration vary with the hash seed
        onto = Ontology.from_hierarchy({"root": {"a": ["a1", "a2"], "b": ["b1", "b2"]}})
        assert onto.subtree("root").ids == onto.ids  # the whole tree, same order
        assert onto.subtree("a").ids == ("a", "a1", "a2")

    def test_subtree_preserves_aliases(self):
        # subtree() rebuilds an Ontology from pruned concepts; the alias map has
        # to survive that round trip or canonical() breaks inside the subtree
        onto = self.merged_ontology()
        sub = onto.subtree("ex:Car")
        assert sub.canonical("ex:Car") == "ex:Auto"
        assert sub.aliases("ex:Auto") == ("ex:Car",)
        assert sub.ids == ("ex:Auto", "ex:Sedan")

    def test_merge_entailing_further_equivalence_reaches_a_fixpoint(self):
        # A = B and A subclass-of C and C subclass-of B entails A = B = C.
        # A single grouping pass sees only the A-B edge, then canonicalizing
        # C's parent manufactures a fresh A -> C -> A cycle.
        onto = Ontology([
            OntologyConcept(id="A", label="A", equivalent_to=("B",), parents=("C",)),
            OntologyConcept(id="B", label="B"),
            OntologyConcept(id="C", label="C", parents=("B",)),
        ])
        assert onto.ids == ("A",)
        assert onto.aliases("A") == ("B", "C")
        assert onto.concept("A").parents == ()

    def test_mutual_pair_with_a_cross_edge_collapses(self):
        # a <-> b is licensed; b -> c -> a then closes over the merged group
        onto = Ontology([
            OntologyConcept(id="a", label="a", parents=("b",)),
            OntologyConcept(id="b", label="b", parents=("a", "c")),
            OntologyConcept(id="c", label="c", parents=("a",)),
        ])
        assert onto.ids == ("a",)
        assert onto.aliases("a") == ("b", "c")

    def test_fixpoint_does_not_swallow_unlicensed_cycles(self):
        # still no mutual pair anywhere, so this stays a typo
        with pytest.raises(OntologyCycleError):
            Ontology([
                OntologyConcept(id="a", label="A", parents=("b",)),
                OntologyConcept(id="b", label="B", parents=("c",)),
                OntologyConcept(id="c", label="C", parents=("a",)),
            ])

    def test_canonical_election_prefers_a_real_label_over_an_id(self):
        # from_rdflib falls back to `label = str(subject)` for an unlabelled
        # class; electing that as canonical would demote the human label to a
        # synonym and surface a raw IRI everywhere concept.label is displayed
        onto = Ontology([
            OntologyConcept(id="http://ex/A0001", label="http://ex/A0001"),
            OntologyConcept(id="http://ex/Dog", label="Dog", definition="A dog.", equivalent_to=("http://ex/A0001",)),
        ])
        assert onto.ids == ("http://ex/Dog",)
        assert onto.concept("http://ex/A0001").label == "Dog"

    def test_canonical_election_still_breaks_ties_by_smallest_id(self):
        # when every member carries a real label the rule is unchanged
        onto = Ontology([
            OntologyConcept(id="ex:Car", label="Car", equivalent_to=("ex:Auto",)),
            OntologyConcept(id="ex:Auto", label="Auto"),
        ])
        assert onto.ids == ("ex:Auto",)

    def test_canonical_election_falls_back_when_no_member_is_labelled(self):
        onto = Ontology([
            OntologyConcept(id="ex:B", label="ex:B", equivalent_to=("ex:A",)),
            OntologyConcept(id="ex:A", label="ex:A"),
        ])
        assert onto.ids == ("ex:A",)

    def test_hierarchy_cycle_error_names_the_cycle_not_a_downstream_leaf(self):
        # 'aaa_leaf' has one parent and no cycle; it merely sorts first among
        # the unresolved set, which includes everything downstream of the cycle
        with pytest.raises(OntologyCycleError) as exc_info:
            Ontology.from_hierarchy({"mid1": {"mid2": {"mid3": {"mid1": None, "aaa_leaf": None}}}})
        message = str(exc_info.value)
        assert "'mid1'" in message
        assert "'mid2'" in message
        assert "'mid3'" in message
        assert "aaa_leaf" not in message

    def test_an_unrelated_equivalence_does_not_rewrite_other_concepts(self):
        # _absorb ran over every concept whenever any group existed, so an
        # unrelated concept's synonyms were silently deduplicated and stripped
        alone = Ontology([OntologyConcept(id="x", label="L", synonyms=("L", "S", "S"))])
        with_group = Ontology([
            OntologyConcept(id="x", label="L", synonyms=("L", "S", "S")),
            OntologyConcept(id="p", label="P", equivalent_to=("q",)),
        ])
        assert alone.concept("x").synonyms == ("L", "S", "S")
        assert with_group.concept("x").synonyms == alone.concept("x").synonyms

    def test_a_non_member_still_gets_its_parents_canonicalized(self):
        # the one rewrite a non-member does need, or the edge is lost
        onto = Ontology([
            OntologyConcept(id="ex:Car", label="Car", equivalent_to=("ex:Auto",)),
            OntologyConcept(id="ex:Auto", label="Auto"),
            OntologyConcept(id="ex:Sedan", label="Sedan", synonyms=("Sedan", "Saloon"), parents=("ex:Car",)),
        ])
        assert onto.concept("ex:Sedan").parents == ("ex:Auto",)
        assert onto.concept("ex:Sedan").synonyms == ("Sedan", "Saloon")  # untouched


@pytest.mark.required
def test_equivalence_merging_is_idempotent_for_already_joined_concepts():
    """Two owl:equivalentClass edges over the same pair must not double-merge."""
    pytest.importorskip("rdflib")
    turtle = TURTLE + (
        "\nex:Dog owl:equivalentClass ex:Hound .\n"
        "ex:Hound owl:equivalentClass ex:Dog .\n"
        "ex:Dog owl:equivalentClass ex:Hound .\n"
    )
    onto = Ontology.from_rdf(turtle, format="turtle")
    assert onto is not None
