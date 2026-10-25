# ---
# jupyter:
#   jupytext:
#     default_lexer: ipython3
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: dataeval
#     language: python
#     name: python3
# ---

# %% [markdown]
# # How to reconcile labels against an ontology
#
# A dataset's class names belong to a domain taxonomy: "sedan" and "pickup truck" are both land vehicles, and "fighter jet" is
# an aircraft. {func}`.label_reconciliation` matches class names against a reference {class}`.Ontology` and recovers their
# place in its hierarchy. See [Ontology](../concepts/Ontology.md).
#
# Use it when you have a set of class names and a reference ontology, and you want to:
#
# - check which names map to known concepts, and which are unmatched or ambiguous
# - recover each class's is-a path and its relationships to the other classes
#
# Matching is exact on preferred labels, synonyms and ids, and an unmatched label means "not found," not "invalid." Reading
# RDF, OWL or JSON-LD needs `dataeval[ontology]`.

# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval[ontology]
except Exception:
    pass

# %%
from dataeval import Ontology
from dataeval.core import label_reconciliation
from dataeval.types import OntologyConcept

# %% [markdown]
# ## Build an ontology
#
# The simplest, dependency-free way to define a taxonomy is from a plain nested
# dictionary with {meth}`.Ontology.from_hierarchy`. Mapping values may be
# `None` (a leaf), a list of child labels, or a further nested mapping.

# %%
ontology = Ontology.from_hierarchy({
    "vehicle": {
        "land vehicle": {"sedan": None, "pickup truck": None},
        "watercraft": {"frigate": None, "cargo ship": None},
        "aircraft": {"airliner": None, "fighter jet": None},
    },
})
print(ontology)

# %% [markdown]
# The `repr` summarizes the concepts, roots, leaves, and external references (more on
# those below).

# %% [markdown]
# ## Reconcile a dataset's class names
#
# Pass your class names — typically `index2label.values()` — to
# {func}`.label_reconciliation`. Here the label set includes a class that isn't in
# the ontology.

# %%
index2label = {0: "sedan", 1: "pickup truck", 2: "fighter jet", 3: "rowboat"}

result = label_reconciliation(index2label.values(), ontology)

# %% [markdown]
# The return value is a {class}`.LabelReconciliationResult` — a `TypedDict` whose
# keys fall into two groups: a *match report* (`matched`, `unmatched`,
# `ambiguous`) and the recovered *hierarchy* of the matched classes
# (`ancestor_paths`, `external_ancestors`, `induced_edges`, `relations`).

# %%
print("keys:", list(result))

# %% [markdown]
# Start with the match report:

# %%
print("matched:  ", result["matched"])
print("unmatched:", result["unmatched"])
print("ambiguous:", result["ambiguous"])

# %% [markdown]
# `"rowboat"` is flagged as `unmatched` — it isn't a concept in this ontology.
# The remaining classes resolved to concepts. The result also recovers hierarchy
# information for the matched classes.

# %%
# Each matched class's is-a path, from nearest parent up to the root
for name, path in result["ancestor_paths"].items():
    print(f"{name:>14}  <-  {' < '.join(path)}")

# %% [markdown]
# Pairwise `relations` describe how matched classes relate to one another
# (`ancestor`, `descendant`, `sibling`, or `unrelated`):

# %%
print("sedan vs pickup truck:", result["relations"][("sedan", "pickup truck")])
print("sedan vs fighter jet: ", result["relations"][("sedan", "fighter jet")])

# %% [markdown]
# When your label set includes classes at different levels of the hierarchy,
# `induced_edges` gives the minimal is-a tree connecting just those classes
# (intermediate concepts are collapsed):

# %%
label_reconciliation(["vehicle", "land vehicle", "sedan"], ontology)["induced_edges"]

# %% [markdown]
# ## Load an ontology from OWL, RDF or JSON-LD
#
# Parse OWL/RDF/JSON-LD content that is already in memory with
# {meth}`.Ontology.from_rdf` (this requires the `dataeval[ontology]` extra).
# DataEval does not read files itself: load the bytes/text however you like
# (here, a small inline JSON-LD document) and pass them in.

# %%
JSONLD = """
{
  "@context": {
    "owl": "http://www.w3.org/2002/07/owl#",
    "subClassOf": {"@id": "http://www.w3.org/2000/01/rdf-schema#subClassOf", "@type": "@id"},
    "prefLabel": {"@id": "http://www.w3.org/2004/02/skos/core#prefLabel"},
    "altLabel": {"@id": "http://www.w3.org/2004/02/skos/core#altLabel"},
    "definition": {"@id": "http://www.w3.org/2004/02/skos/core#definition"},
    "cv": "http://example.org/cv#"
  },
  "@graph": [
    {"@id": "cv:Aircraft", "@type": "owl:Class", "prefLabel": "Aircraft"},
    {"@id": "cv:FighterJet", "@type": "owl:Class", "subClassOf": "cv:Aircraft",
     "prefLabel": "Fighter Jet",
     "altLabel": ["F-16", "Viper"],
     "definition": "A fast, maneuverable military aircraft."}
  ]
}
"""

owl_ontology = Ontology.from_rdf(JSONLD, format="json-ld")
print(owl_ontology)

# %% [markdown]
# Concepts are identified by their IRI, while labels and synonyms are used for
# matching. `find` resolves a name (case-insensitively) across preferred labels,
# synonyms, and exact ids — so an annotator's `"F-16"` resolves to the canonical
# *Fighter Jet* concept:

# %%
print("find('F-16'):", owl_ontology.find("F-16"))

concept = owl_ontology.concept("http://example.org/cv#FighterJet")
print("label:     ", concept.label)
print("synonyms:  ", concept.synonyms)
print("definition:", concept.definition)

# %% [markdown]
# ## Work with an incomplete ontology
#
# Ontologies are frequently distributed as subsets, where a concept's parent is
# referenced but not itself included. DataEval keeps these as *external
# references* rather than failing — they still participate in hierarchy queries, and
# {func}`.label_reconciliation` reports where a class's is-a path is truncated via
# `external_ancestors`.

# %%
subset = Ontology([
    # 'warship' is referenced as a parent but never defined in this subset
    OntologyConcept(id="frigate", label="Frigate", parents=("warship",)),
    OntologyConcept(id="sedan", label="Sedan"),
])
print("external_ids:", subset.external_ids)

subset_result = label_reconciliation(["Frigate", "Sedan"], subset)
print("external_ancestors:", subset_result["external_ancestors"])

# %% [markdown]
# `"Frigate"` matched, but its hierarchy is unresolved above `warship`. `"Sedan"` is
# fully rooted, so it is absent from `external_ancestors`.

# %% [markdown]
# ## Explore the hierarchy
#
# The {class}`.Ontology` object exposes dependency-free graph queries for ad-hoc
# exploration:

# %%
print("ancestors(sedan): ", ontology.ancestors("sedan"))
print("siblings(sedan):  ", ontology.siblings("sedan"))
print("descendants(vehicle):", ontology.descendants("vehicle"))
print("depth_of(sedan):  ", ontology.depth_of("sedan"))
print("leaves:           ", ontology.leaves)

# %% [markdown]
# Extract a focused sub-ontology rooted at any concept with `subtree` (parent
# links pointing outside the subtree are pruned, so the concept becomes a root):

# %%
watercraft = ontology.subtree("watercraft")
print(repr(watercraft))
print("ids:", sorted(watercraft.ids))

# %% [markdown]
# ## Next steps
#
# - [Ontology](../concepts/Ontology.md) — Learn about ontologies, taxonomies, and label reconciliation concepts.
# - [Data Integrity](../concepts/DataIntegrity.md) — Understand label-quality checks and data consistency validation.
