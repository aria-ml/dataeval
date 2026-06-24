# ---
# jupyter:
#   jupytext:
#     default_lexer: ipython3
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: dataeval
#     language: python
#     name: python3
# ---

# %% [markdown]
# # How to reconcile labels against an ontology

# %% [markdown]
# ## Problem statement
#
# A dataset's class names rarely live in isolation — they belong to a domain
# *taxonomy*. "sedan" and "pickup truck" are both land vehicles; "fighter jet" is
# an aircraft. Knowing this hierarchy lets you sanity-check labels (does every
# class actually exist in the reference vocabulary?) and reason about
# relationships between them (which classes are siblings, which subsume others).
#
# DataEval represents a taxonomy with the {class}`.Ontology` class — a small,
# in-memory, strongly-typed graph of concepts — and reconciles a dataset's class
# names against it with {func}`.label_reconciliation`.

# %% [markdown]
# ### When to use
#
# Use this workflow when you have a set of class names (e.g.
# `index2label.values()`) and a reference ontology, and you want to:
#
# - check which class names map to known concepts (and which are unmatched or
#   ambiguous)
# - recover each class's place in the hierarchy (its is-a path to the root)
# - understand pairwise relationships (ancestor / descendant / sibling) between
#   your classes
#
# This is *exact* reconciliation (matching on preferred labels, synonyms, and
# ids). Fuzzy / semantic *normalization* of messy labels is a separate, future
# capability — here, an "unmatched" label simply means "not found in the
# ontology," not "invalid."

# %% [markdown]
# ### What you will need
#
# 1. A set of class names to reconcile.
# 1. A reference ontology.
# 1. A Python environment with `dataeval[ontology]` installed.
#
# ```{note}
# Parsing a reference ontology from an RDF/OWL/JSON-LD source requires the
# `ontology` extra, which pulls in `rdflib`. Building an ontology in
# memory needs no extra dependencies.
# ```

# %% [markdown]
# ## Getting started
#
# Import the pieces you need.

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
# ## 1. Build an ontology
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
# The `repr` summarizes the structure: total concepts, roots (top-level
# concepts), leaves (most specific), and external references (more on those below).

# %% [markdown]
# ## 2. Reconcile a dataset's class names
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
# (`ancestor_paths`, `external_ancestors`, `induced_edges`, `relations`). The rest
# of this section walks through each.

# %%
print("keys:", list(result))

# %% [markdown]
# Start with the match report — which class names resolved to a concept, and which
# did not:

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
# ## 3. Richer ontologies from OWL / RDF / JSON-LD
#
# Real ontologies usually ship as standards-based OWL/RDF/JSON-LD files with
# preferred labels, synonyms, and definitions. Parse already-in-memory content
# with {meth}`.Ontology.from_rdf` (this requires the `dataeval[ontology]` extra).
# DataEval does not read files itself — load the bytes/text however you like
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
# ## 4. Incomplete (subset) ontologies
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
# `"Frigate"` matched, but its hierarchy is unresolved above `warship` — useful
# for deciding whether the subset is sufficient or the full ontology is needed.
# `"Sedan"` is fully rooted, so it is absent from `external_ancestors`.

# %% [markdown]
# ## 5. Exploring the hierarchy
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
# ## Summary
#
# - {meth}`.Ontology.from_hierarchy` builds a taxonomy from plain Python with no
#   extra dependencies; {meth}`.Ontology.from_rdf` parses standards-based
#   OWL/RDF/JSON-LD (with the `dataeval[ontology]` extra).
# - {func}`.label_reconciliation` returns a {class}`.LabelReconciliationResult`
#   reporting `matched` / `unmatched` / `ambiguous` classes plus hierarchy
#   (`ancestor_paths`, `induced_edges`, `relations`) and flags truncated
#   hierarchies via `external_ancestors`.
# - The {class}`.Ontology` object supports graph queries (`ancestors`,
#   `descendants`, `siblings`, `depth_of`, `subtree`, …) for exploration.
#
# Reconciliation here is exact (labels, synonyms, ids). Fuzzy and semantic
# normalization of messy raw labels is planned future work — an `unmatched`
# label means "not present in this ontology," and is the natural input to such a
# normalization step.

# %% [markdown]
# ## Related concepts
#
# - [Ontology](../concepts/Ontology.md) — what an ontology is, the vocabulary it
#   uses, and how reconciliation and conformance are defined.
# - [Data Integrity](../concepts/DataIntegrity.md) — the other label-quality
#   checks reconciliation sits alongside.
