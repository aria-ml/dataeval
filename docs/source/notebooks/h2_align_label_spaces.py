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
# # How to align two label spaces

# %% [markdown]
# ## Problem statement
#
# Two datasets rarely name things the same way. One annotates `"motorbike"` where
# another writes `"motorcycle"`; one has a single `"car"` class where another
# distinguishes `"sedan"` from `"suv"`. Before annotations from different sources
# can be compared, graded against one another, or combined, their labels must be
# *related*: which concept here corresponds to which concept there, and **how**.
#
# Establishing those correspondences is **ontology alignment**. DataEval performs
# it with {func}`.label_alignment`, which maps a *source* vocabulary onto a *target*
# (reference) {class}`.Ontology` and reports, for each source class, the typed
# correspondence it found — and whether the source can be expressed in the target
# without loss.

# %% [markdown]
# ### When to use
#
# Use this workflow when you have two label vocabularies and want to:
#
# - find which source classes are *equivalent* to a target class (a safe rename)
# - find which are *narrower* (a fine source class that can be safely **coarsened**
#   up to a more general target) or *broader* (a coarse source class that spans
#   several finer target classes — a granularity mismatch)
# - get a **class_remap** (`source -> target`) for carrying source labels into the
#   target vocabulary, and a **mergeability** verdict on how completely that can be
#   done
#
# Aligning a *structureless* source — a bare list of class names — against the
# target is exactly label [reconciliation](h2_reconcile_labels_ontology.py)
# ({func}`.label_reconciliation`) plus structural inference; `label_alignment`
# generalizes it.

# %% [markdown]
# ### What you will need
#
# 1. A source vocabulary (a list of class names, or an {class}`.Ontology`).
# 1. A target/reference {class}`.Ontology`.
# 1. Nothing else for the exact/structural core — it has no extra dependencies. The
#    optional fuzzy-matching recipe in section 4 uses `rapidfuzz`
#    (`pip install rapidfuzz`).

# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval rapidfuzz
except Exception:
    pass

# %%
from dataeval import Ontology
from dataeval.core import label_alignment

# %% [markdown]
# ## 1. A reference vocabulary
#
# Alignment maps a source onto a *target* — the reference vocabulary everything is
# expressed in. Here is a small two-level reference taxonomy, built dependency-free
# with {meth}`.Ontology.from_hierarchy`.

# %%
reference = Ontology.from_hierarchy({
    "vehicle": {"car": {"sedan": None, "suv": None}, "truck": None},
    "animal": {"dog": None, "cat": None},
})
print(reference)

# %% [markdown]
# ## 2. Align a list of class names
#
# The simplest source is another dataset's class names. Pass them to {func}`.label_alignment`
# with the reference. The result is a {class}`.LabelAlignmentResult` `TypedDict`.

# %%
result = label_alignment(["sedan", "truck", "spaceship"], reference)
print("keys:", list(result))

# %% [markdown]
# Each accepted mapping is a {class}`.Correspondence` — a typed
# ⟨source, target, relation, confidence⟩ tuple, with a note of the matcher that
# produced it:

# %%
for c in result["correspondences"]:
    print(f"  {c.source:>10}  {c.relation:<10} {c.target:<8} ({c.confidence:.2f}, {c.matcher})")

# %% [markdown]
# `class_remap` is the actionable artifact: the `source -> target` rewrite for the
# correspondences that license one (equivalence and coarsening). `unaligned_source`
# and `unaligned_target` are read *open-world* — an unaligned concept is
# out-of-vocabulary with respect to the other side, not invalid.

# %%
print("class_remap:     ", dict(result["class_remap"]))
print("unaligned_source:", result["unaligned_source"])
print("mergeability:    ", result["mergeability"])

# %% [markdown]
# `"spaceship"` is out-of-vocabulary, so the source is only **partially**
# expressible in the reference. On a structureless source this matches
# {func}`.label_reconciliation` exactly — the equivalence `class_remap` *is*
# the matched set:

# %%
from dataeval.core import label_reconciliation

rec = label_reconciliation(["sedan", "truck", "spaceship"], reference)
print("reconciliation matched:", dict(rec["matched"]))

# %% [markdown]
# ## 3. Reasoning across granularity
#
# The point of carrying a *relation* on each correspondence is that the relation,
# not just the pairing, says what the data permits. The asymmetry between coarsening
# and splitting is the heart of it.
#
# ### Source coarser than target → `broader` (a granularity mismatch)
#
# A model that only predicts `"car"` cannot be scored against the reference's
# `"sedan"`/`"suv"` ground truth without acknowledging that `"car"` spans both.
# `label_alignment` flags this with `broader` correspondences (diagnostics — they are
# deliberately **excluded** from `class_remap`, because splitting a coarse label into
# finer ones is not licensed by the relation alone).

# %%
coarse = label_alignment(["car"], reference)
for c in coarse["correspondences"]:
    print(f"  {c.source}  {c.relation:<10} {c.target}")
print("class_remap:", dict(coarse["class_remap"]), "| mergeability:", coarse["mergeability"])

# %% [markdown]
# ### Source finer than target → `narrower` (safe coarsening)
#
# The reverse direction *is* safe: every sedan is a car, so a fine source class can
# always be coarsened up to a more general target. When the source carries its own
# hierarchy, `label_alignment` propagates an equivalence anchor down to its descendants. Here
# the reference is coarse (`car` is a leaf) and the source is fine.

# %%
coarse_reference = Ontology.from_hierarchy({"vehicle": {"car": None, "truck": None}})
fine_source = Ontology.from_hierarchy({"car": {"sedan": None}})

result = label_alignment(fine_source, coarse_reference)
for c in result["correspondences"]:
    print(f"  {c.source:>6}  {c.relation:<10} {c.target}")
print("class_remap:", dict(result["class_remap"]), "| mergeability:", result["mergeability"])

# %% [markdown]
# `"sedan"` is coarsened to `"car"` — valid, but now `"sedan"` and `"car"` both map
# to `"car"` and can no longer be told apart, so the source is only **lossily**
# expressible. (A source is *lossless* only when its `class_remap` is injective.)

# %% [markdown]
# ## 4. Bring your own matcher: fuzzy name matching
#
# Exact matching anchors on labels, synonyms, and ids; typos and word-order
# variants slip past it. The `matchers=` argument is the **extension seam**: any
# object implementing the {class}`.Matcher` protocol can propose additional
# correspondences for the source concepts the exact pass left unanchored.
#
# DataEval ships the protocol and the dependency-free engine, not specific matchers
# — string similarity, embeddings, and instance overlap are different, tunable
# strategies. Here is a compact fuzzy matcher built on
# [`rapidfuzz`](https://github.com/rapidfuzz/RapidFuzz) (`pip install rapidfuzz`)
# that you can adapt or swap out.

# %%
from collections.abc import Iterable

from rapidfuzz import fuzz, process

from dataeval.types import Correspondence, OntologyConcept


class FuzzyMatcher:
    """A Matcher proposing equivalences by fuzzy string similarity."""

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold

    # A matcher only needs to iterate concepts (id / label / synonyms) — not the
    # full Ontology — so it accepts any iterable of OntologyConcept.
    def __call__(self, source: Iterable[OntologyConcept], target: Iterable[OntologyConcept]) -> list[Correspondence]:
        # case-folded target label/synonym -> concept id
        names = {n.casefold(): c.id for c in target for n in (c.label, *c.synonyms)}
        proposals = []
        for concept in source:
            match = process.extractOne(
                concept.label.casefold(),
                list(names),
                scorer=fuzz.token_sort_ratio,
                score_cutoff=self.threshold * 100,
            )
            if match is not None:
                proposals.append(
                    Correspondence(
                        source=concept.id,
                        target=names[match[0]],
                        relation="equivalent",
                        confidence=match[1] / 100,
                        matcher="fuzzy",
                    )
                )
        return proposals


# %% [markdown]
# It satisfies the {class}`.Matcher` protocol structurally — no inheritance needed:

# %%
from dataeval.protocols import Matcher

print("is a Matcher:", isinstance(FuzzyMatcher(), Matcher))

# %% [markdown]
# Pass it via `matchers=` to catch the near-miss names the exact pass missed:

# %%
messy = ["motorcyle", "sedans", "truck pickup", "stop sign"]  # 3 near-misses, 1 OOV
catalog = Ontology.from_hierarchy({"vehicle": {"motorcycle": None, "sedan": None, "pickup truck": None}})

fuzzy = label_alignment(messy, catalog, matchers=[FuzzyMatcher(threshold=0.85)])
for c in fuzzy["correspondences"]:
    print(f"  {c.source:>12}  ->  {c.target:<14} ({c.confidence:.2f}, {c.matcher})")
print("unaligned_source:", fuzzy["unaligned_source"])

# %% [markdown]
# A proposal below the acceptance threshold is **withheld** rather than guessed —
# alignment favors precision over recall, leaving `"stop sign"` unaligned for
# inspection. The same seam accepts an embedding- or instance-based matcher with no
# change to `label_alignment`.

# %% [markdown]
# ## 5. Aligning more than two sources
#
# To relate *several* sources, align each one to a single shared **reference
# (pivot)** vocabulary rather than every pair. Correspondences between sources are
# then read off through the pivot, and the per-source `class_remap`s express them all in
# one label space.

# %%
sources = {
    "dataset_a": ["sedan", "truck"],
    "dataset_b": ["suv", "dog"],
}
for name, labels in sources.items():
    r = label_alignment(labels, reference)
    print(f"{name:>10}: class_remap={dict(r['class_remap'])}  mergeability={r['mergeability']}")

# %% [markdown]
# ## Summary
#
# - {func}`.label_alignment` maps a source vocabulary (a list of names or an
#   {class}`.Ontology`) onto a target {class}`.Ontology`, returning a
#   {class}`.LabelAlignmentResult` of typed {class}`.Correspondence` objects.
# - Relations carry meaning: `equivalent` (rename) and `narrower` (safe coarsening)
#   populate `class_remap`; `broader` flags a granularity mismatch and is excluded.
# - `mergeability` summarizes how completely the source is expressible in the target
#   — `lossless`, `lossy` (specificity collapses), or `partial` (unaligned classes).
# - The `matchers=` seam accepts any {class}`.Matcher` (e.g. the fuzzy recipe above)
#   to catch near-miss names; below-threshold proposals are withheld, not guessed.
# - Align several sources to one shared **pivot** ontology to express them in a
#   common label space.

# %% [markdown]
# ## Related concepts
#
# - [Ontology](../concepts/Ontology.md) — the taxonomic model alignment operates on
#   and the reconciliation it generalizes, plus the [alignment
#   theory](../concepts/Ontology.md#alignment-relating-two-vocabularies):
#   correspondences, relations, matchers, mergeability, and the common cut.
# - [Distribution Shift](../concepts/DistributionShift.md) and
#   [Divergence](../concepts/Divergence.md) — the distributional differences between
#   sources that remain *after* their label spaces are aligned.
