<!-- markdownlint-disable MD051 -->

# Ontology

A dataset's class names rarely stand alone — they belong to a domain
_vocabulary_ with structure. "sedan" and "pickup truck" are both land vehicles;
"fighter jet" is an aircraft, not a watercraft. Written down in a machine-readable
form, that structure lets a team do three things it otherwise cannot:

- Check that every label a dataset uses is a sanctioned concept.
- Reason about how those labels relate — to one another and to the boundary of
  what the model is meant to recognize.
- Relate one dataset's vocabulary to another's, so labels from different sources
  can be compared, graded against each other, or combined.

DataEval calls that written-down structure an {class}`.Ontology`. This page
defines the term, the vocabulary the library uses, and the operations DataEval
performs over an ontology — **reconciliation** (checking one dataset's labels
against it), **alignment** (relating two vocabularies through it), and
**validation** (checking the ontology artifact itself) — mapping each to the formal
knowledge-representation standards it derives from and the language working
computer-vision practitioners use day to day.

## What is it

In the formal sense established by [Gruber (1993)](#ref1), an ontology is "an
explicit specification of a conceptualization" — a declared inventory of the
concepts in a domain and the relationships among them. The full machinery of
ontologies (arbitrary relations, logical axioms, inference) is more than label
reconciliation needs. DataEval uses the part that matters for a class vocabulary: a
set of **concepts** arranged by a single relationship, **is-a** (subsumption).

Strictly, a structure built from is-a alone is a **taxonomy** (a subsumption
hierarchy) or, in the vocabulary of the W3C [SKOS](#ref4) standard, a _concept
scheme_ related by `broader`/`narrower`. We nonetheless name the class
{class}`.Ontology` deliberately, because **"ontology"** is the word the
computer-vision annotation industry has settled on for exactly this artifact:
the controlled set of classes (and their attributes and relationships) that
annotators must conform to. Annotation platforms — Scale AI, Labelbox, Encord,
V7, CVAT — all label this object the project's "ontology." Matching that usage
keeps the term recognizable to the practitioners who will use the library, while
the formal grounding below keeps it precise.

Concretely, an {class}`.Ontology` is an immutable, in-memory **directed acyclic
graph** (DAG) of {class}`.OntologyConcept` nodes, queryable for ancestors,
descendants, siblings, lowest common ancestors, and the like. It is a DAG rather
than a strict tree because a concept may have **more than one parent** (an
amphibious vehicle is-a land vehicle _and_ is-a watercraft) — multiple
inheritance is normal in real taxonomies. Cyclic inheritance, by contrast, is
rejected: a concept cannot be its own ancestor, or subsumption becomes meaningless.

## Why it matters for object detection and tracking

For a detection or tracking workload, the ontology is the reference a dataset's
**label space** is judged against — and, when more than one label space is in
play, the bridge between them.

- **Reconciliation and conformance.** Detection datasets carry a _category set_
  (COCO's `categories`, a TensorFlow `label_map`, a MAITE `index2label` map), and
  names drift across datasets and annotation passes: `"motorbike"` vs
  `"motorcycle"`, `"car"` vs `"sedan"`, `"person"` vs `"pedestrian"`. _Reconciling_
  each name against the ontology answers whether it is a known concept, an unknown
  one, or an ambiguous one that matches several; a label set in which all names
  reconcile unambiguously _conforms_ to the vocabulary.

- **Label space and granularity.** The ontology delimits what the model is
  _supposed_ to recognize — its label space, $\mathcal{Y}$. A label outside it
  signals an unsanctioned class (one nobody agreed to) or a gap (a real concept the
  ontology hasn't captured yet). The is-a structure also lets you reason across
  granularity: a model trained on `"vehicle"` cannot be graded against `"sedan"`
  ground truth until the two are related through the hierarchy.

- **Relating sources.** Datasets carve the world up differently — [COCO](#ref6)
  groups 80 categories under 12 supercategories, [Open Images](#ref7) ships a
  multi-level hierarchy, [nuScenes](#ref8) a driving-specific taxonomy. Relating
  their category sets through a shared ontology is what lets annotations from one
  be read in terms of another, rather than compared as opaque strings.

- **Tracking and class consistency.** When detectors with different class lists
  feed one tracker, an object's class should stay stable along its track. A shared
  ontology is the single authority for which classes are legal and how a coarse
  detection (`"vehicle"`) subsumes a fine one (`"ambulance"`), so assignments can
  be reconciled across frames and detectors rather than string-matched.

## Vocabulary

Each term below is defined with its DataEval API name, the knowledge-representation
standard it derives from, and the everyday CV term it corresponds to.

### Core elements

**Concept** — the unit of an ontology: one class in the hierarchy
({class}`.OntologyConcept`). This is the canonical noun; prefer it over "node,"
"class," or "category" when referring to an element _of the ontology_. It
corresponds to `owl:Class` / `skos:Concept` in the standards, and to a "class,"
"category," or "label" in CV usage. We reserve **class name** for the
_dataset-side_ string being checked, to distinguish the thing being validated
(a dataset's label) from the thing it is validated against (an ontology concept).

**Id** — a concept's unique, stable identifier (`OntologyConcept.id`), typically
an IRI or CURIE. Each id identifies exactly one concept. Ids, not labels, are
what hierarchy queries return and compare.

**Label** — a concept's preferred human-readable name
(`OntologyConcept.label`; `skos:prefLabel`, falling back to `rdfs:label`). One
per concept.

**Synonyms** — a concept's alternate labels (`OntologyConcept.synonyms`;
`skos:altLabel`), used so that `"motorbike"` can resolve to a concept whose
preferred label is `"motorcycle"`. The WordNet tradition behind
[ImageNet](#ref3) calls such an equivalence set a _synset_.

**Definition** — a concept's optional textual gloss
(`OntologyConcept.definition`; `skos:definition`).

### Structural relations

The ontology models exactly one relation between concepts — **is-a**
(subsumption: `rdfs:subClassOf` / `skos:broader`) — and exposes it through a
family of query terms. CV practitioners and graph libraries use two
interchangeable vocabularies for it: a _subsumption_ vocabulary
(subclass/superclass) and a _graph_ vocabulary (ancestor/descendant). DataEval
uses both and treats them as exact equivalents:

| DataEval term                    | is-a meaning                              | Standard / graph term              |
| -------------------------------- | ----------------------------------------- | ---------------------------------- |
| **parent** / **child**           | direct super- / subconcept (one step)     | `rdfs:subClassOf` / `skos:broader` |
| **ancestor** = **superclass**    | transitive parent (broader)               | DAG ancestor                       |
| **descendant** = **subclass**    | transitive child (narrower)               | DAG descendant                     |
| **root**                         | concept with no parents                   | source node                        |
| **leaf**                         | concept with no children (most specific)  | sink node                          |
| **sibling**                      | shares a parent                           | —                                  |
| **depth**                        | longest is-a path from a root             | longest path length                |
| **lowest common ancestor** (LCA) | deepest shared superclass of two concepts | DAG LCA                            |

So {meth}`ontology.is_a(a, b) <.Ontology.is_a>` ("a is a subclass of b") and
"b is an ancestor (superclass) of a" describe the same fact;
{meth}`ontology.ancestors(a) <.Ontology.ancestors>` returns concept `a`'s
superclasses, {meth}`ontology.descendants(a) <.Ontology.descendants>` its
subclasses.

### Defined vs. external concepts

Ontologies are frequently distributed as **subsets** — a detection project may
ship only the slice of a large taxonomy its classes touch. DataEval distinguishes
two kinds of id accordingly:

**Defined concept** — a concept actually present in the ontology, with a label
and (optionally) synonyms, a definition, and parents. The `Ontology` container
counts, iterates, and resolves only defined concepts.

**External reference** — an id named as a _parent_ of some concept but not itself
defined in the ontology (`Ontology.external_ids`). Externals are kept, not
rejected: they still participate in ancestor and LCA queries, but they have no
label or further ancestors, so they mark the point where the is-a hierarchy is
**truncated**. An external reference means "this concept's parent exists in the
fuller ontology, but isn't included here." Earlier terminology called these
"external boundary nodes"; the canonical term is _external reference_.

## Reconciliation: checking labels against the ontology

Checking a dataset's labels against an ontology is two distinct operations: the
matching _operation_ (**reconciliation**) and the _property_ it establishes
(**conformance**). The library's {func}`.label_reconciliation` performs both —
reconciling each class name and reporting whether the label set conforms.

**Reconciliation** — the operation of mapping a dataset's class name to concept
id(s) by matching it (case-insensitively) against preferred labels, synonyms, and
exact ids (`Ontology.find`). This is the term used in data curation
(OpenRefine, Wikidata) for matching free-text values to a controlled vocabulary;
the NLP equivalents are entity / term **normalization** and **grounding**. A
class name falls into one of three outcomes:

- **matched** — reconciled to exactly one concept.
- **unmatched** — reconciled to no concept. Note this is the **open-world**
  reading: unmatched means _"not found in this ontology"_, i.e.
  **out-of-vocabulary (OOV)**; it does not mean _"invalid"_. DataEval does exact
  reconciliation, not fuzzy normalization — an unmatched label may be a genuine gap
  in the ontology, a typo, or simply outside the intended label space.
- **ambiguous** — reconciled to more than one concept (e.g. a synonym shared by
  two concepts); disambiguate upstream by passing a concept id.

**Conformance** — the property a label set has when every class name reconciles
unambiguously to a concept (no unmatched, no ambiguous). What
{func}`.label_reconciliation` reports is a **conformance check** of the dataset's
label set against a controlled vocabulary — similar to how the W3C SHACL standard
gives "validation."

**Induced sub-hierarchy** — given a set of matched classes spanning different
levels, the minimal is-a tree connecting just those classes, with intermediate
concepts collapsed (`induced_edges`; formally the _transitive reduction_ of the
hierarchy restricted to the matched set). This is what lets a mixed label set
like `{"vehicle", "sedan", "fighter jet"}` be drawn as a clean parent/child
structure.

**Label space** — the set of concepts a dataset's labels are
expected to fall within; the standard ML name for the class set $\mathcal{Y}$, and
in formal terms the ontology's _domain of discourse_. An ontology _is_ a
declaration of the label space: reconciled labels are in-vocabulary; unmatched
(OOV) labels and unexpected externals are where the label space and the dataset
disagree. In open-set terms, the ontology specifies the **known** classes, against
which OOV labels stand out as **novel**.

## Alignment: relating two vocabularies

Reconciliation checks one label set against one ontology. **Alignment**
(equivalently _ontology matching_) is its generalization: relating a whole
_source_ vocabulary to a _target_ one, so heterogeneous label spaces can be
treated as one. Reconciliation is the special case of a structureless source,
exact name matching, and equivalence alone; relax those restrictions and it
becomes alignment. DataEval performs alignment with {func}`.label_alignment`,
which maps a source vocabulary (class names or an {class}`.Ontology`) onto a
target {class}`.Ontology` and reports a typed correspondence for each source class.

### What an alignment is

In the formal treatment of the field ([Euzenat & Shvaiko, 2013](#ref12); its
shared evaluation benchmark is the [OAEI](#ref13)), an **alignment** between a
source and a target ontology is a set of **correspondences**, each a tuple

$$\langle e_s,\; e_t,\; r,\; c \rangle$$

relating a source entity $e_s$ to a target entity $e_t$ by a **relation** $r$ —
equivalence ($\equiv$), subsumption ($\sqsubseteq$ / $\sqsupseteq$), or
relatedness — with a **confidence** $c \in [0,1]$. The alignment is the accepted
correspondences together with the entities on each side left **unaligned**. This
is the same is-a core an ontology models, now considered two at a time with a
typed mapping between them; logical reasoning and other relations stay out of
scope, as they do from the model itself.

A correspondence is not merely an assertion that two concepts are associated — it
licenses a concrete **transformation of a label**. Equivalence licenses renaming a
source label to its target; subsumption licenses rewriting a label _up_ the
hierarchy to a more general concept. Which rewrites an alignment permits, and which
it must refuse, follows entirely from the relations it carries (see
[Relations](#relations)) — which is what makes alignment the prerequisite for
relating or combining annotations across sources, rather than an end in itself.

### Source, target, and reference vocabulary

The **source** is the vocabulary being mapped _from_ (another dataset's classes);
the **target** is the reference vocabulary being mapped _to_. Equivalence is
symmetric, but subsumption is directional, so an alignment has an orientation. When
more than two sources must be related, aligning every pair is quadratic and yields
no single result; the standard resolution is a **pivot** (reference) ontology —
each source aligns to one shared vocabulary, and correspondences between sources
are read off through it. The unaligned set is read **open-world**: an unaligned
concept is out-of-vocabulary with respect to the other side, not invalid — possibly
a genuine gap between the vocabularies.

### Relations

A correspondence's **relation** says how the two concepts correspond, and
therefore what rewrite of the source label it licenses. The relations derive from
the W3C [SKOS mapping properties](#ref4) and are most usefully understood by the
rewrite each permits:

| relation                     | Transformation it licenses                                           | Effect on the data                                  | SKOS / OWL                                |
| ---------------------------- | -------------------------------------------------------------------- | --------------------------------------------------- | ----------------------------------------- |
| **equivalent** ($\equiv$)    | rename source label → target concept                                 | lossless                                            | `skos:exactMatch` / `owl:equivalentClass` |
| **narrower** ($\sqsubseteq$) | **coarsen**: rewrite a fine source label up to a more general target | valid; specificity is lost                          | `skos:narrowMatch`                        |
| **broader** ($\sqsupseteq$)  | would require **splitting** a coarse source label into finer targets | underdetermined; not licensed by the relation alone | `skos:broadMatch`                         |
| **related** ($\oplus$)       | none — associated (shared ancestor) but neither subsumes the other   | not a label rewrite                                 | `skos:relatedMatch`                       |

The asymmetry between **narrower** and **broader** is fundamental: coarsening a
specific label to a general one is always valid (every sedan is a vehicle), while
specializing a general label to a specific one is not (not every vehicle is a
sedan). An alignment can therefore safely carry a source into a target by
equivalence or coarsening, but a broader correspondence is evidence of a
_granularity mismatch_ the relation alone cannot resolve — {func}`.label_alignment`
emits these as diagnostics, not rewrites. (SKOS's `skos:closeMatch` denotes a
weaker, not-quite-exact equivalence — the natural reading of a high-but-imperfect
confidence.)

### Matchers

A **matcher** is a method for proposing correspondences. Ontology matching
classifies them into families ([Euzenat & Shvaiko, 2013](#ref12)) by the evidence
they use:

- **Element-level, terminological** — compares concept _names_: exact match over
  labels, synonyms, and ids ({meth}`.Ontology.find`); approximate (fuzzy) string
  match for variants and typos; embedding similarity for synonyms with no shared
  surface form (`"automobile"` ↔ `"car"`).
- **Structure-level, taxonomic** — compares position in the is-a graph: shared
  ancestors and the {meth}`lowest common ancestor <.Ontology.lowest_common_ancestor>`,
  {meth}`sibling <.Ontology.siblings>` sets, and
  {meth}`descendant <.Ontology.descendants>` overlap. When several sibling targets
  score near-equally, the structural reading favors a **broader** correspondence to
  their shared parent over an arbitrary equivalence to one sibling.
- **Extensional, instance-based** — compares the _instances_ that fall under each
  concept rather than its name. When the concepts are dataset classes, the overlap
  of their distributions in an [embedding](Embeddings.md) space is direct evidence
  for a correspondence, and the _direction_ of distributional containment is
  evidence for subsumption versus equivalence — independent of, and complementary
  to, the name-based families.

{func}`.label_alignment` anchors exact terminological matches first, then consults
any additional {class}`.Matcher` implementations supplied for the concepts left
unanchored, then propagates structurally up the hierarchy. Logical / deductive
(reasoning-based) matchers are outside the present scope.

### Confidence and abstention

Correspondences carry a **confidence** $c \in [0,1]$ and a record of the matcher
that produced them. The cost of error is asymmetric — a false correspondence
silently misrepresents one side's data as the other, worse than producing no
correspondence at all — so alignment favors **precision over recall**: a
correspondence below the acceptance `threshold` is withheld, leaving the concept
unaligned for inspection rather than committing a likely-wrong mapping. Abstention
is the conservative default, consistent with the open-world reading of the
unaligned set.

### Mergeability and the common cut

When the purpose of an alignment is to express several sources in one vocabulary,
two properties summarize it.

**Mergeability** — the generalization of reconciliation's _conformance_. A source
is _losslessly expressible_ in the reference if every class aligns by equivalence
or coarsening; _lossily expressible_ if coarsening discards needed specificity; and
only _partially expressible_ if some classes are broader, related, or unaligned and
cannot be carried over without additional evidence. {func}`.label_alignment`
reports this verdict alongside the safe label `class_remap`.

**Common cut (frontier)** — to express several sources at a comparable granularity,
each is projected onto a shared **antichain** of the reference hierarchy: the
finest set of concepts that _every_ source can reach by equivalence or coarsening,
a "cut" across the is-a graph. The cut fixes the effective granularity of the
combined label space and makes the granularity/coverage trade-off explicit (see
hierarchical classification, [Silla & Freitas, 2011](#ref14)).

Alignment relates **what the labels mean**. It does not address differences in
geometry conventions, sensor domains, or sampling — the distributional gaps between
sources that are the subject of [distribution shift](DistributionShift.md) and
[divergence](Divergence.md), and that persist after the label spaces are reconciled.

## Validation: checking the ontology artifact

Reconciliation and alignment judge data _against_ an ontology; both presume the
ontology itself is sound. {func}`.ontology_validation` turns the lens on the
**artifact**, reporting the structural and naming facts that bear on its quality,
independent of any dataset. An {class}`.Ontology` already guarantees the hard
invariants at construction — unique ids and an acyclic is-a graph — so what remains
to check is the **legal-but-questionable** structure those invariants do not preclude.

Like reconciliation, it reports **ingredients, not a verdict**: an empty finding is
the "clean" signal, but whether a finding is a defect is contextual — a dangling
ancestor is expected in a deliberately distributed subset and a problem only in an
ontology meant to be complete. The call records four families of fact:

- **Connectivity** — the `roots` and `leaves`, `isolated` concepts (with neither
  parents nor children), and `external_ancestors`: concepts whose is-a path is
  truncated at an undefined ("floating") parent.
- **Redundancy and contradiction** — `redundant_edges` (a direct is-a edge already
  implied by a longer path), `ancestor_siblings` (a concept declared alongside one
  of its own ancestors, e.g. `car` placed next to `vehicle`), and `unary_parents` (a
  single-child link, which adds depth without discriminating).
- **Naming** — `label_collisions`, names resolving to more than one concept (the
  artifact-side cause of reconciliation _ambiguity_), and, when a `label_pattern` is
  supplied, `nonconforming_labels` that fail it (e.g. a `lowercase_snake_case` lint).
- **Shape** — the per-concept `depth`, `fan_out`, and `parent_count`: the raw
  material for judging depth imbalance, over-broad parents, and multiple-inheritance
  load, without the function imposing a threshold of its own.

Turning these facts into a pass/fail call — which findings matter, at what severity,
where the thresholds sit — is policy left to a downstream evaluator, the same way a
conformance verdict is read off a reconciliation result rather than baked into it.

## The taxonomic core vs. the operational annotation schema

"Ontology" is used in computer vision at two levels of richness, and because the
word is overloaded it is worth being explicit about the split. **DataEval models
the first — the taxonomic core.** The second — the operational annotation schema —
is out of scope, except for two checks — performed by {func}`.ontology_validation`
(see [Validation](#validation-checking-the-ontology-artifact)) — that are taxonomic
despite where annotation platforms file them.

- **The taxonomic (semantic) core** (what {class}`.Ontology` models) — concepts
  related by is-a, carrying labels, synonyms, and definitions. This is what
  {class}`.Ontology` _is_: a portable, format-neutral hierarchy you validate label
  _names_ against and reason over (ancestor / descendant / sibling / LCA). It comes
  from the OWL/RDF/SKOS tradition and is what COCO supercategories, WordNet, and the
  Open Images hierarchy express. Both reconciliation and alignment stay within this
  core.

- **The operational annotation schema** (out of scope — see below) — the richer
  object that annotation platforms (Encord, Labelbox, [Avala](#ref11), V7) call a
  project's "ontology" ([Encord](#ref10)). On top of the class hierarchy it binds,
  per class:
  - a **geometry / drawing tool** (`bounding_box`, `polygon`, `bitmask`,
    `polyline`, `keypoint` / `skeleton`) the annotator must use;
  - **nested attributes**, each typed (`text`, `radio`, `checklist`) with an
    allowed **option set**, sometimes **conditional** on another attribute (a
    `vehicle` of `type: emergency` requires a `siren` attribute);
  - **required** toggles, at the object level and for **frame-level
    classifications** — scene metadata such as `time_of_day` or `weather` that
    must be present on every frame of a tracked sequence;
  - **naming conventions** (typically `lowercase_snake_case`), enforced as a lint.

The taxonomic core validates _which concepts a dataset's labels denote and how
they relate_ — what {func}`.label_reconciliation` and {func}`.label_alignment` do
today, over class _names_ alone. The operational schema additionally validates _how
each instance was annotated_: that a `tree` was drawn as a `polygon` and not a
`bounding_box`, that a required `color` attribute is present and one of its allowed
options, that every frame carries its scene classifications.

The operational annotation schema validates per-instance annotation data —
geometry type, attribute values — that DataEval's current dataset model (an
`ObjectDetectionTarget` of `boxes`, `labels`, `scores`) does not carry. It is
therefore **out of scope for the present taxonomy model** and would be a
separate, schema-driven validator rather than an extension of {class}`.Ontology`.

Two checks that annotation platforms bundle into the operational layer are
nonetheless purely _taxonomic_, and DataEval performs them on the artifact in
{func}`.ontology_validation`:

- **Naming / format linting** — flagging concept labels that mix separators or are
  not `lowercase_snake_case`, to keep the vocabulary uniform.
- **Structural smells** — a concept and one of its own ancestors appearing as
  _siblings_ (`car` placed alongside `vehicle`), redundant is-a edges, or
  single-child chains.

One related check is _not_ artifact-only: over-specification — many leaf concepts
carrying very few samples — needs a _dataset_ to judge, so it sits with
reconciliation rather than with validation, which reports depth and breadth as
metrics for an evaluator to weigh.

## How this maps to standards and datasets

The vocabulary above is not invented for DataEval; it is the intersection of
well-established traditions, which is why it should read as familiar to domain
experts:

- **Knowledge-representation standards.** The concept/label/definition fields and
  the is-a relation come directly from W3C [RDFS/OWL](#ref5) (`owl:Class`,
  `rdfs:subClassOf`, `rdfs:label`) and [SKOS](#ref4) (`skos:Concept`,
  `skos:prefLabel`, `skos:altLabel`, `skos:broader`, `skos:definition`).
  {meth}`.Ontology.from_rdf` reads exactly these.

- **Ontology-matching standards.** The correspondence tuple, the matcher families,
  and the evaluation framing come from [Euzenat & Shvaiko (2013)](#ref12) and the
  [OAEI](#ref13) campaigns; the relation set is the SKOS mapping properties
  (`skos:exactMatch`, `skos:broadMatch`, `skos:narrowMatch`, `skos:relatedMatch`,
  `skos:closeMatch`) and [OWL](#ref5) (`owl:equivalentClass`, cross-ontology
  `rdfs:subClassOf`).

- **Lexical hierarchies.** [WordNet](#ref2)'s synsets and hypernym (is-a) links
  are the model [ImageNet](#ref3) used to organize 1000+ visual categories into a
  hierarchy — the original large-scale "ontology for computer vision," and the
  source of the _synset_ framing behind our synonyms.

- **Detection dataset taxonomies.** [COCO](#ref6) groups its 80 `categories` under
  12 `supercategories` (a one-level is-a hierarchy), [Open Images](#ref7) ships an
  explicit multi-level hierarchy, and [nuScenes](#ref8) defines a driving taxonomy
  _with attributes_ for detection and tracking. Relating these category sets is the
  concrete CV instance of ontology matching — and the case where structural and
  extensional evidence matter most, because the same surface name sits at different
  granularities across taxonomies.

A note on the **open-world** framing: classical detection benchmarks are
_closed-set_ (a fixed category list), and DataEval's exact reconciliation matches
that — but the same vocabulary extends to _open-vocabulary detection_
([ViLD](#ref9)), where the ontology becomes the controlled set of concept names a
model is queried with rather than a fixed integer label map.

## Related concept pages

- [Data Integrity](DataIntegrity.md) — where reconciliation sits among the other
  label-quality checks (duplicates, outliers, label errors).
- [Embeddings](Embeddings.md) — the space in which extensional, instance-based
  matching compares concepts.
- [Distribution Shift](DistributionShift.md) and [Divergence](Divergence.md) — the
  distributional differences between sources that remain once labels are
  reconciled, and that alignment does not address.

## See this in practice

### How-to guides

- [How to reconcile labels against an ontology](../notebooks/h2_reconcile_labels_ontology.py)
  — build an ontology, reconcile a dataset's class names, and explore the
  recovered hierarchy.
- [How to align label spaces](../notebooks/h2_align_label_spaces.py) — map one
  vocabulary onto another, read the typed correspondences, and get a mergeability
  verdict.

## References

1. [Gruber, T. R. (1993). A translation approach to portable ontology
   specifications. _Knowledge Acquisition_, 5(2), 199–220.
   doi: 10.1006/knac.1993.1008
   [paper](https://tomgruber.org/writing/ontolingua-kaj-1993.htm)]{#ref1}

2. [Miller, G. A. (1995). WordNet: A lexical database for English.
   _Communications of the ACM_, 38(11), 39–41.
   doi: 10.1145/219717.219748
   [paper](https://dl.acm.org/doi/10.1145/219717.219748)]{#ref2}

3. [Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009).
   ImageNet: A large-scale hierarchical image database. In _CVPR_ (pp. 248–255).
   doi: 10.1109/CVPR.2009.5206848
   [paper](https://ieeexplore.ieee.org/document/5206848)]{#ref3}

4. [Miles, A., & Bechhofer, S. (2009). SKOS Simple Knowledge Organization System
   Reference. _W3C Recommendation._
   [spec](https://www.w3.org/TR/skos-reference/) ·
   [mapping properties](https://www.w3.org/TR/skos-reference/#mapping)]{#ref4}

5. [W3C OWL Working Group. (2012). OWL 2 Web Ontology Language Document Overview
   (2nd ed.). _W3C Recommendation._
   [spec](https://www.w3.org/TR/owl2-overview/)]{#ref5}

6. [Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D.,
   Dollár, P., & Zitnick, C. L. (2014). Microsoft COCO: Common objects in context.
   In _ECCV_ (pp. 740–755). doi: 10.1007/978-3-319-10602-1_48
   [paper](https://arxiv.org/abs/1405.0312)]{#ref6}

7. [Kuznetsova, A., Rom, H., Alldrin, N., Uijlings, J., Krasin, I., Pont-Tuset, J.,
   Kamali, S., Popov, S., Malloci, M., Kolesnikov, A., Duerig, T., & Ferrari, V.
   (2020). The Open Images Dataset V4: Unified image classification, object
   detection, and visual relationship detection at scale. _International Journal of
   Computer Vision_, 128(7), 1956–1981. doi: 10.1007/s11263-020-01316-z
   [paper](https://arxiv.org/abs/1811.00982)]{#ref7}

8. [Caesar, H., Bankiti, V., Lang, A. H., Vora, S., Liong, V. E., Xu, Q.,
   Krishnan, A., Pan, Y., Baldan, G., & Beijbom, O. (2020). nuScenes: A
   multimodal dataset for autonomous driving. In _CVPR_ (pp. 11621–11631).
   doi: 10.1109/CVPR42600.2020.01164
   [paper](https://arxiv.org/abs/1903.11027)]{#ref8}

9. [Gu, X., Lin, T.-Y., Kuo, W., & Cui, Y. (2022). Open-vocabulary object
   detection via vision and language knowledge distillation (ViLD). In _ICLR._
   [paper](https://arxiv.org/abs/2104.13921)]{#ref9}

10. [Encord. Ontologies — platform documentation. Accessed 2026.
    [docs](https://docs.encord.com/)]{#ref10}

11. [Avala. Annotation platform — schema/ontology documentation. Accessed 2026.
    [site](https://www.avala.ai/)]{#ref11}

12. [Euzenat, J., & Shvaiko, P. (2013). _Ontology Matching_ (2nd ed.). Springer.
    doi: 10.1007/978-3-642-38721-0
    [book](https://link.springer.com/book/10.1007/978-3-642-38721-0)]{#ref12}

13. [Ontology Alignment Evaluation Initiative (OAEI). Annual ontology-matching
    evaluation campaigns. [site](http://oaei.ontologymatching.org/)]{#ref13}

14. [Silla, C. N., & Freitas, A. A. (2011). A survey of hierarchical classification
    across different application domains. _Data Mining and Knowledge Discovery_,
    22(1–2), 31–72. doi: 10.1007/s10618-010-0175-9
    [paper](https://link.springer.com/article/10.1007/s10618-010-0175-9)]{#ref14}
