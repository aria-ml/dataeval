<!-- markdownlint-disable MD051 -->

# Ontology

A dataset's class names rarely stand alone — they belong to a domain
_vocabulary_ with structure. "sedan" and "pickup truck" are both land vehicles;
"fighter jet" is an aircraft, not a watercraft. When that structure is written
down in a machine-readable form, a team can do two things it otherwise cannot:

- Check that every label a dataset uses is a sanctioned concept.
- Reason about how those labels relate to one another and to the boundary of
  what the model is meant to recognize.

DataEval calls that written-down structure an {class}`.Ontology`. This page
defines what we mean by the term, introduces the vocabulary the library uses, and
maps each term to both the formal knowledge-representation standards it derives
from and the language working computer-vision practitioners use day to day.

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

For a detection or tracking workload, the ontology is the reference against which
a dataset's **label space** is judged.

- **Reconciliation and conformance.** Object-detection datasets carry a
  _category set_ (COCO's `categories`, a TensorFlow `label_map`, a MAITE
  `index2label` map). Names drift across datasets and annotation passes:
  `"motorbike"` vs `"motorcycle"`, `"car"` vs `"sedan"`, `"person"` vs
  `"pedestrian"`. _Reconciling_ those names against an ontology for each class
  answers: is this a known concept, an unknown one, or an ambiguous one that
  matches several? A label set in which all names reconcile unambiguously
  _conforms_ to the vocabulary. See [Data Integrity](DataIntegrity.md) for where
  this sits among other label checks.

- **Label space.** The ontology delimits what the model is _supposed_ to
  recognize — its label space, $\mathcal{Y}$. A label outside it signals an
  unsanctioned class (one nobody agreed to) or a gap (a real concept the ontology
  hasn't captured yet). The is-a structure also lets you reason about granularity:
  a model trained on `"vehicle"` cannot be graded against `"sedan"` ground truth
  without first relating the two through the hierarchy.

- **Tracking and class consistency.** In multi-object tracking, an object's class
  should be stable along its track. A shared ontology gives a single authority for
  what the legal classes are and how a coarse detection (`"vehicle"`) subsumes a
  fine one (`"ambulance"`), so class assignments can be reconciled across frames
  and across detectors rather than compared as opaque strings.

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

### Reconciliation, conformance, and the label space

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

## The taxonomic core vs. the operational annotation schema

"Ontology" is used in computer vision at two levels of richness, and because the
word is overloaded it is worth being explicit about the split. **DataEval models
the first — the taxonomic core.** The second — the operational annotation schema —
is out of scope, except for two checks (covered at the end) that are taxonomic
despite where annotation platforms file them.

- **The taxonomic (semantic) core** (what {class}`.Ontology` models) — concepts
  related by is-a, carrying labels, synonyms, and definitions. This is what
  {class}`.Ontology` _is_: a portable, format-neutral hierarchy you validate label
  _names_ against and reason over (ancestor / descendant / sibling / LCA). It comes
  from the OWL/RDF/SKOS tradition and is what COCO supercategories, WordNet, and the
  Open Images hierarchy express.

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
they relate_ — what {func}`.label_reconciliation` does today, over class _names_
alone. The operational schema additionally validates _how each instance was
annotated_: that a `tree` was drawn as a `polygon` and not a `bounding_box`, that
a required `color` attribute is present and one of its allowed options, that every
frame carries its scene classifications.

The operational annotation schema validates per-instance annotation data —
geometry type, attribute values — that DataEval's current dataset model (an
`ObjectDetectionTarget` of `boxes`, `labels`, `scores`) does not carry. It is
therefore **out of scope for the present taxonomy model** and would be a
separate, schema-driven validator rather than an extension of {class}`.Ontology`.

Two checks that annotation platforms bundle into the operational layer are
nonetheless purely _taxonomic_ and within DataEval's scope:

- **Naming / format linting** — flag concept labels that are not
  `lowercase_snake_case`, or that mix separators, to keep the vocabulary uniform.
- **Structural smells** — a concept and one of its own ancestors appearing as
  _siblings_ (e.g. `car` placed alongside `vehicle`), over-specification (many
  leaf concepts carrying very few samples, when paired with a dataset), or
  lopsided depth across the hierarchy.

## How this maps to standards and datasets

The vocabulary above is not invented for DataEval; it is the intersection of
three well-established traditions, which is why it should read as familiar to
domain experts:

- **Knowledge-representation standards.** The concept/label/definition fields and
  the is-a relation come directly from W3C [RDFS/OWL](#ref5) (`owl:Class`,
  `rdfs:subClassOf`, `rdfs:label`) and [SKOS](#ref4)
  (`skos:Concept`, `skos:prefLabel`, `skos:altLabel`, `skos:broader`,
  `skos:definition`). {meth}`.Ontology.from_rdf` reads exactly these.

- **Lexical hierarchies.** [WordNet](#ref2)'s synsets and hypernym (is-a) links
  are the model [ImageNet](#ref3) used to organize 1000+ visual categories into a
  hierarchy — the original large-scale "ontology for computer vision," and the
  source of the _synset_ framing behind our synonyms.

- **Detection dataset taxonomies.** [COCO](#ref6) groups its 80 `categories` under
  12 `supercategories` (`"vehicle"` over `"car"`, `"bus"`, `"truck"`) — a one-level
  is-a hierarchy. [Open Images](#ref7) ships an explicit multi-level class
  hierarchy, and autonomous-driving benchmarks such as [nuScenes](#ref8) define a
  class taxonomy _with attributes_ for detection and tracking — precisely the
  "ontology" object annotation platforms expose to labelers.

A note on the **open-world** framing: classical detection benchmarks are
_closed-set_ (a fixed category list), and DataEval's exact reconciliation matches
that — but the same vocabulary extends to _open-vocabulary detection_
([ViLD](#ref9)), where the ontology becomes the controlled set of concept names a
model is queried with rather than a fixed integer label map.

## Related concept pages

- [Data Integrity](DataIntegrity.md) — where label reconciliation sits among the
  other label-quality checks (duplicates, outliers, label errors).

## See this in practice

### How-to guides

- [How to reconcile labels against an ontology](../notebooks/h2_reconcile_labels_ontology.py)
  — build an ontology, reconcile a dataset's class names, and explore the
  recovered hierarchy.

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
   [spec](https://www.w3.org/TR/skos-reference/)]{#ref4}

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
