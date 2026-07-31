"""Ontology concept and alignment types."""

__all__ = [
    "AlignmentRelation",
    "Correspondence",
    "OntologyConcept",
]

from typing import ClassVar, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


class OntologyConcept(BaseModel):
    """
    A single concept (class) within an ontology.

    A portable, serializable descriptor for one concept of a class hierarchy. The
    behavior-bearing container that links these into a queryable graph is
    :class:`dataeval.Ontology`.

    Attributes
    ----------
    id : str
        Stable identifier for the concept, typically an IRI or CURIE.
    label : str
        Preferred human-readable name (``skos:prefLabel`` or ``rdfs:label``).
    synonyms : tuple[str, ...]
        Alternate labels (``skos:altLabel``) used for matching.
    parents : tuple[str, ...]
        Ids of direct superclasses (``rdfs:subClassOf`` / ``skos:broader``).
        Parent ids that are not themselves defined concepts are treated as
        external references, not errors (ontologies are frequently
        distributed as subsets).
    definition : str or None
        Optional textual definition (``skos:definition``).
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    id: str
    label: str
    synonyms: tuple[str, ...] = ()
    parents: tuple[str, ...] = ()
    definition: str | None = None


AlignmentRelation: TypeAlias = Literal["equivalent", "narrower", "broader", "related"]
"""
How a :class:`Correspondence`'s source concept relates to its target concept.

The vocabulary mirrors the W3C SKOS mapping properties:

- ``"equivalent"`` (``skos:exactMatch`` / ``owl:equivalentClass``) — the two
  concepts denote the same class; the source label can be renamed losslessly.
- ``"narrower"`` (``skos:narrowMatch``) — the source is *more specific* than the
  target; the source label can be safely **coarsened** up to the target.
- ``"broader"`` (``skos:broadMatch``) — the source is *more general* than the
  target; carrying it over would require **splitting** it into finer targets, so
  the relation alone does not license a rewrite (diagnostic of a granularity
  mismatch).
- ``"related"`` (``skos:relatedMatch``) — the two are associated (e.g. share an
  ancestor) but neither subsumes the other; not a label rewrite.
"""


class Correspondence(BaseModel):
    """
    One typed mapping between a source and a target concept in an alignment.

    A correspondence is the unit of an :func:`ontology alignment
    <dataeval.core.label_alignment>`: it relates a concept in a *source* vocabulary to a
    concept in a *target* vocabulary by a :data:`relation <AlignmentRelation>`,
    with a confidence and a record of the matcher that produced it. The relation
    determines which label rewrite (if any) the correspondence licenses — see
    :data:`AlignmentRelation`.

    Attributes
    ----------
    source : str
        Id of the source concept (the concept being mapped *from*).
    target : str
        Id of the target concept (the reference concept being mapped *to*).
    relation : AlignmentRelation
        How ``source`` relates to ``target``.
    confidence : float
        Strength of the correspondence in ``[0.0, 1.0]``. Exact and structurally
        entailed correspondences are ``1.0``; approximate matchers report a score.
    matcher : str
        Name of the matcher or strategy that produced the correspondence (e.g.
        ``"exact"``, ``"structural"``, ``"fuzzy"``), for provenance.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    source: str
    target: str
    relation: AlignmentRelation
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    matcher: str = ""
