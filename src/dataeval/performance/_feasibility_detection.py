"""
Object detection for the feasibility evaluator: the checks on a detector's predictions and its AP witness.

The witness bounds one class's AP at one IoU and a cap of K detections per image. Its thresholds are fixed on the fit
split; on the audit, per image, it counts true and false positives at each threshold and the targets of the class,
and bounds AP below from a lower TP, an upper FP and an upper target-count endpoint, each a hybrid mean endpoint.
"""

__all__ = []

import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, cast

import numpy as np
from numpy.typing import NDArray

from dataeval._metadata._metadata import _caller_stacklevel
from dataeval.core._ap_bounds import ap_lower_staircase
from dataeval.core._detection_counts import check_boxes, detection_counts, kept_matches
from dataeval.performance._feasibility_classification import _make_bound, _mean, _UnavailableError
from dataeval.performance._feasibility_constants import WITNESS_QUANTILES
from dataeval.performance._feasibility_evidence import Assumption, Bound, Reason
from dataeval.performance._feasibility_target import TargetSpec
from dataeval.performance._feasibility_units import DetectionUnits, _declared_class_names, _declared_indices
from dataeval.types._target import whole_ids
from dataeval.utils._array import as_numpy

_SHAPE = "boxes (N, 4) as (x0, y0, x1, y1), labels (N,) and scores (N,) or (N, n_classes)"


class ImageDetections(NamedTuple):
    """One image's checked detections: float64 ``boxes`` ``(N, 4)``, ``labels`` ``(N,)`` and float64 ``scores``."""

    boxes: NDArray[np.float64]
    labels: NDArray[np.intp]
    scores: NDArray[np.float64]


def _labels(where: str, raw: Any, n_classes: int) -> NDArray[np.intp]:
    values = as_numpy(raw).reshape(-1)
    if values.dtype.kind == "b":  # whole_ids would read True and False as the classes 1 and 0
        raise ValueError(f"{where}: labels are booleans; pass integer indices of the declared classes.")
    labels = whole_ids(values, f"{where}: labels", "Pass integer indices of the declared classes.")
    outside = labels[(labels < 0) | (labels >= n_classes)]
    if len(outside):
        raise ValueError(
            f"{where}: label {int(outside[0])} is outside the classes the dataset declares, [0, {n_classes}) (without "
            "index2label, the labels its targets hold). Drop detections of other classes, or map the detector's "
            "labels onto the dataset's class indices."
        )
    return labels


def _scores(where: str, raw: Any, labels: NDArray[np.intp], n_classes: int) -> NDArray[np.float64]:
    """One score per detection: a per-class layout gives each detection the column of its own label."""
    scores = np.asarray(as_numpy(raw), dtype=np.float64)
    n = len(labels)
    if scores.ndim == 2 and scores.shape == (n, n_classes):
        scores = scores[np.arange(n), labels]
    elif scores.shape != (n,):
        raise ValueError(f"{where}: scores of shape {scores.shape} for {n} boxes; give ({n},) or ({n}, {n_classes}).")
    if not np.isfinite(scores).all():
        raise ValueError(f"{where}: scores must be finite; found NaN or infinite scores. Fix or drop those detections.")
    return scores


def check_detections(predictions: Sequence[Any], n_items: int, n_classes: int) -> list[ImageDetections]:
    """
    Check a detector's predictions, one detections object per dataset item in dataset order, and return them read.

    Each object carries ``boxes``, ``labels`` and ``scores``, the MAITE ``ObjectDetectionTarget`` shape. Labels are
    indices of the declared classes; a float array of whole values is accepted. Scores of shape ``(N, n_classes)``
    give each detection the score in its own label's column. Boxes are not clipped.

    Raises
    ------
    ValueError
        Naming the item and the fix, if there is not one detections object per item, an object is a mapping or lacks
        ``boxes``, ``labels`` or ``scores``, a box is not finite or has no positive width and height, labels are
        booleans, not whole numbers, not one per box, or outside ``[0, n_classes)``, or scores are not finite or
        neither ``(N,)`` nor ``(N, n_classes)``.
    """
    check_detection_layout(predictions, n_items)
    return [_image(f"predictions[{i}]", detections, n_classes) for i, detections in enumerate(predictions)]


def check_detection_layout(predictions: Any, n_items: int) -> None:
    """
    Refuse predictions that are not one detections object per item, each with ``boxes``, ``labels`` and ``scores``.

    It reads no box and needs no pass over the dataset, so a whole array or tensor, a wrong count, a mapping per image
    or an array per image is refused before one.
    """
    shape = getattr(predictions, "shape", None)  # one array or tensor for the whole dataset
    if shape is not None or not hasattr(predictions, "__len__") or len(predictions) != n_items:
        got = f"an array of shape {tuple(shape)}" if shape is not None else _count(predictions)
        raise ValueError(
            f"predictions for a detection dataset are one detections object per image ({n_items}), got {got}; "
            f"each holds {_SHAPE}."
        )
    for i, detections in enumerate(predictions):
        _check_fields(f"predictions[{i}]", detections)


def _check_fields(where: str, detections: Any) -> None:
    if isinstance(detections, Mapping):
        raise ValueError(
            f"{where} is a mapping; each image's detections need attributes boxes, labels and scores (the MAITE "
            "ObjectDetectionTarget shape), not keys. Wrap each dict, e.g. types.SimpleNamespace(**d)."
        )
    for name in ("boxes", "labels", "scores"):
        if not hasattr(detections, name):
            raise ValueError(f"{where} has no {name!r}; each image's detections hold {_SHAPE}.")


def _image(where: str, detections: Any, n_classes: int) -> ImageDetections:
    boxes = check_boxes(as_numpy(detections.boxes), where)
    labels = _labels(where, detections.labels, n_classes)
    if len(labels) != len(boxes):
        raise ValueError(f"{where}: {len(labels)} labels for {len(boxes)} boxes; give one label and score per box.")
    return ImageDetections(boxes, labels, _scores(where, detections.scores, labels, n_classes))


def _count(predictions: Any) -> str:
    return str(len(predictions)) if hasattr(predictions, "__len__") else f"a {type(predictions).__name__}"


# --- the checks of an object detection call ---------------------------------------------------------------------------

_DETECTION_CALL = (
    "Object detection takes the targets, each requirement's max_detections K and Feasibility(max_objects=...) for the "
    "capacity envelope, and predictions=, one detections object per image, for the witness."
)


def check_detection_call(
    *,
    model: Any,
    predictions: Any,
    embeddings: Any,
    replicas: Any,
    allowances: Any,
    specs: Sequence[TargetSpec],
    max_objects: Sequence[int],
) -> None:
    """
    Refuse, in one error, every argument an object detection call cannot use, and a missing ``predictions``.

    A missing ``predictions`` is refused only where a requirement's K is at or above its B, ``max_objects`` in
    ``specs`` order: its cap cannot bind, so only a witness can judge it.
    """
    unusable = {"model": model, "embeddings": embeddings, "replicas": replicas or None, "allowances": allowances}
    given = [f"{name}=" for name, value in unusable.items() if value is not None]
    verb = "applies" if len(given) == 1 else "apply"
    problems = [f"{', '.join(given)} {verb} to classification only"] if given else []
    unbound = [
        (f"{s.class_name!r} at K = {s.max_detections}, max_objects {b} (requirement {r})", b)
        for r, (s, b) in enumerate(zip(specs, max_objects, strict=True))
        if cast(int, s.max_detections) >= b  # an OD target always has its K
    ]
    if predictions is None and unbound:
        # K is at least 1, so a bound of 1 leaves no K below it to state
        fix = ", or state a max_detections below max_objects" if any(b > 1 for _, b in unbound) else ""
        problems.append(
            "predictions= is missing, and every requirement whose max_detections K is at or above max_objects needs "
            f"it, since its cap cannot bind and no capacity envelope is built: {', '.join(n for n, _ in unbound)}. "
            f"Pass predictions={fix}"
        )
    if problems:
        raise ValueError(f"{'; '.join(problems)}. {_DETECTION_CALL}")


def resolve_count_bounds(max_objects: int | Mapping[str, int] | None, specs: Sequence[TargetSpec]) -> list[int]:
    """B for each requirement, in ``specs`` order: an int for every class, or the mapping's value for its class."""
    classes = list(dict.fromkeys(cast(str, s.class_name) for s in specs))  # an OD target always names its class
    if max_objects is None:
        raise ValueError(
            "Object detection needs Feasibility(max_objects=...), B: the largest number of targets of the class in "
            "any image of the population, assumed and not verified. Pass one int for every class, or a mapping from "
            f"class name to B, e.g. {{{classes[0]!r}: 18}}."
        )
    if not isinstance(max_objects, Mapping):
        return [max_objects] * len(specs)
    missing = [c for c in classes if c not in max_objects]
    if missing:
        raise ValueError(
            f"Feasibility(max_objects=...) has no bound for {missing}: its keys are {list(max_objects)}, and the "
            f"requirements name {classes}. Add a key for every requirement class."
        )
    return [max_objects[cast(str, s.class_name)] for s in specs]


def resolve_class_ids(dataset: Any, specs: Sequence[TargetSpec]) -> list[int]:
    """
    Map each requirement's class to its label, read without a pass over the data.

    The label is the class's declared id in ``index2label``, sparse ids included, else the class name read as a digit
    string.
    """
    names, declared = _declared_class_names(dataset), _declared_indices(dataset)
    ids: list[int] = []
    for spec in specs:
        name = cast(str, spec.class_name)
        if names is not None:
            ids.append(declared[names.index(name)])  # parse_targets checked that index2label declares it
        elif name.isdecimal():
            ids.append(int(name))
        else:
            raise ValueError(
                f"Class {name!r} cannot be mapped to a label: the dataset declares no index2label, so a class is "
                "named by its label as a digit string, e.g. '0'. Declare index2label on the dataset metadata, or "
                "name the class by its label."
            )
    return ids


def check_counts(
    units: DetectionUnits, class_ids: Sequence[int], max_objects: Sequence[int], specs: Sequence[TargetSpec]
) -> list[NDArray[np.intp]]:
    """
    Check each requirement's class against the data: a label the data can hold, and at most B targets per image.

    One error lists every class over its B, with its largest count, the item that holds it and B. It returns, for each
    requirement in ``specs`` order, the number of targets of its class in every image.
    """
    item_ids = units.item_ids.tolist()
    over: dict[str, str] = {}
    out: list[NDArray[np.intp]] = []
    for spec, c, b in zip(specs, class_ids, max_objects, strict=True):
        name = cast(str, spec.class_name)
        if c >= units.n_classes:  # only without index2label: a declared id is always below n_classes
            found = sorted({int(label) for labels in units.labels for label in labels})
            raise ValueError(
                f"Class {name!r} is label {c}, but the dataset declares no index2label and its targets hold labels "
                f"{found}. Name the class by a label its targets hold, or declare index2label."
            )
        counts = np.array([np.count_nonzero(labels == c) for labels in units.labels], dtype=np.intp)
        out.append(counts)
        i = int(np.argmax(counts))  # the first image with the largest count
        if counts[i] > b:
            over[name] = (
                f"{name!r}: {counts[i]} targets in item {item_ids[i]!r} (dataset index {i}), above max_objects {b}"
            )
    if over:
        raise ValueError(
            f"Some images hold more targets of a class than Feasibility(max_objects=...) declares: "
            f"{'; '.join(over.values())}. max_objects bounds every image of the population, so the largest count in "
            "the data is only a floor: raise it to a bound you can justify for the population."
        )
    return out


# --- the AP witness ---------------------------------------------------------------------------------------------------

_ASSUMED: tuple[Assumption, ...] = (
    "model_independent_of_audit_items",
    "class_order_matches_dataset",
    "max_objects_holds",
)


def detection_grid(fit_scores: NDArray[np.float64], fit_matched: NDArray[np.bool_]) -> NDArray[np.float64]:
    """
    Fix the witness thresholds on the fit split: ``-inf``, then the quantiles of the matched kept scores.

    With no match the quantiles are of every kept score, and with no kept score the grid is ``-inf`` alone. ``-inf``
    counts every kept detection, so the staircase can reach the detector's top recall.
    """
    if not len(fit_scores):
        return np.array([-np.inf])
    chosen = fit_scores[fit_matched] if fit_matched.any() else fit_scores
    return np.r_[-np.inf, np.unique(np.quantile(chosen, WITNESS_QUANTILES))]


def ap_from_counts(
    tp: NDArray[Any],
    fp: NDArray[Any],
    n: NDArray[Any],
    *,
    max_objects: int,
    max_detections: int,
    alpha_claim: float,
) -> tuple[float, dict[str, Any]]:
    """
    Bound AP below from per-image counts: ``tp`` and ``fp`` of shape ``(n_audit, T)``, targets ``n`` ``(n_audit,)``.

    Each of the ``2T + 1`` hybrid endpoints spends ``alpha_claim / (2T + 1)``: per threshold a lower TP endpoint, with
    support ``min(B, K)``, and an upper FP endpoint, with support ``K``; then an upper endpoint on the targets per
    image, with support ``B``. Fewer than 2 images raise ``_UnavailableError("too_few_units")``.
    """
    tp, fp = np.asarray(tp, dtype=np.float64), np.asarray(fp, dtype=np.float64)
    t = tp.shape[1]
    delta = alpha_claim / (2 * t + 1)
    tp_results = [_mean(tp[:, j], min(max_objects, max_detections), delta, "lower") for j in range(t)]
    fp_results = [_mean(fp[:, j], max_detections, delta, "upper") for j in range(t)]
    mu_result = _mean(np.asarray(n, dtype=np.float64), max_objects, delta, "upper")
    tp_lower, fp_upper = [r["endpoint"] for r in tp_results], [r["endpoint"] for r in fp_results]
    value = ap_lower_staircase(np.array(tp_lower), np.array(fp_upper), mu_result["endpoint"])
    raw = {"value": value, "tp_lower": tp_lower, "fp_upper": fp_upper, "mu_upper": mu_result["endpoint"]}
    return value, raw | {
        "tp_results": tp_results,
        "fp_results": fp_results,
        "mu_result": mu_result,
        "deltas": [delta] * (2 * t + 1),
    }


def capacity_from_counts(
    n: NDArray[Any], *, max_objects: int, max_detections: int, alpha_claim: float
) -> tuple[float, dict[str, Any]]:
    """
    Bound AP above for any detector capped at K detections per image, from the targets per image ``n`` ``(n_images,)``.

    Four hybrid endpoints each spend ``alpha_claim / 4``: the count per image, lower and upper, with support ``B``;
    the shortfall ``(N - K)+``, lower, with support ``B - K``; and ``min(N, K)``, upper, with support ``K``. The value
    is the smaller of the direct form ``U[min(N, K)] / L[N]`` (1 when ``L[N]`` is 0) and the risk form
    ``1 - L[(N - K)+] / U[N]``. ``raw`` also holds the sample ratio ``mean``, None when no image holds a target. Fewer
    than 2 images raise ``_UnavailableError("too_few_units")``.

    Raises
    ------
    ValueError
        If ``max_detections`` is not below ``max_objects``. Such a cap cannot bind and the bound is 1 for any data, so
        build no capacity bound for it.

    Notes
    -----
    A detector with at most K detections in an image matches at most ``min(N, K)`` of its N targets, so its recall at
    every operating point is at most ``E[min(N, K)] / E[N]``. Precision is at most 1, so AP, the integral of the
    interpolated precision over recall, is at most that ratio too [1]_, and the value also caps recall. Since
    ``E[min(N, K)] = E[N] - E[(N - K)+]``, both forms bound the same ratio, and they hold together when the four
    endpoints do, with probability at least ``1 - alpha_claim``. To first order, with radii in proportion to the
    supports, the direct form is the tighter one when the ratio is below ``1 - K/B``.

    References
    ----------
    .. [1] DFE manuscript, capacity evidence (the output-cap bounds on risk and AP).
    """
    if max_detections >= max_objects:
        raise ValueError(
            f"max_detections ({max_detections}) must be below max_objects ({max_objects}) for a capacity bound; at or "
            "above it the cap cannot bind and the bound is 1, so build none."
        )
    n = np.asarray(n, dtype=np.float64)
    delta = alpha_claim / 4
    count_lower = _mean(n, max_objects, delta, "lower")
    count_upper = _mean(n, max_objects, delta, "upper")
    shortfall_lower = _mean(np.maximum(n - max_detections, 0), max_objects - max_detections, delta, "lower")
    matched_upper = _mean(np.minimum(n, max_detections), max_detections, delta, "upper")
    low = count_lower["endpoint"]
    direct = min(1.0, matched_upper["endpoint"] / low) if low > 0 else 1.0
    risk = 1.0 - shortfall_lower["endpoint"] / count_upper["endpoint"]  # U[N] > 0: every radius is positive
    value = max(0.0, min(direct, risk))
    total = n.sum()
    return value, {
        "value": value,
        "count_lower": count_lower,
        "count_upper": count_upper,
        "shortfall_lower": shortfall_lower,
        "matched_upper": matched_upper,
        "mean": float(np.minimum(n, max_detections).sum() / total) if total else None,
        "n": len(n),
        "deltas": [delta] * 4,
    }


def _of_class(
    units: DetectionUnits, detections: Sequence[ImageDetections], rows: NDArray[np.intp], c: int
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Select the target boxes, detection boxes and detection scores of class ``c`` on the images at ``rows``."""
    targets, boxes, scores = [], [], []
    for i in map(int, rows):
        mine = detections[i].labels == c
        targets.append(units.boxes[i][units.labels[i] == c])
        boxes.append(detections[i].boxes[mine])
        scores.append(detections[i].scores[mine])
    return targets, boxes, scores


def ap_witness(
    units: DetectionUnits,
    detections: Sequence[ImageDetections],
    fit: NDArray[np.intp],
    audit: NDArray[np.intp],
    *,
    class_id: int,
    class_name: str,
    iou: float,
    max_detections: int,
    max_objects: int,
    alpha_claim: float,
) -> tuple[float, dict[str, Any]]:
    """
    Bound the AP of class ``class_name`` (label ``class_id``) below on the audit, from thresholds fixed on the fit.

    Every audit image counts, an image with no target or detection of the class included. ``raw`` holds the
    thresholds and what :func:`ap_from_counts` records. A fit split whose targets of the class match none of its kept
    detections warns.
    """
    fit_targets, fit_boxes, fit_scores = _of_class(units, detections, fit, class_id)
    scores, matched = kept_matches(fit_targets, fit_boxes, fit_scores, iou=iou, max_detections=max_detections)
    if len(scores) and not matched.any() and any(len(t) for t in fit_targets):
        warnings.warn(
            f"None of the {len(scores)} kept detections of class {class_name!r} (label {class_id}) on the fit split "
            f"matches a target at IoU {iou}, so the witness thresholds come from every kept score. Check that the "
            "boxes are (x0, y0, x1, y1), in the same image frame as the targets (pixels, not normalized), and that "
            "the labels follow the dataset's class order.",
            UserWarning,
            stacklevel=_caller_stacklevel(),  # the caller's own line, through evaluate's wrappers or directly
        )
    thresholds = detection_grid(scores, matched)
    targets, boxes, audit_scores = _of_class(units, detections, audit, class_id)
    counts = detection_counts(
        targets, boxes, audit_scores, iou=iou, thresholds=thresholds, max_detections=max_detections
    )
    value, raw = ap_from_counts(
        counts["true_positives"],
        counts["false_positives"],
        np.array([len(t) for t in targets]),
        max_objects=max_objects,
        max_detections=max_detections,
        alpha_claim=alpha_claim,
    )
    return value, raw | {"thresholds": thresholds.tolist()}


def gather_detection_evidence(
    specs: Sequence[TargetSpec],
    *,
    units: DetectionUnits,
    detections: Sequence[ImageDetections],
    fit: NDArray[np.intp],
    audit: NDArray[np.intp],
    alpha_claim: float,
    class_ids: Sequence[int],
    max_objects: Sequence[int],
) -> list[Bound]:
    """
    One ``user_model`` AP witness bound per requirement, in ``specs`` order, each spent at ``alpha_claim``.

    ``class_ids`` and ``max_objects`` give each requirement's declared class index and B, in the same order.
    """
    out: list[Bound] = []
    for r, (spec, c, b) in enumerate(zip(specs, class_ids, max_objects, strict=True)):
        reason: Reason | None = None
        try:
            value, raw = ap_witness(
                units,
                detections,
                fit,
                audit,
                class_id=c,
                class_name=cast(str, spec.class_name),  # an OD target always has its class, IoU and K
                iou=cast(float, spec.iou),
                max_detections=cast(int, spec.max_detections),
                max_objects=b,
                alpha_claim=alpha_claim,
            )
        except _UnavailableError as e:
            value, raw, reason = None, None, e.reason
        spent = 0.0 if raw is None else math.fsum(raw["deltas"])
        if spent > alpha_claim + 1e-12:
            raise RuntimeError(
                f"Internal error: the detection witness splits {spent} inside an ap bound, over its claim {alpha_claim}"
            )
        shared = {
            "requirement": r,
            "source": "user_model",
            "role": "witness",
            "information_scope": "full_input",
            "population": "items",
            "sample": "audit_split",
            "unit": "image",
            "assumptions": _ASSUMED,
        }
        out.append(_make_bound(shared, "ap", value, None, alpha_claim, reason))
    return out


# --- the capacity envelope --------------------------------------------------------------------------------------------


def capacity_evidence(
    specs: Sequence[TargetSpec], *, counts: Sequence[NDArray[Any]], alpha_claim: float, max_objects: Sequence[int]
) -> list[Bound]:
    """
    One ``capacity`` AP envelope per requirement whose K is below its B, in ``specs`` order, each at ``alpha_claim``.

    ``counts`` and ``max_objects`` give each requirement's targets per image and B, in the same order. A requirement
    with K at or above B gets no row: its cap cannot bind. The rows read the targets alone, never detections or a split.
    """
    out: list[Bound] = []
    for r, (spec, n, b) in enumerate(zip(specs, counts, max_objects, strict=True)):
        k = cast(int, spec.max_detections)  # an OD target always has its K
        if k >= b:
            continue
        reason: Reason | None = None
        try:
            value, raw = capacity_from_counts(n, max_objects=b, max_detections=k, alpha_claim=alpha_claim)
        except _UnavailableError as e:
            value, raw, reason = None, None, e.reason
        spent = 0.0 if raw is None else math.fsum(raw["deltas"])
        if spent > alpha_claim + 1e-12:
            raise RuntimeError(
                f"Internal error: the capacity envelope splits {spent} inside an ap bound, over its claim {alpha_claim}"
            )
        shared = {
            "requirement": r,
            "source": "capacity",
            "role": "envelope",
            "information_scope": "full_input",  # it bounds any predictor, so it decides at every scope
            "population": "items",
            "sample": "all_items",
            "unit": "image",
            "assumptions": ("max_objects_holds",),
        }
        stats = None if raw is None else {"mean": raw["mean"], "n": raw["n"]}
        out.append(_make_bound(shared, "ap", value, stats, alpha_claim, reason))
    return out
