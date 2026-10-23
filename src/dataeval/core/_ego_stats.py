"""Statistics describing camera ego-motion in video sequences.

Camera motion is a shortcut hazard. A detector can learn "the camera was panning" instead
of learning the object, whenever panning correlates with a label. These statistics measure
the motion so an audit can condition on it.

The estimator is deliberately cheap. It uses dense DIS optical flow at an ultrafast preset,
subsampled to a coarse lattice. A shortcut is by definition cheaply extractable, so a cheap
estimator gives an upper bound on what a model could pick up for free.

Model
-----
For one frame pair, with ``x`` the pixel offset from the image center and ``v = (u, w)``
the flow in pixels, the camera contributes an affine field::

    v(x) = A x + b,    A = [[a00, a01], [a10, a11]]

whose parts name the physical motions::

    div   = a00 + a11                 looming; zoom or dolly, signed in/out
    curl  = a10 - a01                 roll; a pure roll of phi gives curl = 2 phi
    shear = (a00 - a11, a01 + a10)    off-axis or oblique motion
    t0    = b                         flow at the center; the pan/tilt rate
    FOE   = -A^-1 b                   zero of the field, where the camera is heading

``A`` is a ratio of pixels to pixels, so it is resolution-free. ``div``, ``curl`` and
``shear`` are per-frame rates that do not change when the frames are downscaled. ``t0`` and
``FOE`` are lengths, reported in the full-resolution pixels of the pair each was measured on.

The fit excludes independent movers instead of modeling them. Rejection happens inside
the robust fit itself, which is IRLS with Tukey biweight weights. That fit is deterministic
and does none of the random sampling RANSAC would. No mask leaves the fit. What survives
is reported on three separate axes, so an audit can condition on each independently:
camera kinematics, an independent-motion proxy, and per-frame trust flags. Declared
roll-ups summarize all three to one row per sequence.
"""

from __future__ import annotations

__all__ = []

from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from functools import partial
from types import ModuleType
from typing import Any, Generic, Literal, NamedTuple, TypeAlias, TypedDict, TypeVar, cast, overload

import numpy as np
from numpy.typing import NDArray

from dataeval._log import get_logger
from dataeval.core._compute_stats import FactorResult
from dataeval.core._video_runner import FrameView, VideoClock, VideoOptions, run_videos, walk
from dataeval.protocols import MultiobjectTrackingDataset, VideoStream
from dataeval.types import Aggregator

_logger = get_logger(__name__)

FlowMethod = Literal["dis", "lk"]
"""Which optical flow estimator to use: dense DIS, or sparse Lucas-Kanade on a lattice."""

#: Tukey biweight tuning constant, giving 95% efficiency against Gaussian noise.
_TUKEY_C = 4.685
#: Scale factor turning a median absolute deviation into a Gaussian-consistent sigma.
_MAD_TO_SIGMA = 1.4826
#: Weight above which a sample counts as an inlier to the fit it was weighted against.
_INLIER_WEIGHT = 0.05
#: Smallest number of flow samples an affine fit is attempted on. A fit from 3 samples
#: would be exactly determined.
_MIN_SAMPLES = 12
#: Smallest processed side, in pixels, that a frame pair is read at. Measured on OpenCV 4.11.0, DIS fails on
#: some shapes with a side under 32 px, raising or crashing the process natively depending on the aspect ratio.
#: As (height, width), (31, 640) raises and (10, 64) and (28, 512) crash, while (31, 31), (31, 55), (16, 28),
#: (16, 64) and (28, 48) run. At 32 px or more nothing failed at any width up to 16384, or with the axes
#: transposed. The rule covers both estimators, so there is one rule: Lucas-Kanade never failed, but a frame
#: that small gives no reliable motion either way.
_MIN_FLOW_SIDE = 32


def _require_cv2() -> ModuleType:
    """Import OpenCV, naming the extra that provides it.

    OpenCV is an optional dependency and ``dataeval.core`` is imported eagerly, so a
    top-level import would make the whole package unimportable without the extra. The
    import is therefore deferred. Matches :class:`~dataeval.extractors.BoVWExtractor`.
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError(
            "ego_stats requires 'opencv-python' or related package. "
            "Please install it via 'pip install opencv-python' or using the extra `dataeval[opencv]`.",
        ) from e
    return cv2


# ============================== robust affine fit ==============================


class _AffineFit(NamedTuple):
    """One robust affine fit of a flow field, in the coordinates it was fitted in.

    Attributes
    ----------
    au, aw : NDArray[np.float64]
        The two decoupled 3-parameter solutions, ``[d/dx, d/dy, constant]`` for the ``u``
        and ``w`` flow components respectively. The constant is the flow at `centroid`,
        because the coordinates are centered there.
    centroid : NDArray[np.float64]
        Mean sample position the fit was centered on.
    weights : NDArray[np.float64]
        Final biweight weight per sample, in ``[0, 1]``.
    condition : float
        Ratio of the smallest to the largest singular value of the normal matrix. Near
        zero where the sample geometry cannot determine an affine field, such as when all
        samples are collinear. It is scale-free, unlike the singular value itself, which
        grows with both sample count and coordinate magnitude.
    """

    au: NDArray[np.float64]
    aw: NDArray[np.float64]
    centroid: NDArray[np.float64]
    weights: NDArray[np.float64]
    condition: float

    @property
    def matrix(self) -> NDArray[np.float64]:
        """The 2x2 gradient ``A``, whose entries are per-pixel rates."""
        return np.array([[self.au[0], self.au[1]], [self.aw[0], self.aw[1]]])

    @property
    def offset(self) -> NDArray[np.float64]:
        """Flow at `centroid`, the constant term of both component solutions."""
        return np.array([self.au[2], self.aw[2]])

    def residuals(self, pts: NDArray[np.float64], flow: NDArray[np.float64]) -> NDArray[np.float64]:
        """Flow minus this model, evaluated at arbitrary points, as ``(N, 2)`` velocities."""
        design = _design_matrix(pts, self.centroid)
        return np.column_stack([flow[:, 0] - design @ self.au, flow[:, 1] - design @ self.aw])


def _design_matrix(pts: NDArray[np.float64], centroid: NDArray[np.float64]) -> NDArray[np.float64]:
    """Build the ``(N, 3)`` design matrix ``[x - cx, y - cy, 1]``."""
    return np.column_stack([pts[:, 0] - centroid[0], pts[:, 1] - centroid[1], np.ones(len(pts))])


def _biweight_weights(residuals: NDArray[np.float64], cutoff: float) -> NDArray[np.float64]:
    """Tukey biweight weights for residual magnitudes, zero at or beyond `cutoff`."""
    z = residuals / max(cutoff, 1e-12)
    weights = (1.0 - z * z) ** 2
    weights[np.abs(z) >= 1.0] = 0.0
    return weights


def _robust_cutoff(residuals: NDArray[np.float64]) -> float:
    """Biweight cutoff from the residuals' own scale, floored away from zero."""
    return _TUKEY_C * max(_MAD_TO_SIGMA * float(np.median(residuals)), 1e-6)


def _seed_weights(flow: NDArray[np.float64]) -> NDArray[np.float64]:
    """Weights rejecting movers before the first affine solve.

    The failure these guard against is a static camera with one large coherent mover.
    Such a mover inflates the residual scale enough that the biweight cutoff can no longer
    separate it, and the fit locks onto the mover instead of the background. Seeding from
    the component-wise median translation puts the fit in the majority basin first. For a
    static camera the median is near zero, so the deviation scale is small, the mover sits
    far outside it, and the mover is zeroed before it can influence the fit.

    On a clean field the seeding costs nothing. An exact affine field has zero residual at
    the truth under any weighting, so the recovered parameters are unchanged.
    """
    deviation = np.hypot(flow[:, 0] - np.median(flow[:, 0]), flow[:, 1] - np.median(flow[:, 1]))
    weights = _biweight_weights(deviation, _robust_cutoff(deviation))
    # A mover holding an outright majority flips the median onto itself, leaving the seed
    # with too few weighted samples. Fall back to an unweighted start. The trust gate
    # catches the resulting ambiguity.
    return weights if weights.sum() >= 3 else np.ones(len(flow))


def _fit_irls(
    pts: NDArray[np.float64],
    flow: NDArray[np.float64],
    n_iter: int = 4,
) -> _AffineFit | None:
    """Fit an affine flow field by iteratively reweighted least squares.

    Returns None where the sample geometry is singular enough that the normal equations
    cannot be solved.
    """
    centroid = pts.mean(axis=0)
    design = _design_matrix(pts, centroid)
    u, w = flow[:, 0], flow[:, 1]

    weights = _seed_weights(flow)
    solution: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
    previous: NDArray[np.float64] | None = None
    condition = 0.0

    for _ in range(n_iter):
        weighted = design.T * weights
        normal = weighted @ design
        try:
            # astype rather than a cast. The inputs are float64, so this is a no-op at
            # runtime, but `solve` is typed as widening to `floating[Any]`.
            au = np.linalg.solve(normal, weighted @ u).astype(np.float64, copy=False)
            aw = np.linalg.solve(normal, weighted @ w).astype(np.float64, copy=False)
        except np.linalg.LinAlgError:
            break
        solution = (au, aw)
        singular = np.linalg.svd(normal, compute_uv=False)
        condition = float(singular[-1] / singular[0]) if singular[0] > 0 else 0.0

        magnitude = np.hypot(u - design @ au, w - design @ aw)
        weights = _biweight_weights(magnitude, _robust_cutoff(magnitude))

        current = np.concatenate([au, aw]).astype(np.float64, copy=False)
        converged = previous is not None and bool(np.max(np.abs(current - previous)) < 1e-4)
        previous = current
        if converged:
            break

    if solution is None:
        return None
    return _AffineFit(solution[0], solution[1], centroid, weights, condition)


# ============================== per-pair ego motion ==============================


class _PairReading(NamedTuple):
    """What one frame pair yields about the camera.

    Only the scalars the per-frame rows report survive the fit. The flow samples and the
    camera/mover split they were read off are not kept.

    Attributes
    ----------
    div, curl, shear : float
        Camera rates per frame, resolution-free.
    t0x, t0y, t0_mag : float
        Pan/tilt at the image center, full-resolution pixels per frame.
    foe_x, foe_y : float
        Focus of expansion relative to the image center, full-resolution pixels. NaN when
        the field has no in-frame zero, which includes every pure pan.
    motion : str
        Which single motion dominates, or ``"complex"`` when none does.
    degenerate, border_capture, ambiguous : bool
        Trust flags. See :func:`ego_stats` for what each refuses.
    mover_frac : float
        Fraction of samples the camera model does not explain.
    mover_speed : float
        Median residual speed over those samples, full-resolution pixels per frame. NaN
        when there are none.
    """

    div: float
    curl: float
    shear: float
    t0x: float
    t0y: float
    t0_mag: float
    foe_x: float
    foe_y: float
    motion: str
    degenerate: bool
    border_capture: bool
    ambiguous: bool
    mover_frac: float
    mover_speed: float

    @property
    def foe_in_frame(self) -> bool:
        """Whether the field has a usable zero, meaning a direction the camera is heading."""
        return not np.isnan(self.foe_x)

    @property
    def trusted(self) -> bool:
        """Whether this pair's camera estimate can be told apart from a mover's."""
        return not (self.ambiguous or self.border_capture or self.degenerate)


def _refit_on_border(
    fit: _AffineFit,
    pts: NDArray[np.float64],
    flow: NDArray[np.float64],
    proc_wh: tuple[int, int],
    margin_frac: float,
    capture_thresh: float,
) -> tuple[_AffineFit, bool]:
    """Re-seed the fit from the frame border when a central mover appears to have captured it.

    A mover filling the middle of the frame can win the fit outright. The true background
    is then mostly what lies near the edges, and it is rejected as outlying. The signature
    is border samples mostly excluded while interior samples are mostly kept. Refitting on
    the border alone breaks the tie, and whichever fit explains more of the whole frame is
    kept.
    """
    width, height = proc_wh
    margin = margin_frac * min(width, height)
    on_border = (
        (pts[:, 0] < margin) | (pts[:, 0] > width - margin) | (pts[:, 1] < margin) | (pts[:, 1] > height - margin)
    )
    if on_border.sum() < _MIN_SAMPLES or (~on_border).sum() < _MIN_SAMPLES:
        return fit, False

    inlier = fit.weights > _INLIER_WEIGHT
    border_kept, interior_kept = inlier[on_border].mean(), inlier[~on_border].mean()
    if border_kept >= capture_thresh or interior_kept <= border_kept + 0.25:
        return fit, False

    candidate = _fit_irls(pts[on_border], flow[on_border])
    if candidate is None or _explained(candidate, pts, flow) <= _explained(fit, pts, flow):
        return fit, False
    return candidate, True


def _explained(fit: _AffineFit, pts: NDArray[np.float64], flow: NDArray[np.float64]) -> int:
    """How many of all samples this fit brings inside its own robust residual cutoff."""
    magnitude = np.hypot(*fit.residuals(pts, flow).T)
    return int((magnitude < _robust_cutoff(magnitude)).sum())


def _focus_of_expansion(
    fit: _AffineFit,
    img_center: NDArray[np.float64],
    scale: float,
    half_diagonal: float,
    limit_mult: float,
) -> tuple[float, float]:
    """Locate the zero of the flow field relative to the image center, or NaN if there is none.

    Only the divergent part of the field has a focus. A pure roll is a rotation about a
    point, and its field does have a zero, but that point is where the camera pivots, and
    the focus is meant to report where the camera is heading. Reporting the pivot would put
    a spurious focus on every rolling sequence. The gate is therefore on the trace, which a
    roll leaves at zero, rather than on the whole gradient, whose singular values a roll
    makes large.
    """
    matrix = fit.matrix
    divergence = abs(matrix[0, 0] + matrix[1, 1])
    if divergence <= 1e-4 or np.linalg.svd(matrix, compute_uv=False)[-1] <= 1e-4:
        return np.nan, np.nan
    try:
        zero = np.linalg.solve(matrix, -fit.offset) + fit.centroid
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    offset = (zero - img_center) * scale
    # A focus far outside the frame only means the pan had a slight divergence in it, so
    # it is no use as a heading.
    if np.max(np.abs(offset)) >= limit_mult * half_diagonal:
        return np.nan, np.nan
    return float(offset[0]), float(offset[1])


def _classify_motion(div: float, curl: float, t0_mag: float, half_diagonal: float, static_px: float) -> str:
    """Name the dominant motion by comparing every component at the frame corner.

    ``div`` and ``curl`` are rates and ``t0_mag`` is a speed, so they are not comparable
    until the rates are given a lever arm. That lever arm is the half-diagonal, which turns
    each rate into the pixel speed it produces at the corner of the frame. A component is
    named only if it is clearly ahead of the runner-up, and otherwise the motion counts
    as mixed.
    """
    speeds = {
        "zoom_dolly": abs(div) * half_diagonal,
        "roll": abs(curl) * half_diagonal,
        "pan_tilt": t0_mag,
    }
    if max(speeds.values()) < static_px:
        return "static"
    ranked = sorted(speeds.values())
    return "complex" if ranked[-1] < 1.4 * ranked[-2] else max(speeds, key=lambda k: speeds[k])


def _is_ambiguous(
    pts: NDArray[np.float64],
    flow: NDArray[np.float64],
    outlier: NDArray[np.bool_],
) -> bool:
    """Whether the rejected samples form a coherent motion of their own.

    A single frame pair cannot always tell the camera from a mover. If a mover fills much
    of the frame, a still camera with a moving object and a moving camera with a still
    object produce the same flow. What tells a genuine mover from that tie is whether the
    rejected samples agree among themselves. When they do, the pair has two consistent
    readings and no way to choose between them, so the pair is refused.
    """
    if outlier.mean() <= 0.30 or int(outlier.sum()) < _MIN_SAMPLES:
        return False
    second = _fit_irls(pts[outlier], flow[outlier])
    return second is not None and bool((second.weights > _INLIER_WEIGHT).mean() > 0.60)


def _pair_ego(
    pts: NDArray[np.float64],
    flow: NDArray[np.float64],
    proc_wh: tuple[int, int],
    full_wh: tuple[int, int],
    gap: int,
    margin_frac: float = 0.12,
    capture_thresh: float = 0.45,
    foe_limit_mult: float = 8.0,
    static_px: float = 0.2,
) -> _PairReading | None:
    """Fit the camera motion for one frame pair, or None where no fit is possible."""
    proc_w, proc_h = proc_wh
    full_w, full_h = full_wh
    scale = 0.5 * (full_w / proc_w + full_h / proc_h)
    img_center = np.array([proc_w / 2.0, proc_h / 2.0])
    half_diagonal = 0.5 * float(np.hypot(full_w, full_h))

    fit = _fit_irls(pts, flow)
    if fit is None:
        return None
    fit, border_capture = _refit_on_border(fit, pts, flow, proc_wh, margin_frac, capture_thresh)

    matrix = fit.matrix
    div = (matrix[0, 0] + matrix[1, 1]) / gap
    curl = (matrix[1, 0] - matrix[0, 1]) / gap
    shear = float(np.hypot((matrix[0, 0] - matrix[1, 1]) / gap, (matrix[0, 1] + matrix[1, 0]) / gap))
    t0 = (matrix @ (img_center - fit.centroid) + fit.offset) * scale / gap
    t0_mag = float(np.hypot(t0[0], t0[1]))
    foe_x, foe_y = _focus_of_expansion(fit, img_center, scale, half_diagonal, foe_limit_mult)

    resid = fit.residuals(pts, flow)
    speed = np.hypot(resid[:, 0], resid[:, 1]) * scale / gap
    outlier = _biweight_weights(speed, _robust_cutoff(speed)) <= _INLIER_WEIGHT

    return _PairReading(
        div=div,
        curl=curl,
        shear=shear,
        t0x=float(t0[0]),
        t0y=float(t0[1]),
        t0_mag=t0_mag,
        foe_x=foe_x,
        foe_y=foe_y,
        motion=_classify_motion(div, curl, t0_mag, half_diagonal, static_px),
        degenerate=fit.condition < 1e-12,
        border_capture=border_capture,
        ambiguous=_is_ambiguous(pts, flow, outlier),
        mover_frac=float(outlier.mean()),
        mover_speed=float(np.median(speed[outlier])) if outlier.any() else np.nan,
    )


# ============================== optical flow ==============================


def _lattice(shape: tuple[int, ...], grid_target: int) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Column and row indices of a roughly ``grid_target``-square sampling lattice."""
    height, width = shape[:2]
    step = max(1, max(width, height) // grid_target)
    gx, gy = np.meshgrid(np.arange(step // 2, width, step), np.arange(step // 2, height, step))
    return gx.ravel(), gy.ravel()


def _grid_flow_dis(
    dis: Any,
    previous: NDArray[np.uint8],
    current: NDArray[np.uint8],
    grid_target: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Dense DIS flow, subsampled onto the lattice."""
    dense = dis.calc(previous, current, None)
    gx, gy = _lattice(previous.shape, grid_target)
    return np.column_stack([gx, gy]).astype(float), dense[gy, gx].astype(float)


def _grid_flow_lk(
    previous: NDArray[np.uint8],
    current: NDArray[np.uint8],
    grid_target: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sparse Lucas-Kanade flow on the lattice, dropping points it could not track."""
    cv2 = _require_cv2()
    gx, gy = _lattice(previous.shape, grid_target)
    start = np.column_stack([gx, gy]).astype(np.float32).reshape(-1, 1, 2)
    end, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, start, None, winSize=(21, 21), maxLevel=3)
    tracked = status.ravel().astype(bool)
    pts = start[tracked].reshape(-1, 2).astype(float)
    return pts, (end[tracked].reshape(-1, 2) - pts).astype(float)


# ============================== sequence walk ==============================


def _downscale(gray: NDArray[np.uint8], proc_short_side: int) -> NDArray[np.uint8]:
    """Shrink a 2D grayscale frame to `proc_short_side` on its short side, never enlarging it."""
    cv2 = _require_cv2()
    height, width = gray.shape[:2]
    short = min(width, height)
    if short <= proc_short_side:
        return gray
    factor = proc_short_side / short
    return cv2.resize(gray, (round(width * factor), round(height * factor)), interpolation=cv2.INTER_AREA)


_T = TypeVar("_T")


class _PairBuffer(Generic[_T]):
    """Hold the last `gap` + 1 frames, so each new frame can be paired with the one `gap` back.

    Forward streaming only, so a long sequence never has to be decoded into memory at once.
    """

    def __init__(self, gap: int) -> None:
        self._gap = gap
        self._frames: deque[_T] = deque(maxlen=gap + 1)

    def push(self, frame: _T) -> _T | None:
        """Take in the next frame and return the frame `gap` before it, or None until one is held."""
        self._frames.append(frame)
        return self._frames[0] if len(self._frames) > self._gap else None


def _stream_pairs(
    frames: Iterable[NDArray[np.uint8]],
    gap: int,
) -> Iterator[tuple[NDArray[np.uint8], NDArray[np.uint8]]]:
    """Yield every frame pair `gap` apart, holding at most `gap` + 1 frames."""
    buffer: _PairBuffer[NDArray[np.uint8]] = _PairBuffer(gap)
    for frame in frames:
        previous = buffer.push(frame)
        if previous is not None:
            yield previous, frame


def _trimmed_center(state: NDArray[np.float64]) -> NDArray[np.float64]:
    """Find the center of the dominant cluster, unmoved by a burst of any width.

    The median is global and iteratively trimmed. A short rolling window would track a
    wide burst and stop seeing it as a departure, while trimming against the whole sequence
    holds the baseline wherever the quiet pairs are still a plurality.
    """
    center = np.median(state, axis=0)
    for _ in range(3):
        distance = np.linalg.norm(state - center, axis=1)
        keep = distance < 4.0 * (_MAD_TO_SIGMA * np.median(distance) + 1e-6)
        if keep.sum() < 4:
            break
        center = np.median(state[keep], axis=0)
    return center


def _temporal_outliers(
    readings: Sequence[_PairReading | None],
    half_diagonal: float,
    scales: Sequence[float],
    k: float = 5.0,
) -> NDArray[np.bool_]:
    """Flag pairs whose camera state departs from what the sequence mostly agrees on.

    The camera is whatever most pairs agree about, so a burst of disagreement marks a burst
    of capture by something else. Two of the components are rates and two are speeds, which
    are not otherwise comparable, so all four are put in common units before distances are
    taken. Those units are the first frame's pixels per frame at the frame corner. `scales`
    holds the factor that takes each pair's speeds from its own pixels into those, which is
    not 1 only where the frame size changed mid-sequence.

    The test assumes a roughly steady camera state across the sequence. Time-varying
    camera motion would need a long window instead.
    """
    flags = np.zeros(len(readings), bool)
    state = np.full((len(readings), 4), np.nan)
    for i, pair in enumerate(readings):
        if pair is not None:
            lever = np.array([scales[i], scales[i], half_diagonal, half_diagonal])
            state[i] = np.array([pair.t0x, pair.t0y, pair.div, pair.curl]) * lever

    measured = ~np.isnan(state).any(axis=1)
    if measured.sum() < 8:
        return flags

    distance = np.linalg.norm(state - _trimmed_center(state[measured]), axis=1)
    # Floored at two pixels per frame. Below that the camera is effectively still and the
    # spread is estimator noise, which would otherwise make every pair an outlier.
    sigma = max(_MAD_TO_SIGMA * float(np.median(distance[measured])), 2.0)
    flags[measured] = distance[measured] > k * sigma
    return flags


# ============================== per-frame rows ==============================


class EgoFactors(TypedDict):
    """Per-frame camera measurements, one entry per frame of every sequence measured.

    Rows are concatenated in dataset-item order. ``item_index`` and ``unit_index`` together
    name the frame each value belongs to, which lets the result be attached with
    ``add_factors(..., level="unit", key="unit_index")``.

    A reading describes the motion arriving at its frame, measured against the frame
    ``gap`` earlier, so the leading ``gap`` frames of every sequence are NaN.

    The kinematics are NaN wherever ``ego_trusted`` is False. A frame whose camera cannot be
    told apart from a mover abstains rather than contributing a wrong number, and a roll-up's
    coverage threshold gates on those abstentions. ``mover_frac``, ``mover_speed`` and
    ``ego_trusted`` are not nulled. Those three are most informative on a pair that was
    refused because a mover dominated it.

    Attributes
    ----------
    item_index : Sequence[int]
        Dataset item the frame belongs to.
    unit_index : Sequence[int]
        Frame's position within its own sequence, zero-based.
    pan_x, pan_y : Sequence[float]
        Signed pan/tilt at the image center, full-resolution pixels per frame.
    pan_speed : Sequence[float]
        Magnitude of the above.
    zoom_rate : Sequence[float]
        Signed divergence per frame. Positive is looming. Resolution-free.
    roll_rate : Sequence[float]
        Signed curl per frame, twice the rotation rate in radians. Resolution-free.
    shear_rate : Sequence[float]
        Magnitude of the traceless symmetric part, per frame. Resolution-free.
    foe_x, foe_y : Sequence[float]
        Focus of expansion relative to the image center, full-resolution pixels. NaN where
        the field has no in-frame zero, which includes every pure pan and every pure roll.
    motion : Sequence[str]
        Dominant motion for this frame: ``"static"``, ``"pan_tilt"``, ``"zoom_dolly"``,
        ``"roll"``, or ``"complex"`` where no component leads.
    mover_frac : Sequence[float]
        Fraction of flow samples the camera model does not explain.
    mover_speed : Sequence[float]
        Median speed of those samples, full-resolution pixels per frame. NaN where none.
    ego_trusted : Sequence[bool]
        Whether this frame's camera estimate could be told apart from a mover's. False as well
        where the pair could not be read: the two frames differ in size, a processed side is under
        32 px, a frame holds a non-finite pixel, or a frame is a single gray level once converted to
        grayscale, as a black dropout or a frame whose values all saturate the conversion is.
    """

    item_index: Sequence[int]
    unit_index: Sequence[int]
    pan_x: Sequence[float]
    pan_y: Sequence[float]
    pan_speed: Sequence[float]
    zoom_rate: Sequence[float]
    roll_rate: Sequence[float]
    shear_rate: Sequence[float]
    foe_x: Sequence[float]
    foe_y: Sequence[float]
    motion: Sequence[str]
    mover_frac: Sequence[float]
    mover_speed: Sequence[float]
    ego_trusted: Sequence[bool]


EgoStatsResult: TypeAlias = FactorResult[EgoFactors]
"""Per-frame camera ego-motion. Values under ``stats``. See :class:`EgoFactors`."""


#: Share of a sequence's frames that must carry a trusted reading for a roll-up to answer.
#: Below this the sequence abstains, which is the sequence-level form of refusing a pair.
_TRUST_COVERAGE = 0.25


def _roll(how: str, *factors: str, suffix: str | None = None) -> Aggregator:
    """One unit-to-sequence declaration at the shared trust threshold."""
    return Aggregator(how, "unit", "sequence", factors, min_coverage=_TRUST_COVERAGE, suffix=suffix)


#: How per-frame camera measurements summarize over a whole sequence.
#:
#: Declared beside the measurement because the producer is what knows the reductions: that
#: a rate accumulates as a total rather than averaging, that a motion label reduces by
#: plurality, and how much of a sequence must be trustworthy before a summary means
#: anything.
#:
#: Names come from the reduction, which keeps the operation legible in the column. The one
#: exception is ``abs_sum``, which reads as an implementation detail rather than a
#: measurement, so those two columns carry an ``excursion`` suffix instead.
EGO_AGGREGATIONS: tuple[Aggregator, ...] = (
    _roll("median", "pan_speed", "zoom_rate", "roll_rate", "shear_rate", "mover_frac", "mover_speed"),
    _roll("variability", "pan_x", "pan_y"),
    _roll("abs_sum", "zoom_rate", "roll_rate", suffix="_excursion"),
    _roll("mode", "motion"),
)


def _frame_rows(
    readings: Sequence[_PairReading | None],
    scales: Sequence[float],
    n_frames: int,
    gap: int,
    half_diagonal: float,
) -> dict[str, list[Any]]:
    """Lay one sequence's pair readings out as one row per frame.

    A pair spans ``(k, k + gap)`` and is attributed to the later frame. At a scene cut the
    pair straddling the cut produces a meaningless reading. Attributing it to the first
    frame of the new shot is accurate, because that frame is discontinuous from its
    predecessor. Attributing it to the last frame of the old shot would not be.

    `scales` only puts each pair in the first frame's units for the temporal gate. The rows
    report every pixel quantity in the pixels of the pair it was measured on.
    """
    temporal = _temporal_outliers(readings, half_diagonal, scales)
    rows: dict[str, list[Any]] = {name: [] for name in EgoFactors.__annotations__ if name != "item_index"}
    for frame in range(n_frames):
        pair = readings[frame - gap] if frame >= gap else None
        trusted = pair is not None and pair.trusted and not temporal[frame - gap]
        rows["unit_index"].append(frame)
        rows["ego_trusted"].append(bool(trusted))
        rows["mover_frac"].append(pair.mover_frac if pair is not None else np.nan)
        rows["mover_speed"].append(pair.mover_speed if pair is not None else np.nan)
        rows["motion"].append(pair.motion if trusted and pair else None)
        for name, value in (
            ("pan_x", pair.t0x if pair else np.nan),
            ("pan_y", pair.t0y if pair else np.nan),
            ("pan_speed", pair.t0_mag if pair else np.nan),
            ("zoom_rate", pair.div if pair else np.nan),
            ("roll_rate", pair.curl if pair else np.nan),
            ("shear_rate", pair.shear if pair else np.nan),
            ("foe_x", pair.foe_x if pair else np.nan),
            ("foe_y", pair.foe_y if pair else np.nan),
        ):
            rows[name].append(value if trusted else np.nan)
    return rows


# ============================== public API ==============================


class _FlowFrame(NamedTuple):
    """One frame as a pair reads it: the downscaled grayscale and the original size it came from.

    Two frames can share a processed shape but not an original size (640x360 and 1280x720 both
    process to 640x360), so a pair compares both.
    """

    gray: NDArray[np.uint8]
    full_wh: tuple[int, int]


def _flow_frame(view: FrameView) -> _FlowFrame | None:
    """Take a frame as a pair reads it, or None where it holds a non-finite pixel.

    The check runs before the grayscale conversion, which would turn NaN into a flat field that
    reads as a still camera.
    """
    if np.issubdtype(view.pixels.dtype, np.inexact) and not np.isfinite(view.pixels).all():
        return None
    # MAITE declares frame pixels as (C, H, W), which is where the trailing axes come from.
    return _FlowFrame(view.gray_small, (int(view.pixels.shape[-1]), int(view.pixels.shape[-2])))


def _flow_estimator(flow: FlowMethod) -> Any | None:
    """Build the DIS estimator for a video, or None where Lucas-Kanade needs none."""
    cv2 = _require_cv2()
    return cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST) if flow == "dis" else None


def _measure_pair(
    estimator: Any | None,
    previous: _FlowFrame,
    current: _FlowFrame,
    grid_target: int,
    gap: int,
) -> _PairReading | None:
    """Read the camera motion between two frames, or None where the pair cannot be read.

    A pair is refused before any flow is computed when the frames differ in original size or in
    processed shape, when a processed side is under `_MIN_FLOW_SIDE`, or when either frame is a
    single gray level, which holds no motion to read and would read as a still camera. Pixel
    quantities are scaled by the pair's own full resolution.
    """
    height, width = previous.gray.shape[:2]
    if previous.full_wh != current.full_wh or previous.gray.shape != current.gray.shape:
        return None
    if min(width, height) < _MIN_FLOW_SIDE:
        return None
    if np.ptp(previous.gray) == 0 or np.ptp(current.gray) == 0:
        return None
    pts, vectors = (
        _grid_flow_dis(estimator, previous.gray, current.gray, grid_target)
        if estimator is not None
        else _grid_flow_lk(previous.gray, current.gray, grid_target)
    )
    return _pair_ego(pts, vectors, (width, height), current.full_wh, gap) if len(pts) >= _MIN_SAMPLES else None


class _EgoEngine:
    """Measure one video's camera ego-motion as a frame consumer, one frame pair at a time.

    Holds one DIS estimator for the video, the frames still waiting for their pair, and one
    reading per pair, so a long video never has to be decoded into memory at once. Every
    reported pixel quantity is scaled back to the original size of the pair it was measured on.
    """

    def __init__(self, options: VideoOptions) -> None:
        self._options = options
        self._estimator = _flow_estimator(options.ego_flow)
        self._pairs: _PairBuffer[_FlowFrame | None] = _PairBuffer(options.ego_gap)
        # Kept None where a pair could not be measured, so refusals count against the pairs attempted.
        self._readings: list[_PairReading | None] = []
        # Per pair, the first frame's diagonal over the pair's, so the temporal gate reads one unit.
        self._scales: list[float] = []
        self._full_wh: tuple[int, int] | None = None
        self._n_frames = 0

    def update(self, view: FrameView) -> None:
        """Measure the pair the frame closes, if any."""
        full_wh = (int(view.pixels.shape[-1]), int(view.pixels.shape[-2]))
        if self._full_wh is None:
            self._full_wh = full_wh
        self._n_frames += 1
        current = _flow_frame(view)
        previous = self._pairs.push(current)
        gap = self._options.ego_gap
        if self._n_frames <= gap:
            return
        # An unreadable frame refuses both pairs it is in.
        reading = (
            None
            if previous is None or current is None
            else _measure_pair(self._estimator, previous, current, self._options.ego_grid_target, gap)
        )
        self._readings.append(reading)
        # A measured pair shares one original size, which is this frame's.
        self._scales.append(float(np.hypot(*self._full_wh) / np.hypot(*full_wh)))

    def finish(self, clock: VideoClock) -> None:
        """Nothing to close. The temporal gate needs the whole video, so it runs in :meth:`frame_rows`."""

    def frame_rows(self) -> dict[str, list[Any]]:
        """Lay the video's pair readings out as one row per frame."""
        half_diagonal = 0.0 if self._full_wh is None else 0.5 * float(np.hypot(*self._full_wh))
        return _frame_rows(self._readings, self._scales, self._n_frames, self._options.ego_gap, half_diagonal)


def _measure_video(options: VideoOptions, max_frames: int | None, stream: VideoStream) -> dict[str, list[Any]]:
    """Walk one video through an ego engine and lay its readings out as one row per frame."""
    engine = _EgoEngine(options)
    walk(stream, [engine], options, stop_after=max_frames)
    return engine.frame_rows()


@overload
def ego_stats(
    source: VideoStream,
    gap: int = 1,
    proc_short_side: int = 360,
    flow: FlowMethod = "dis",
    grid_target: int = 40,
    max_frames: int | None = None,
) -> EgoStatsResult: ...
@overload
def ego_stats(
    source: Sequence[VideoStream],
    gap: int = 1,
    proc_short_side: int = 360,
    flow: FlowMethod = "dis",
    grid_target: int = 40,
    max_frames: int | None = None,
) -> EgoStatsResult: ...
@overload
def ego_stats(
    source: MultiobjectTrackingDataset,
    gap: int = 1,
    proc_short_side: int = 360,
    flow: FlowMethod = "dis",
    grid_target: int = 40,
    max_frames: int | None = None,
) -> EgoStatsResult: ...


def ego_stats(
    source: VideoStream | Sequence[VideoStream] | MultiobjectTrackingDataset,
    gap: int = 1,
    proc_short_side: int = 360,
    flow: FlowMethod = "dis",
    grid_target: int = 40,
    max_frames: int | None = None,
) -> EgoStatsResult:
    """Measure camera ego-motion for one video sequence or a whole tracking dataset.

    Camera motion is a shortcut hazard. A model can learn "the camera was panning" in place
    of the object whenever the two correlate. These statistics measure the camera so an
    audit can condition on it. They are reported on three deliberately separate axes: what
    the camera did, how much independent motion contaminated the estimate, and whether the
    estimate can be trusted.

    Motion is estimated from cheap optical flow fitted to an affine field by iteratively
    reweighted least squares. The robust weighting rejects independent movers instead of
    modeling them. Pairs where the camera cannot be told apart from a mover are refused
    outright. See `ego_trusted`.

    Parameters
    ----------
    source : VideoStream, Sequence[VideoStream] or MultiobjectTrackingDataset
        One sequence's frames (item 0), a sequence of streams (one video each), or a
        tracking dataset of them. Rows for every sequence are concatenated in item order.
    gap : int, default 1
        Frames between the two frames of each measured pair. Larger values give a bigger
        baseline, which helps on slow motion that a single frame step leaves in the noise.
        The cost is more independent movement inside the pair. Every rate is divided by
        `gap`, so the reported units are per frame either way. A reading is attributed to
        the later frame of its pair, so the leading `gap` frames of every sequence are
        NaN.
    proc_short_side : int, default 360
        Frames are downscaled to this short side before flow is computed. Rates are
        resolution-free and speeds are scaled back up, so each speed is in the full-resolution
        pixels of the pair it was measured on regardless.
    flow : {"dis", "lk"}, default "dis"
        Optical flow estimator: dense DIS at its ultrafast preset, or sparse Lucas-Kanade
        on the sampling lattice. DIS is more robust on weak texture. LK is cheaper and
        reports which points it failed to track.
    grid_target : int, default 40
        Roughly how many flow samples to take along the longer frame axis. Raising it
        gives a finer picture of what moved independently, at quadratic cost.
    max_frames : int or None, default None
        Stop after this many frames of each sequence. None measures all of them.

    Returns
    -------
    EgoStatsResult
        One row per frame of every sequence measured. See :class:`EgoFactors` for the
        fields. Also carries 11 declared roll-ups to the sequence level, which
        :meth:`~dataeval.Metadata.add_factors` applies by default (see Notes below).

    Raises
    ------
    ImportError
        If OpenCV is not installed. Install it with the ``dataeval[opencv]`` extra.
    ValueError
        If `gap`, `proc_short_side`, `grid_target` or `max_frames` is not positive, if
        `proc_short_side` is under 32 px, or if `flow` is not one of the supported estimators.

    See Also
    --------
    :func:`~dataeval.core.video_stats` : With ``stats=VideoStats.EGO``, one row of these
        summaries per video

    Notes
    -----
    **Reading the result.** Check `ego_trusted` first. Where it is False, the kinematic
    fields such as `pan_x` are NaN, because the camera could not be told apart from a mover
    on that frame. `mover_frac` and `mover_speed` stay populated regardless. They are most
    informative on the frames the ego fit refuses.

    **Sequence-level trusted fraction.** The declared roll-ups (see `add_factors`) do not
    include one for `ego_trusted` itself. The named-reduction registry's ``mean`` accepts
    numeric columns only, and `ego_trusted` is Boolean. Get the fraction with an explicit
    aggregation instead, aliased so the output does not collide with the source column:

    >>> metadata.agg("unit", "sequence", pl.col("ego_trusted").mean().alias("trusted_frac"))  # doctest: +SKIP

    **Units.** ``zoom_rate``, ``roll_rate`` and ``shear_rate`` are per-frame rates and do
    not change with resolution, so they are comparable across differently sized sequences.
    ``pan_speed`` and ``mover_speed`` are pixels per frame in the full resolution of the pair
    each was measured on, so they are not comparable that way, even within a sequence whose
    frame size changes. Normalize them by frame size before comparing sequences shot at
    different resolutions.

    **Cost.** Dominated by optical flow, which is linear in frames measured and roughly
    linear in pixels at `proc_short_side`. Lower `proc_short_side` before lowering
    `grid_target`.

    **Parallelism.** Sequences are measured concurrently, one per worker thread, when
    :func:`~dataeval.config.set_max_processes` allows more than one worker. Each worker
    fetches and decodes its own sequence, so a dataset whose items share one open decoder is
    not safe to measure this way. Set ``max_processes`` to 1 for such a dataset.

    Examples
    --------
    Measure a dataset and attach the result as unit-level metadata:

    >>> from dataeval.core import ego_stats
    >>> result = ego_stats(tracking_dataset)  # doctest: +SKIP
    >>> metadata.add_factors(result, level="unit", key="unit_index")  # doctest: +SKIP

    Then condition on the camera when auditing, for instance by splitting on `motion` or
    filtering to `ego_trusted`.
    """
    _reject_bad_params(gap, proc_short_side, flow, grid_target, max_frames)
    # Imported here, on the caller's thread, because run_videos caps OpenCV's threads only
    # when cv2 is already imported, and the workers would otherwise import it first.
    _require_cv2()
    options = VideoOptions(ego_gap=gap, ego_proc_short_side=proc_short_side, ego_flow=flow, ego_grid_target=grid_target)
    item_rows = run_videos(source, partial(_measure_video, options, max_frames))
    _logger.info("Ego stats complete for %d video(s).", len(item_rows))
    return EgoStatsResult(stats=_concat_items(item_rows), aggregations=EGO_AGGREGATIONS)


def _reject_bad_params(
    gap: int,
    proc_short_side: int,
    flow: str,
    grid_target: int,
    max_frames: int | None,
) -> None:
    """Reject parameters that cannot produce a meaningful measurement."""
    positive = {"gap": gap, "proc_short_side": proc_short_side, "grid_target": grid_target}
    if max_frames is not None:
        positive["max_frames"] = max_frames
    bad = {name: value for name, value in positive.items() if value < 1}
    if bad:
        raise ValueError(f"Parameters must be positive integers; got {bad}.")
    _check_proc_short_side(proc_short_side, "proc_short_side")
    if flow not in ("dis", "lk"):
        raise ValueError(f"Unknown flow estimator {flow!r}; expected one of 'dis', 'lk'.")


def _check_proc_short_side(value: int, name: str) -> None:
    """Refuse a processing size under the flow floor, naming the caller's keyword, as every pair would be refused."""
    if value < _MIN_FLOW_SIDE:
        raise ValueError(
            f"{name} must be at least {_MIN_FLOW_SIDE} px, the smallest side optical flow reads; got {value}.",
        )


def _concat_items(item_rows: Sequence[dict[str, list[Any]]]) -> EgoFactors:
    """Concatenate per-sequence frame rows in dataset-item order, adding ``item_index``.

    Column names come from the result type rather than from whatever the walk produced, so
    an empty dataset still answers with every field.
    """
    merged: dict[str, list[Any]] = {name: [] for name in EgoFactors.__annotations__}
    for item, rows in enumerate(item_rows):
        merged["item_index"].extend([item] * len(rows["unit_index"]))
        for name, values in rows.items():
            merged[name].extend(values)
    return cast("EgoFactors", merged)
