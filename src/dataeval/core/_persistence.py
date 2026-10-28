"""Per-video maps of what is fixed to the camera, behind ``VideoStats.PERSISTENCE``.

Two maps per video, on a grid of cells 48 rows high:

- **persistence** (``ego_map``): per pixel, how often it stayed unchanged in frame pairs where the
  camera's own motion moved it at least `MIN_SHIFT` px, over textured pixels; per cell, the share
  of its known pixels that persist. A pair counts only when one similarity fitted to what changed
  explains more of the frame than "nothing moved" does, so movers before a still camera count for
  nothing, and a pixel the camera barely moved, which stays put whatever it shows, is not evidence.
- **softness** (``blur_map_raw``, and ``blur_map`` once detrended): per cell, the log attenuation
  of sharpness the *same content* undergoes there. Each frame pair is aligned by a similarity
  transform, so content at cell x in the later frame came from src(x) in the earlier one, giving
  ``a(x) - a(src(x)) = log s_later(x) - log s_earlier(src(x))`` per cell per pair. A robust
  least-squares solve over every pair, with a little smoothness and a zero mean, gives a(x): a
  flat-field estimate applied to sharpness. Scene content cancels, because it is compared only
  with itself.

The constants are those the camera-maps spike validated; every one counted in cells assumes the
48-row grid, and every one counted in pixels assumes frames shrunk to a 360 px short side. A
smaller frame is not enlarged, as ego-motion does not enlarge one, so on such footage the pixel
constants act at a coarser scale than the spike validated.
"""

__all__ = []

import math
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, cast

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, diags, vstack
from scipy.sparse.linalg import lsqr

from dataeval.core._ego_stats import _downscale, _require_cv2
from dataeval.protocols import SharpnessMeasure
from dataeval.utils.preprocessing import edge_filter

ROWS = 48  # grid rows; columns follow the aspect ratio
REPORT_GRID = (48, 85)  # every map is reported on this grid, the solved grid of 16:9 video
PROC = 360  # short side, in pixels, for persistence and alignment
STEP, GAPS = 5, (10, 30)  # frame pairs (k, k + gap), from start frames every STEP frames
TEXTURE = 4.0  # 3x3 local standard deviation (gray levels) for a pixel to count as textured
MOVED_SHARE = 0.3  # share of textured pixels that must change before a pair's camera motion is read
CHANGE = 4.0  # absolute difference (gray levels) for a pixel to count as changed
MIN_SHIFT = 2.0  # px the camera must move a pixel in a pair for it to be evidence; slower, scenery looks fixed
STILL = 2.5  # ... and below which, or a fifth of the pair's median, it counts as unchanged
MIN_EVIDENCE = 5  # pairs (persistence) or equations (softness) a pixel or cell needs before its value is known
PERSISTENT = 0.7  # a pixel persists if unchanged in at least this share of its evidence pairs
FLAG_PERSISTENT = 0.1  # a cell is fixed to the camera when at least this share of its known pixels persist
SOURCE_FLOOR = 0.1  # an equation needs its source sharper than this share of the frame's median (not flat)
SMOOTH, HUBER = 0.1, 0.3  # the solve's neighbor-smoothness weight and robust residual scale (log units)
FLOOR = 1e-6  # sharpness floor before the log, so a flat cell's zero has a log


def solved_grid(height: int, width: int) -> tuple[int, int]:
    """Return the grid a video's maps are solved on: 48 rows, and columns that keep cells close to square."""
    return ROWS, round(ROWS * width / height)


def _cell_mean(x: NDArray[Any], grid: tuple[int, int]) -> NDArray[np.float32]:
    cv2 = _require_cv2()
    rows, cols = grid
    return cv2.resize(np.asarray(x, np.float32), (cols, rows), interpolation=cv2.INTER_AREA)


def _cell_std(x: NDArray[Any], grid: tuple[int, int]) -> NDArray[np.float32]:
    return np.sqrt(np.maximum(_cell_mean(x * x, grid) - _cell_mean(x, grid) ** 2, 0.0))


def edge_sharpness(gray: NDArray[np.float32], grid: tuple[int, int], /) -> NDArray[np.float32]:
    """Measure sharpness as the standard deviation of `edge_filter` per cell, as `image_stats` reduces it."""
    return _cell_std(edge_filter(gray), grid)


class Sample(NamedTuple):
    """What the maps keep of one frame.

    Attributes
    ----------
    proc : NDArray[np.uint8]
        The frame shrunk to a 360 px short side, for persistence and alignment.
    log_s : NDArray[np.float64] or None
        Log sharpness per cell, None when no blur map is wanted.
    log_median : float
        Log of the median cell sharpness, NaN when no blur map is wanted.
    """

    proc: NDArray[np.uint8]
    log_s: NDArray[np.float64] | None
    log_median: float


def _measured(measure: SharpnessMeasure, gray: NDArray[np.float32], grid: tuple[int, int]) -> NDArray[np.float64]:
    """Call `measure` and hold it to its contract, naming it when it breaks it."""
    values = np.asarray(measure(gray, grid), dtype=np.float64)
    name = getattr(measure, "__qualname__", None) or type(measure).__qualname__
    if values.shape != grid:
        raise ValueError(f"blur_measure {name} must return one value per cell, shape {grid}; got {values.shape}.")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"blur_measure {name} must return finite, non-negative values.")
    return values


def sample(gray: NDArray[np.uint8], grid: tuple[int, int], measure: SharpnessMeasure | None) -> Sample:
    """Keep what the maps need of one full-resolution grayscale frame."""
    # A copy, because the downscale returns a frame already at 360 px unchanged, and keeping
    # that would keep the decoded frame alive.
    proc = _downscale(gray, PROC).copy()
    if measure is None:
        return Sample(proc, None, math.nan)
    s = np.maximum(_measured(measure, gray.astype(np.float32), grid), FLOOR)
    return Sample(proc, np.log(s), float(np.log(np.median(s))))


class StartSampler:
    """Start frames every `STEP`-th frame, the stride doubling whenever keeping one more would exceed `ceiling`.

    The kept starts cover everything seen so far evenly, however long the video turns out to be,
    and never number more than `ceiling`.
    """

    def __init__(self, ceiling: int) -> None:
        self.ceiling = ceiling
        self.stride = STEP
        self.kept: list[int] = []
        self._kept: set[int] = set()

    def offer(self, index: int) -> bool:
        """Return whether frame `index` becomes a kept start, dropping earlier starts off a doubled stride."""
        if index % self.stride:
            return False
        while len(self.kept) >= self.ceiling:
            self.stride *= 2
            self.kept = [k for k in self.kept if k % self.stride == 0]
            self._kept = set(self.kept)
            if index % self.stride:
                return False
        self.kept.append(index)
        self._kept.add(index)
        return True

    def needs(self, index: int, *, blur: bool) -> bool:
        """Return whether frame `index` is a kept start or the partner of one."""
        return index in self._kept or index - GAPS[0] in self._kept or (blur and index - GAPS[1] in self._kept)


def detrend(cells: NDArray[np.float64], sigma: float) -> NDArray[np.float64]:
    """`cells` minus its NaN-aware Gaussian trend at `sigma` cells, NaN kept where it was."""
    cv2 = _require_cv2()
    known = np.isfinite(cells)
    filled = np.where(known, cells, 0.0).astype(np.float32)
    weight = cv2.GaussianBlur(known.astype(np.float32), (0, 0), sigma)
    trend = cv2.GaussianBlur(filled, (0, 0), sigma) / np.maximum(weight, 1e-6)
    return np.where(known, cells - trend, np.nan)


def report(cells: NDArray[np.float64]) -> NDArray[np.float64]:
    """Resample a solved map onto the report grid, NaN-aware, never by nearest cell.

    An area average where cells shrink and linear interpolation where they grow, each over the
    known cells only, so a thin defect is kept and a cell is NaN only where nothing it reads is
    known.
    """
    rows, cols = REPORT_GRID
    if cells.shape[1] == cols:
        return cells.astype(np.float64, copy=True)
    cv2 = _require_cv2()
    known = np.isfinite(cells)
    interpolation = cv2.INTER_AREA if cells.shape[1] > cols else cv2.INTER_LINEAR
    num = cv2.resize(np.where(known, cells, 0.0), (cols, rows), interpolation=interpolation)
    den = cv2.resize(known.astype(np.float64), (cols, rows), interpolation=interpolation)
    return np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)


def _why(*checks: tuple[bool, str]) -> str:
    """Return the reason of the first check that failed, in the order a map's evidence is gathered."""
    return next(reason for failed, reason in checks if failed)


def _local_std(x: NDArray[np.float32]) -> NDArray[np.float32]:
    cv2 = _require_cv2()
    m = cv2.blur(x, (3, 3))
    return np.sqrt(np.maximum(cv2.blur(x * x, (3, 3)) - m * m, 0.0))


def _evidence(
    earlier: NDArray[np.uint8], later: NDArray[np.uint8], d: NDArray[np.float32], tex: NDArray[np.bool_]
) -> NDArray[np.bool_] | None:
    """Return the pair's pixels that can show what is fixed to the camera, or None where none can.

    A textured pixel can when the camera moved it at least `MIN_SHIFT` px; one moved less stays put whatever it
    shows. The camera's motion is one similarity fitted to corners in what changed, read only when `MOVED_SHARE`
    of the textured pixels changed. It counts only when it explains more textured pixels than "nothing moved"
    does, each by a `CHANGE` margin: movers before a still camera fit a motion that misaligns the still
    background, so there "nothing moved" explains more.
    """
    if (d[tex] > CHANGE).mean() < MOVED_SHARE:
        return None
    cv2 = _require_cv2()
    near = cv2.dilate((tex & (d > CHANGE)).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    m = _similarity(earlier, later, ~near)
    if m is None:
        return None
    h, w = earlier.shape
    a = earlier.astype(np.float32)
    residual = np.abs(later.astype(np.float32) - cv2.warpAffine(a, m, (w, h)))
    t = tex & (cv2.warpAffine(np.ones_like(a), m, (w, h)) > 0.99)
    if (t & (residual + CHANGE < d)).sum() <= (t & (d + CHANGE < residual)).sum():
        return None
    yy, xx = np.mgrid[0:h, 0:w]
    shift = np.hypot(m[0, 0] * xx + m[0, 1] * yy + m[0, 2] - xx, m[1, 0] * xx + m[1, 1] * yy + m[1, 2] - yy)
    evidence = tex & (shift >= MIN_SHIFT)
    return evidence if evidence.any() else None


def _persistence_map(
    pairs: Sequence[tuple[NDArray[np.uint8], NDArray[np.uint8]]], grid: tuple[int, int], shape: tuple[int, ...]
) -> tuple[NDArray[np.float64], NDArray[np.bool_], str | None]:
    """Return the per-cell share of persistent pixels, the persistent pixels, and why every cell is NaN, if it is.

    A cell is NaN where under a fifth of its pixels are known.
    """
    seen = np.zeros(shape)
    still = np.zeros(shape)
    textured = moving = 0
    for earlier, later in pairs:
        a, b = earlier.astype(np.float32), later.astype(np.float32)
        d = np.abs(a - b)
        tex = np.maximum(_local_std(a), _local_std(b)) > TEXTURE
        if tex.mean() < 0.01:
            continue
        textured += 1
        evidence = _evidence(earlier, later, d, tex)
        if evidence is None:
            continue  # the camera did not move enough to show what moves with it: no evidence either way
        moving += 1
        seen += evidence
        still += evidence & (d < max(STILL, 0.2 * float(np.median(d[tex]))))
    known = seen >= MIN_EVIDENCE
    persistent = known & (still >= PERSISTENT * np.maximum(seen, 1))
    known_share = _cell_mean(known, grid)
    share = _cell_mean(persistent, grid) / np.maximum(known_share, 1e-6)
    cells = np.where(known_share >= 0.2, share, np.nan).astype(np.float64)
    if not np.isnan(cells).all():
        return cells, persistent, None
    return (
        cells,
        persistent,
        _why(
            (not pairs, "the video has too few frames"),
            (not textured, "too little texture"),
            (not moving, "too little camera motion"),
            (True, "too few frame pairs with camera motion"),
        ),
    )


def _similarity(a: NDArray[np.uint8], b: NDArray[np.uint8], avoid: NDArray[np.bool_]) -> NDArray[np.float64] | None:
    """Return the similarity transform taking frame a's coordinates to frame b's, from tracked corners, or None."""
    cv2 = _require_cv2()
    pts = cv2.goodFeaturesToTrack(a, 500, 0.01, 7, mask=(~avoid).astype(np.uint8))
    if pts is None or len(pts) < 20:
        return None
    nxt, ok, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None, winSize=(21, 21), maxLevel=3)
    ok = ok.ravel() == 1
    if ok.sum() < 20:
        return None
    m, inliers = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC, ransacReprojThreshold=1.5)
    return m if m is not None and inliers.sum() >= 20 else None


def _smoothness(grid: tuple[int, int]) -> tuple[list[int], list[int], list[float], int]:
    """Return the neighbor-smoothness rows of the solve, as COO rows, columns and values, and their count."""
    rows, cols = grid
    smooth_r: list[int] = []
    smooth_c: list[int] = []
    smooth_v: list[float] = []
    s_eq = 0
    for i in range(rows):
        for j in range(cols):
            for di, dj in ((0, 1), (1, 0)):
                if i + di < rows and j + dj < cols:
                    smooth_r += [s_eq, s_eq]
                    smooth_c += [i * cols + j, (i + di) * cols + j + dj]
                    smooth_v += [SMOOTH, -SMOOTH]
                    s_eq += 1
    return smooth_r, smooth_c, smooth_v, s_eq


def _softness_map(  # noqa: C901
    pairs: Sequence[tuple[Sample, Sample]], avoid: NDArray[np.bool_], fixed: NDArray[np.bool_], grid: tuple[int, int]
) -> tuple[NDArray[np.float64], str | None]:
    """Return the per-cell log attenuation of the same content's sharpness, and why every cell is NaN, if it is.

    A cell is NaN where it is the target of too few equations.
    """
    cv2 = _require_cv2()
    rows, cols = grid
    ph, pw = avoid.shape
    ch, cw = ph / rows, pw / cols
    yy, xx = np.mgrid[0:rows, 0:cols]
    centers = np.c_[(xx.ravel() + 0.5) * cw, (yy.ravel() + 0.5) * ch, np.ones(rows * cols)]
    r_idx, c_idx, vals, rhs = [], [], [], []
    count = np.zeros(rows * cols)
    eq = aligned = moving = 0
    for earlier, later in pairs:
        m = _similarity(earlier.proc, later.proc, avoid)
        if m is None:
            continue
        aligned += 1
        inv = cv2.invertAffineTransform(m)
        src = centers @ inv.T
        u, v = src[:, 0] / cw - 0.5, src[:, 1] / ch - 0.5
        if np.max(np.hypot(u - xx.ravel(), v - yy.ravel())) < 0.5:
            continue  # under half a cell of motion: every equation is a(x) - a(x)
        moving += 1
        inside = (u >= 0) & (u <= cols - 1) & (v >= 0) & (v <= rows - 1) & ~fixed.ravel()
        # Blur pairs come only from samples taken with a measure, so both logs are present.
        la, ma, lb = (
            cast(NDArray[np.float64], earlier.log_s),
            earlier.log_median,
            cast(NDArray[np.float64], later.log_s),
        )
        j0 = np.clip(np.floor(u).astype(int), 0, cols - 2)
        i0 = np.clip(np.floor(v).astype(int), 0, rows - 2)
        fu, fv = u - j0, v - i0
        corners = [
            (i0, j0, (1 - fu) * (1 - fv)),
            (i0, j0 + 1, fu * (1 - fv)),
            (i0 + 1, j0, (1 - fu) * fv),
            (i0 + 1, j0 + 1, fu * fv),
        ]
        src_log = cast(NDArray[np.float64], sum(w * la[i, j] for i, j, w in corners))
        keep = np.flatnonzero(inside & (src_log > ma + np.log(SOURCE_FLOOR)))
        if len(keep) < 20:
            continue
        r = lb.ravel()[keep] - src_log[keep]
        # The frame's overall sharpness changes between the two (motion blur, focus, exposure): an
        # offset common to the pair, which left in would integrate into a ramp across the frame.
        rhs.append(r - np.median(r))
        rows_k = eq + np.arange(len(keep))
        r_idx += [rows_k] + [rows_k] * 4
        c_idx += [keep] + [(i[keep] * cols + j[keep]) for i, j, _ in corners]
        vals += [np.ones(len(keep))] + [-w[keep] for _, _, w in corners]
        count[keep] += 1
        eq += len(keep)
    if eq:
        n = rows * cols
        smooth_r, smooth_c, smooth_v, s_eq = _smoothness(grid)
        data = coo_matrix((np.concatenate(vals), (np.concatenate(r_idx), np.concatenate(c_idx))), shape=(eq, n)).tocsr()
        reg = coo_matrix(
            (smooth_v + [1.0] * n, (smooth_r + [s_eq] * n, smooth_c + list(range(n)))), shape=(s_eq + 1, n)
        ).tocsr()
        b = np.concatenate(rhs)
        weights = np.ones(eq)
        a = np.zeros(n)
        for _ in range(3):  # Huber reweighting: a moving object or a bad alignment is a few wild equations
            # Pyright reads `lsqr` as SciPy's submodule of the same name; at runtime it is the function.
            a = lsqr(  # pyright: ignore[reportCallIssue]
                vstack([diags(weights) @ data, reg]), np.r_[weights * b, np.zeros(s_eq + 1)], atol=1e-6, btol=1e-6
            )[0]
            resid = np.abs(data @ a - b)
            weights = np.sqrt(np.minimum(1.0, HUBER / np.maximum(resid, 1e-9)))
        cells = np.where(count.reshape(grid) >= MIN_EVIDENCE, a.reshape(grid), np.nan)
        if not np.isnan(cells).all():
            return cells, None
    return np.full(grid, np.nan), _why(
        (not pairs, "the video has too few frames"),
        (not aligned, "too little texture to align frames"),
        (not moving, "too little camera motion"),
        (True, "too few usable frame pairs"),
    )


class CameraMaps(NamedTuple):
    """One video's maps, on the report grid, and why any of them is NaN everywhere.

    Attributes
    ----------
    ego_map : NDArray[np.float64]
        Share of each cell's known pixels that persist while the camera moves them.
    blur_map : NDArray[np.float64]
        :attr:`blur_map_raw` less its trend; all NaN when no blur map was wanted.
    blur_map_raw : NDArray[np.float64]
        Log attenuation of the same content's sharpness per cell; all NaN when no blur map was wanted.
    warnings : list[str]
        One message per map, or pair of blur maps, that is NaN everywhere, naming the reason.
    """

    ego_map: NDArray[np.float64]
    blur_map: NDArray[np.float64]
    blur_map_raw: NDArray[np.float64]
    warnings: list[str]


def camera_maps(
    samples: Mapping[int, Sample],
    starts: Sequence[int],
    grid: tuple[int, int] | None,
    *,
    blur: bool,
    sigma: float,
) -> CameraMaps:
    """Return one video's maps from its samples, keyed by frame index, and its kept start frames.

    Each start `k` pairs with frames ``k + 10`` (persistence and blur) and ``k + 30`` (blur), where
    both were sampled. `grid` is the solved grid, None when no frame was sampled.
    """
    blur_reason = "blur_map and blur_map_raw are NaN everywhere: {}."
    unknown = np.full(REPORT_GRID, np.nan)
    if grid is None or not samples:
        reason = "the video has too few frames"
        warnings = [f"ego_map is NaN everywhere: {reason}."] + ([blur_reason.format(reason)] if blur else [])
        return CameraMaps(unknown, unknown.copy(), unknown.copy(), warnings)

    shape = next(iter(samples.values())).proc.shape
    pairs = [(samples[k].proc, samples[k + GAPS[0]].proc) for k in starts if k in samples and k + GAPS[0] in samples]
    ego, persistent, reason = _persistence_map(pairs, grid, shape)
    warnings = [] if reason is None else [f"ego_map is NaN everywhere: {reason}."]
    if not blur:
        return CameraMaps(report(ego), unknown, unknown.copy(), warnings)

    cv2 = _require_cv2()
    avoid = cv2.dilate(persistent.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    fixed = np.nan_to_num(ego) >= FLAG_PERSISTENT
    blur_pairs = [(samples[k], samples[k + gap]) for gap in GAPS for k in starts if k in samples and k + gap in samples]
    raw, reason = _softness_map(blur_pairs, avoid, fixed, grid)
    if reason is not None:
        warnings.append(blur_reason.format(reason))
    return CameraMaps(report(ego), report(detrend(raw, sigma)), report(raw), warnings)
