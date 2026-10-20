#!/usr/bin/env python3
"""
Generate the figures on the Parity concept page (docs/source/concepts/Parity.md).

Every figure is computed with the shipped Parity code, and the script asserts the values the
page quotes, so a change that moves them fails here instead of leaving the page stale. The
simulations reuse the null designs that CI runs (tests/core/test_rao_scott.py).

Usage (from the repository root, in the project environment):
    python docs/generate_parity_figures.py [--output-dir DIR]
"""

import argparse
import sys
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataeval.bias import Parity  # noqa: E402
from dataeval.core import parity  # noqa: E402
from tests.core.test_rao_scott import (  # noqa: E402
    mot_frame_factor,
    od_detection_factor,
    one_bursty_level,
    reviewer_design,
    seq40,
)

warnings.filterwarnings("ignore")  # Parity's experimental notice

INK = "#1f2328"
MUTED = "#6e7781"
BAND = "#eaeef2"
UNDER = "#d55e00"  # Okabe-Ito vermilion
OVER = "#0072b2"  # Okabe-Ito blue
NEUTRAL = "#8c959f"
STATUS_COLOR = {"underrepresented": UNDER, "overrepresented": OVER, "no_finding": NEUTRAL}

plt.rcParams.update({
    "font.size": 9,
    "text.color": INK,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "svg.hashsalt": "dataeval-parity",
})


@dataclass
class Rows:
    """The fields of MetadataLike that Parity reads, built from plain arrays."""

    class_labels: np.ndarray
    factor_data: np.ndarray
    factor_names: list[str]
    is_binned: list[bool]
    index2label: dict[int, str]
    item_indices: np.ndarray | None = None


def save(fig: Figure, path: Path) -> None:
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.12, metadata={"Date": None})
    plt.close(fig)
    print(f"wrote {path}")


def sci(p: float) -> str:
    if p >= 0.01:
        return f"{p:.2f}"
    mantissa, exponent = f"{p:.1e}".split("e")
    return rf"${mantissa} \times 10^{{{int(exponent)}}}$"


def lift_band(out_dir: Path) -> None:
    """The temperature example: lift on a log axis, the lift band, and each cell's status."""
    counts = {0: (12, 80, 11), 1: (108, 140, 49)}  # level -> (pedestrian, vehicle, cyclist)
    level = np.concatenate([np.full(sum(c), lv) for lv, c in counts.items()])
    labels = np.concatenate([np.repeat([0, 1, 2], c) for c in counts.values()])
    classes = {0: "pedestrian", 1: "vehicle", 2: "cyclist"}
    rows = Rows(labels, level[:, None].astype(np.int64), ["temp_c"], [True], classes, np.arange(labels.size))
    result = Parity().evaluate(rows)
    cells = result.cells.with_columns(pl.col("level").cast(pl.String), pl.col("class_name").cast(pl.String))

    order = [(lv, name) for lv in ("0", "1") for name in classes.values()]
    picked = [
        cells.filter((pl.col("level") == lv) & (pl.col("class_name") == name)).row(0, named=True) for lv, name in order
    ]
    # The values the page's table quotes.
    assert [round(c["lift"], 2) for c in picked] == [0.39, 1.41, 0.71, 1.21, 0.86, 1.10]
    assert [c["status"] for c in picked] == ["underrepresented", "overrepresented"] + ["no_finding"] * 4
    assert round(result.factors["score"][0], 2) == 0.27
    assert result.factors["status"][0] == "weak"

    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ys = [5.6, 4.6, 3.6, 2.0, 1.0, 0.0]
    ax.axvspan(0.8, 1.25, color=BAND, zorder=0)
    ax.axvline(1.0, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.text(1.0, 6.45, "lift band", ha="center", va="bottom", color=MUTED, fontsize=8)
    for y, c in zip(ys, picked, strict=True):
        color = STATUS_COLOR[c["status"]]
        face = color if c["is_deviant"] else "white"
        ax.scatter(c["lift"], y, s=70, facecolor=face, edgecolor=color, linewidth=1.6, zorder=3)
        ax.annotate(
            f"{c['status']}   adjusted p {sci(c['p_value_adjusted'])}",
            xy=(1.02, y),
            xycoords=("axes fraction", "data"),
            va="center",
            color=color if c["status"] != "no_finding" else INK,
        )
    ax.set_xscale("log")
    ax.set_xlim(0.3, 2.2)
    ticks = [0.4, 0.5, 0.8, 1.0, 1.25, 1.5, 2.0]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_yticks(ys, [name for _, name in order])
    ax.set_ylim(-0.6, 6.4)
    ax.text(-0.02, 6.2, "below 0 °C", transform=ax.get_yaxis_transform(), ha="right", color=MUTED)
    ax.text(-0.02, 2.6, "at or above 0 °C", transform=ax.get_yaxis_transform(), ha="right", color=MUTED)
    ax.set_xlabel("lift = observed / expected (log scale)")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [
        Line2D([], [], ls="", marker="o", ms=8, mfc=INK, mec=INK, label="adjusted p ≤ 0.05"),
        Line2D([], [], ls="", marker="o", ms=8, mfc="white", mec=INK, mew=1.4, label="adjusted p > 0.05"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False)
    save(fig, out_dir / "lift_band.svg")


def kish(rows_per_group: np.ndarray) -> float:
    return rows_per_group.sum() ** 2 / (rows_per_group**2).sum()


def effective_groups(out_dir: Path) -> None:
    """Two levels of 400 rows each: spread over 100 images, or dominated by one."""
    level_a = np.full(100, 4)
    level_b = np.r_[200, np.full(50, 4)]
    k_a, k_b = kish(level_a), kish(level_b)
    assert round(k_a, 1) == 100.0
    assert round(k_b, 1) == 3.9

    # Parity reports the smaller level's effective groups, and with 2 <= k < 5 every cell is too_thin.
    sizes = np.r_[level_a, level_b]
    groups = np.repeat(np.arange(sizes.size), sizes)
    factor = np.repeat(np.r_[np.zeros(level_a.size), np.ones(level_b.size)], sizes).astype(np.intp)
    labels = np.random.default_rng(0).integers(0, 2, groups.size)
    result = parity(factor[:, None], labels, groups=groups)
    assert np.isclose(result["effective_groups"][0], k_b)
    assert result["cells"]["sparse"].all()

    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    for y, level in ((1.0, level_a), (0.0, level_b)):
        ax.barh(y, level, left=np.r_[0, np.cumsum(level)[:-1]], height=0.55, color="#9ecae1", edgecolor="white", lw=0.5)
    ax.text(100, 0.0, "one image, 200 rows", ha="center", va="center", color=INK)
    ax.text(408, 1.0, f"100 images, $k$ = {k_a:.0f}", va="center")
    ax.text(408, 0.0, f"51 images, $k$ = {k_b:.1f}", va="center")
    ax.set_yticks([1.0, 0.0], ["level A", "level B"])
    ax.set_xlim(0, 400)
    ax.set_xticks([0, 100, 200, 300, 400])
    ax.set_xlabel("rows at the level, one segment per image")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, out_dir / "effective_groups.svg")


# The four null designs of the page's "Why rows are not independent" table, as CI generates them.
NULL_DESIGNS = [
    ("Sequence-level factor against detection labels, 40 sequences of 50 frames", seq40, 2000, 1),
    ("Detection-level factor whose values cluster by image (object detection)", od_detection_factor, 2000, 2),
    ("Frame-level factor in multi-object tracking", mot_frame_factor, 2000, 3),
    ("20 images of 20 detections at one level, 380 single-detection images at the other", reviewer_design, 4000, 101),
]


def false_positive_rates(make, reps: int, seed: int) -> tuple[float, float]:
    """Rejection rates at 0.05 of a row-level Pearson test and of Parity's grouped test."""
    rng = np.random.default_rng(seed)
    row = grouped = 0
    for _ in range(reps):
        f, y, g = make(rng)
        # Declaring the rows independent runs Pearson's test, the same as scipy's chi2_contingency(correction=False).
        row += parity(f[:, None], y, assume_independent=True)["p_values"][0] < 0.05
        grouped += parity(f[:, None], y, groups=g)["p_values"][0] < 0.05  # NaN (untestable) is not a rejection
    return row / reps, grouped / reps


def row_level_fpr(out_dir: Path) -> None:
    """False-positive rates under the null: a row-level test against Parity's grouped test."""
    rates = []
    for name, make, reps, seed in NULL_DESIGNS:
        row, grouped = false_positive_rates(make, reps, seed)
        se = np.sqrt(grouped * (1 - grouped) / reps)
        print(f"  {name}: {reps} datasets, row-level {row:.1%}, grouped {grouped:.1%} ± {se:.1%}")
        rates.append((name, row, grouped))

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for i, (_, row, grouped) in enumerate(rates):
        y = len(rates) - 1 - i
        ax.barh(y + 0.19, row * 100, height=0.36, color=UNDER)
        ax.barh(y - 0.19, grouped * 100, height=0.36, color=OVER)
        ax.text(row * 100 + 1, y + 0.19, f"{row:.1%}", va="center", fontsize=8)
        ax.text(max(grouped * 100, 5) + 1, y - 0.19, f"{grouped:.1%}", va="center", fontsize=8)
    ax.axvline(5, color=INK, lw=0.9, ls=(0, (3, 2)))
    ax.text(5.8, len(rates) - 0.45, "nominal 5%", fontsize=8, va="bottom")
    ax.set_yticks(
        [len(rates) - 1 - i for i in range(len(rates))], [textwrap.fill(name, 34) for name, _, _ in rates], fontsize=8
    )
    ax.set_xlim(0, 110)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("false-positive rate at α = 0.05, factor generated independently of the class")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [Patch(color=UNDER, label="row-level Pearson test"), Patch(color=OVER, label="Parity's grouped test")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.35, -0.2), ncol=2, frameon=False)
    save(fig, out_dir / "row_level_fpr.svg")


def bursts(out_dir: Path) -> None:
    """Two draws from one null: a rare class that comes in bursts at level 0 only."""
    tiles = {0: "#ffffff", 1: "#f6c6a4", 2: UNDER}  # no rare rows, a few, a burst
    names = {0: "common", 1: "rare"}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 3.9), gridspec_kw={"hspace": 0.12, "wspace": 0.12})
    for col, (seed, held) in enumerate(((1, 2), (29, 0))):
        f, y, g = one_bursty_level(np.random.default_rng(seed))
        rare = np.bincount(g, weights=y).astype(int)
        image_level = f[::10]  # one_bursty_level gives every image 10 rows
        rows = Rows(y, f[:, None].astype(np.int64), ["site"], [False], names, g)
        result = Parity().evaluate(rows)
        cell = result.cells.filter((pl.col("level").cast(pl.String) == "0") & (pl.col("class_name") == "rare"))
        cell = cell.row(0, named=True)
        assert int(((rare >= 5) & (image_level == 0)).sum()) == held
        assert cell["status"] == ("too_thin" if held else "underrepresented")
        for row, level in enumerate((0, 1)):
            ax = axes[row, col]
            counts = rare[image_level == level]
            shade = np.select([counts >= 5, counts > 0], [2, 1], 0)
            for i, s in enumerate(shade):
                ax.add_patch(Rectangle((i % 15, -(i // 15)), 0.86, 0.86, fc=tiles[s], ec="#afb8c1", lw=0.4))
            ax.set_xlim(-0.2, 15)
            ax.set_ylim(-(len(shade) // 15) - 0.2, 1)
            ax.set_aspect("equal")
            ax.axis("off")
            if col == 0:
                ax.text(-0.6, 0.5 - (len(shade) // 15) / 2, f"level {level}", ha="right", va="center", rotation=90)
        title = f"Draw {'AB'[col]}: {held} bursty image{'s' if held != 1 else ''} at level 0"
        axes[0, col].set_title(title, fontsize=9, loc="left")
        status = cell["status"]
        detail = f", adjusted p {sci(cell['p_value_adjusted'])}" if status == "underrepresented" else ""
        axes[1, col].text(
            0,
            axes[1, col].get_ylim()[0] - 0.9,
            f"cell (level 0, rare): observed {cell['observed']}, expected {cell['expected']:.1f}\n→ {status}{detail}",
            va="top",
            color=UNDER if status == "underrepresented" else INK,
        )
    handles = [
        Patch(fc=tiles[0], ec="#afb8c1", label="no rare rows"),
        Patch(fc=tiles[1], ec="#afb8c1", label="1 to 4 rare rows"),
        Patch(fc=tiles[2], ec="#afb8c1", label="5 or more: a burst"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        frameon=False,
        title="one tile per image (10 rows)",
        title_fontsize=8,
    )
    save(fig, out_dir / "burst_contraindication.svg")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[1])
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "docs/source/_static/images/concepts/parity", help="where to write"
    )
    out_dir = parser.parse_args().output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    lift_band(out_dir)
    effective_groups(out_dir)
    bursts(out_dir)
    row_level_fpr(out_dir)


if __name__ == "__main__":
    main()
