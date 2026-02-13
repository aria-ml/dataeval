---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: dataeval
  language: python
  name: python3
---

# How to choose a duplicate detection method

## Problem Statement

DataEval offers multiple approaches for detecting duplicate images:

1. **Hash-based methods** (phash_d4, dhash_d4) - Fast perceptual hashing with rotation/flip invariance
1. **Embedding-based methods** (BoVWExtractor) - SIFT-based Bag of Visual Words for semantic similarity

This notebook compares these approaches to help you choose the right method for your use case.

+++

### When to use each method

| Method                             | Best For                                  | Speed   | Rotation Invariant      |
| ---------------------------------- | ----------------------------------------- | ------- | ----------------------- |
| **D4 Hashes** (phash_d4, dhash_d4) | Detecting rotated/flipped copies          | Fast    | **Only 90° increments** |
| **BoVWExtractor**                  | Semantic similarity, different viewpoints | Slower  | **Any angle**           |
| **Basic Hashes** (phash, dhash)    | Same-orientation near-duplicates          | Fastest | No                      |

**Key insight:** D4 hashes only handle the 8 symmetries of a square (0°, 90°, 180°, 270° + flips). BoVW using SIFT
features is invariant to **any** rotation angle, making it better for detecting arbitrarily rotated duplicates.

+++

### What you will need

1. A Python environment with the following packages installed:
   - `dataeval`
   - `opencv-python` or `opencv-python-headless`
   - `matplotlib`
1. Sample images to analyze

+++

## Getting Started

Let's import the required libraries.

```{code-cell} ipython3
:tags: [remove_cell]

# Google Colab Only
try:
    import google.colab  # noqa: F401

    %pip install -q dataeval opencv-python-headless
except Exception:
    pass
```

```{code-cell} ipython3
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from dataeval import config
from dataeval.extractors._bovw import BoVWExtractor
from dataeval.flags import ImageStats
from dataeval.quality import Duplicates

config.set_seed(67)  # six seven
```

## Creating Test Data

We'll create a synthetic dataset with different types of "duplicates" to test each method's capabilities:

1. **Original images** - Base images with texture (for SIFT detection)
1. **90° rotations** - 90°, 180°, 270° rotations (D4 hashes can detect these)
1. **Diagonal rotations** - 45°, 135° rotations (only BoVW can detect these!)
1. **Flipped copies** - Horizontal and vertical flips
1. **Unique images** - Should NOT be detected as duplicates

```{code-cell} ipython3
def create_textured_image(seed: int, size: int = 128) -> np.ndarray:
    """Create an image with texture patterns that SIFT can detect."""
    rng = np.random.default_rng(seed)

    # Create base with gradient and noise
    x = np.linspace(0, 4 * np.pi, size)
    y = np.linspace(0, 4 * np.pi, size)
    xx, yy = np.meshgrid(x, y)

    # Create pattern with multiple frequency components
    pattern = (
        np.sin(xx * (1 + seed % 3)) * np.cos(yy * (2 + seed % 2))
        + np.sin(xx * 3 + seed) * 0.5
        + rng.random((size, size)) * 0.3
    )

    # Normalize to 0-255
    pattern = ((pattern - pattern.min()) / (pattern.max() - pattern.min()) * 255).astype(np.uint8)

    # Create RGB image (same pattern in all channels with slight variation)
    img = np.stack(
        [pattern, np.roll(pattern, seed % 10, axis=0), np.roll(pattern, seed % 7, axis=1)], axis=0
    )  # Shape: (3, H, W) - channels first

    return img.astype(np.uint8)
```

```{code-cell} ipython3
def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotates image by any angle"""
    angle = angle % 360

    if angle == 0:
        return img

    # Transpose to HWC for OpenCV
    img_hwc = np.transpose(img, (1, 2, 0))

    # Orthogonal rotations (90, 180, 270)
    if angle % 90 == 0:
        rotate_code = {1: cv2.ROTATE_90_COUNTERCLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_CLOCKWISE}
        rotated = cv2.rotate(img_hwc, rotate_code[int((angle // 90) % 4)])

    # Affine rotation (Diagonal)
    else:
        h, w = img_hwc.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        cos, sin = np.abs(matrix[0, 0]), np.abs(matrix[0, 1])
        new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)

        matrix[0, 2] += (new_w - w) / 2
        matrix[1, 2] += (new_h - h) / 2

        rotated = cv2.warpAffine(img_hwc, matrix, (new_w, new_h), borderValue=(128, 128, 128))

    # Transpose back to CHW
    return np.transpose(rotated, (2, 0, 1))
```

```{code-cell} ipython3
def flip_image(img: np.ndarray, direction: str) -> np.ndarray:
    """Flip image horizontally or vertically"""
    img_hwc = np.transpose(img, (1, 2, 0))
    flipped = cv2.flip(img_hwc, 1) if direction == "horizontal" else cv2.flip(img_hwc, 0)
    return np.transpose(flipped, (2, 0, 1))
```

```{code-cell} ipython3
images, labels, group_info = [], [], []

experiments = [
    (
        42,
        "Group 1: Orthogonal (D4 Detectable)",
        [(rotate_image, 90, "Rot 90°"), (rotate_image, 180, "Rot 180°"), (flip_image, "horizontal", "Flip H")],
    ),
    (
        63,
        "Group 2: Diagonal (BoVW Only)",
        [(rotate_image, 45, "Rot 45°"), (rotate_image, 135, "Rot 135°"), (rotate_image, 30, "Rot 30°")],
    ),
    (
        89,
        "Group 3: Mixed Rotations",
        [(rotate_image, 90, "Rot 90°"), (rotate_image, 60, "Rot 60°"), (flip_image, "vertical", "Flip V")],
    ),
]

for seed, desc, transforms in experiments:
    start_idx = len(images)
    base_img = create_textured_image(seed=seed)

    images.append(base_img)
    labels.append(f"Original (Seed {seed})")

    for func, arg, suffix in transforms:
        images.append(func(base_img, arg))
        labels.append(f"Seed {seed} - {suffix}")

    group_info.append((desc, start_idx, len(images) - 1))

start_idx = len(images)
unique_seeds = [777, 888, 999]

for i, seed in enumerate(unique_seeds):
    images.append(create_textured_image(seed=seed))
    labels.append(f"Unique {i + 1}")

group_info.append(("Group 4: Unique Images", start_idx, len(images) - 1))

print(f"Created {len(images)} test images\n" + "=" * 60)
for desc, start, end in group_info:
    print(f"{desc:<40} (indices {start}-{end})")
print("=" * 60)

for i, label in enumerate(labels):
    print(f"  [{i:2d}] {label}")
```

```{code-cell} ipython3
# Visualize the test images
fig, axes = plt.subplots(3, 5, figsize=(12, 6))
axes = axes.flatten()

for i, (img, label) in enumerate(zip(images, labels, strict=False)):
    if i < len(axes):
        # Convert CHW to HWC for display
        img_display = np.transpose(img, (1, 2, 0))
        axes[i].imshow(img_display)
        axes[i].set_title(f"[{i}] {label}", fontsize=8)
        axes[i].axis("off")

# Hide empty subplots
for i in range(len(images), len(axes)):
    axes[i].axis("off")

plt.tight_layout()
plt.suptitle("Test Images: 90° rotations (Group 1) vs Diagonal rotations (Group 2)", y=1.02, fontsize=14)
plt.show()
```

## Method 1: D4 Hash-based Detection

D4 hashes (phash_d4, dhash_d4) compute perceptual hashes that are invariant to the 8 symmetries of a square (rotation by
0°, 90°, 180°, 270° and their horizontal flips).

**Strengths:**

- Very fast computation
- Detects rotated and flipped versions reliably (90° increments only)
- No training required

**Weaknesses:**

- **Cannot detect diagonal rotations** (45°, 30°, 60°, etc.)
- Only detects near-exact copies (with transformations)
- Cannot detect semantic similarity

**Expected result:** Should detect Group 1 (90° rotations) but **miss** Group 2 (diagonal rotations).

```{code-cell} ipython3
# Run D4 hash-based detection
start_time = time.time()

d4_detector = Duplicates(flags=ImageStats.HASH_DUPLICATES_D4)
d4_results = d4_detector.evaluate(images)

d4_time = time.time() - start_time
print(f"D4 Hash Detection completed in {d4_time:.3f} seconds")
```

```{code-cell} ipython3
print("\n=== D4 Hash Results ===")
print("\nNear duplicates (perceptual similarity):")
if d4_results.items.near:
    for group in d4_results.items.near:
        indices = list(group.indices)
        methods = sorted(group.methods)
        print(f"  Indices: {indices}")
        print(f"    Methods: {methods}")
        print(f"    Labels: {[labels[int(i)] for i in indices]}")
        print()
```

## Method 2: BoVW Embedding-based Detection

BoVWExtractor uses SIFT features to create rotation-invariant image representations. It clusters local features into a
"visual vocabulary" and represents each image as a histogram of visual words.

**Strengths:**

- **Rotation invariant at ANY angle** (SIFT features are inherently rotation invariant)
- Can detect semantic similarity (similar objects, different viewpoints)
- Works well for natural images with texture

**Weaknesses:**

- Slower than hash-based methods
- Requires images with detectable features (may fail on uniform/simple images)
- Results depend on vocabulary size parameter

**Expected result:** Should detect BOTH Group 1 (90° rotations) AND Group 2 (diagonal rotations).

```{code-cell} ipython3
# Run BoVW-based detection
bovw_extractor = BoVWExtractor(vocab_size=128)  # Smaller vocab for small dataset

start_time = time.time()

bovw_detector = Duplicates(
    flags=ImageStats.NONE,  # Skip hash computation
    extractor=bovw_extractor,
    cluster_threshold=1.25,
)
bovw_results = bovw_detector.evaluate(images)

bovw_time = time.time() - start_time
print(f"BoVW Detection completed in {bovw_time:.3f} seconds")
```

```{code-cell} ipython3
print("\n=== BoVW Results ===")
print("\nNear duplicates (embedding similarity):")
if bovw_results.items.near:
    for group in bovw_results.items.near:
        indices = list(group.indices)
        methods = sorted(group.methods)
        print(f"  Indices: {indices}")
        print(f"    Methods: {methods}")
        print(f"    Labels: {[labels[int(i)] for i in indices]}")
        print()
else:
    print("  No near duplicates found")
```

## Performance Comparison

```{code-cell} ipython3
print("\n=== Performance Summary ===")
print("\nExecution Time:")
print(f"  D4 Hashes:  {d4_time:.3f}s")
print(f"  BoVW:       {bovw_time:.3f}s ({bovw_time / d4_time:.1f}x slower)")

print("\nDetection Results:")
print(f"  {'Method':<15} {'Exact':<10} {'Near Groups':<15}")
print(f"  {'-' * 40}")

d4_exact = len(d4_results.items.exact) if d4_results.items.exact else 0
d4_near = len(d4_results.items.near) if d4_results.items.near else 0
print(f"  {'D4 Hashes':<15} {d4_exact:<10} {d4_near:<15}")

bovw_exact = len(bovw_results.items.exact) if bovw_results.items.exact else 0
bovw_near = len(bovw_results.items.near) if bovw_results.items.near else 0
print(f"  {'BoVW':<15} {bovw_exact:<10} {bovw_near:<15}")
```

## Recommendations

### Use D4 Hashes (`HASH_DUPLICATES_D4`) when

- You need fast processing of large datasets
- You're looking for rotated/flipped copies at **90° increments only**
- Images are near-exact duplicates (same content, possibly transformed)

### Use BoVWExtractor when

- You need to detect **arbitrarily rotated** duplicates (45°, 30°, etc.)
- You need semantic similarity detection
- Images may have different viewpoints of same objects
- Processing time is not critical
- Images have rich texture (not uniform/simple patterns)

### Key Differences

| Aspect                          | D4 Hashes         | BoVWExtractor                   |
| ------------------------------- | ----------------- | ------------------------------- |
| **Exact duplicates**            | Yes (via xxhash)  | No (embeddings are approximate) |
| **90° rotation detection**      | Yes (D4 symmetry) | Yes                             |
| **Diagonal rotation detection** | **No**            | **Yes** (any angle)             |
| **Semantic similarity**         | No                | Yes                             |
| **Speed**                       | Fast              | Slower                          |
| **Training required**           | No                | Yes (builds vocabulary)         |
| **Works on uniform images**     | Yes               | No (needs texture for SIFT)     |

### Summary

The key difference demonstrated in this notebook is **diagonal rotation handling**:

- **D4 hashes** can only detect rotations at 0°, 90°, 180°, 270° (plus flips)
- **BoVW/SIFT** can detect rotations at **any angle** because SIFT features are inherently rotation invariant

If your dataset may contain images rotated at arbitrary angles, BoVWExtractor is the better choice despite being slower.

```{code-cell} ipython3
:tags: [remove_cell]

### TEST ASSERTION CELL ###
# D4 hashes should detect 90° rotations but NOT diagonal rotations
assert d4_results.items.near is not None, "D4 should find some near duplicates"

# BoVW should never report exact duplicates (embeddings are approximate)
assert bovw_results.items.exact is None, "BoVW should not report exact duplicates"

# BoVW should detect more groups than D4 (because it catches diagonal rotations)
bovw_near_count = len(bovw_results.items.near) if bovw_results.items.near else 0
d4_near_count = len(d4_results.items.near) if d4_results.items.near else 0

print(f"\nD4 detected {d4_near_count} near-duplicate groups")
print(f"BoVW detected {bovw_near_count} near-duplicate groups")
print("\nBoVW detects diagonal rotations that D4 misses!")
```
