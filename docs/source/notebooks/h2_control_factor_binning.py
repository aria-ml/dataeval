# ---
# jupyter:
#   jupytext:
#     default_lexer: ipython3
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: dataeval
#     language: python
#     name: python3
# ---

# %% [markdown]
# (control-factor-binning)=
# # How to control and reuse a factor's binning
#
# {class}`.Metadata` derives a factor's bins from the values it was handed, so the same factor measured on two
# collections is cut at different places and their {class}`.Balance` results are not comparable. A bin edge is a claim
# about the world, such as freezing at 0 °C, and a percentile makes no such claim. This guide declares the cuts you
# mean, records them and applies them unchanged to the next collection. See [Binning](../concepts/Binning.md).
#
# Use it when you compare bias results across collections, or when a factor has a boundary that means something.
#
# You need `dataeval`. This guide builds a winter and a spring collection of `temp_c`, `haze` and `site` in memory.

# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval
except Exception:
    pass

# %%
import json
import warnings
from pathlib import Path
from tempfile import mkdtemp

import numpy as np

from dataeval import Metadata
from dataeval.bias import Balance

# %% tags=["hide-input"]


def collection(seed: int, mean_temp: float, n: int = 600):
    rng = np.random.default_rng(seed)
    temp = rng.normal(mean_temp, 9.0, n)
    factors = {
        "temp_c": temp,
        "haze": np.clip(0.55 * (temp - mean_temp) / 9.0 + rng.normal(0.0, 0.9, n), -3, 3),
        "site": np.array(["ridge", "valley", "coast"])[rng.integers(0, 3, n)],
    }
    return factors, rng.integers(0, 4, n)


winter_factors, winter_labels = collection(seed=0, mean_temp=-2.0)
spring_factors, spring_labels = collection(seed=1, mean_temp=14.0)

# %% [markdown]
# ## Declare the cuts you mean
#
# {meth}`.Metadata.encoding` returns the map from a factor's values to its codes, and each record carries a `provenance`
# saying who chose it. `continuous_factor_bins` takes explicit edges, or a count when you only care about resolution,
# and `factor_levels` fixes a categorical vocabulary before any data is seen. Use `-np.inf` and `np.inf` as the outer
# edges so a value outside the data you have seen falls in the first or last bin instead of creating a new code.

# %%
FREEZING = [-np.inf, 0.0, 10.0, np.inf]  # below freezing / cold / mild
SITES = ["ridge", "valley", "coast"]

winter = Metadata.from_factors(
    winter_factors,
    class_labels=winter_labels,
    continuous_factor_bins={"temp_c": FREEZING},
    factor_levels={"site": SITES},
)

for name, spec in winter.encoding().items():
    print(f"{name:8} {type(spec).__name__:10} provenance={spec.provenance}")

# %% [markdown]
# `temp_c` reads `edges` and `site` reads `declared`. `haze` is still `derived`. No claim was made about it, and
# DataEval warns that it binned the factor automatically. Pass `strict=True` alongside `factor_levels` when the
# vocabulary is closed, and a value outside it is an error instead of being appended:

# %%
try:
    incomplete = Metadata.from_factors(
        winter_factors,
        class_labels=winter_labels,
        factor_levels={"site": ["ridge", "valley"]},
        strict=True,
    )
    _ = incomplete.factor_data
except ValueError as error:
    print(error)

# %% [markdown]
# ## Accept the cuts you have no opinion about
#
# You will not have a boundary in mind for every factor. {meth}`.Metadata.accept` fixes a derived placement as it is,
# without changing a code, and `accept()` with no arguments ratifies everything still `derived`.

# %%
before = winter.factor_data.copy()
winter.accept()
print(f"provenance: { ({n: s.provenance for n, s in winter.encoding().items()}) }")
print(f"codes unchanged: {np.array_equal(before, winter.factor_data)}")

# %% [markdown]
# ## Save the encoding
#
# {meth}`.Metadata.export_encoding` writes the whole map as JSON with sorted keys and a fixed indent, so a change to one
# factor's cutoff shows up in a code review as a change to one factor's cutoff. Infinities are written as the words
# `"inf"` and `"-inf"`.

# %%
workspace = Path(mkdtemp(prefix="dataeval-encoding-"))
descriptor = workspace / "encoding.json"
winter.export_encoding(descriptor)

print(json.dumps(json.loads(descriptor.read_text())["factors"]["temp_c"], indent=2))

# %% [markdown]
# {meth}`.Metadata.save` also carries the encoding, so a restored instance reproduces its codes instead of re-deriving
# them. What you pass to {meth}`.Metadata.load` still wins. The archive only fills in the factors you say nothing about.

# %%
archive = workspace / "winter.dem"
winter.save(archive)
restored = Metadata.load(archive)

print(f"edges:  {restored.encoding('temp_c').edges}")
print(f"digest: {restored.encoding_digest == winter.encoding_digest}")

# %% [markdown]
# ## Apply it to the next collection
#
# Hand the descriptor, or the mapping `encoding()` returns, to the next collection. The cut is reapplied rather than
# refitted. Spring is 16 °C warmer, and the freezing boundary stays where you put it. {attr}`.Metadata.encoding_digest`
# is a short hash of the whole encoding, and it rides along on every evaluator result under
# `result.meta().state["encoding_digest"]`. Two datasets sharing a digest were encoded identically.

# %%
spring = Metadata.from_factors(spring_factors, class_labels=spring_labels, encoding=descriptor)

print(f"winter temp_c: {winter.encoding('temp_c').edges}")
print(f"spring temp_c: {spring.encoding('temp_c').edges}")

# %%
print(f"winter: {winter.encoding_digest}")
print(f"spring: {spring.encoding_digest}")
print(f"comparable: {winter.encoding_digest == spring.encoding_digest}")

result = Balance().evaluate(spring)
print(f"on the result: {result.meta().state['encoding_digest']}")

# %% [markdown]
# ## Watch for a cut that stops fitting
#
# A locked cut is reapplied even when the data has shifted away from it, and a cut whose bins the data no longer reaches
# says so:

# %%
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    stale = Metadata.from_factors(
        spring_factors,
        class_labels=spring_labels,
        continuous_factor_bins={"temp_c": [-np.inf, -20.0, -10.0, 0.0, np.inf]},
    )
    _ = stale.factor_data

for warning in caught:
    if "bins unused" in str(warning.message):
        print(str(warning.message))

# %% [markdown]
# A cut that fits is silent. DataEval also reports an encoding finer than the sample can support, whether it came as a
# bin count or a declared vocabulary, and names no replacement count on purpose.
# [Binning](../concepts/Binning.md#a-fine-cut-costs-the-same-correlation-from-the-other-side) has the measurements.
# Neither report refits anything. {meth}`.Metadata.reencode` is the explicit request, and it returns a new instance so a
# result computed under the old codes stays attributable to them.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
cold = Metadata.from_factors(winter_factors, class_labels=winter_labels)
warm = Metadata.from_factors(spring_factors, class_labels=spring_labels)
assert cold.encoding("temp_c").provenance == "derived"
assert not np.array_equal(cold.encoding("temp_c").edges, warm.encoding("temp_c").edges)

assert {name: spec.provenance for name, spec in winter.encoding().items()} == {
    "haze": "accepted",
    "site": "declared",
    "temp_c": "edges",
}
try:
    strict = Metadata.from_factors(
        winter_factors, class_labels=winter_labels, factor_levels={"site": ["ridge", "valley"]}, strict=True
    )
    _ = strict.factor_data
except ValueError:
    pass
else:
    raise AssertionError("a value outside a strict vocabulary should be refused")

assert json.loads(descriptor.read_text())["factors"]["temp_c"]["edges"][0] == "-inf"
assert np.array_equal(restored.encoding("temp_c").edges, winter.encoding("temp_c").edges)
assert restored.encoding_digest == winter.encoding_digest
assert np.array_equal(spring.encoding("temp_c").edges, winter.encoding("temp_c").edges)
assert spring.encoding_digest == winter.encoding_digest
assert result.meta().state["encoding_digest"] == spring.encoding_digest
assert any("bins unused" in str(warning.message) for warning in caught)

# %% [markdown]
# ## Next steps
#
# - [Binning](../concepts/Binning.md) — Understand how binning choices change bias scores and when a cut is a claim about the world.
# - [Dataset Bias and Coverage](../concepts/DatasetBias.md) — Understand how class imbalance and missing data impact model reliability.
# - [Identify bias and correlations](./tt_identify_bias.py) — Analyze class and metadata correlations to spot potential dataset bias.
