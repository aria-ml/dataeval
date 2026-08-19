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
# (control-factor-binning)=
# # How to control and reuse a factor's binning

# %% [markdown]
# ## Problem statement
#
# You run {class}`.Balance` on this month's collection and again on next month's,
# and the two `temp_c` results disagree. Nothing about the sensor changed. What
# changed is where the bins fell: {class}`.Metadata` derives them from whatever
# values it was handed, so the same factor measured twice is cut twice, into a
# different number of intervals in different places. The two numbers were never
# comparable.
#
# That is the default because there is no better guess to make on your behalf.
# But a bin edge is not a guess - it is a *claim about the world*. Cutting
# temperature at 0 °C says freezing matters and 3 °C is the same as 20 °C for
# this question. Nobody made that claim on the run above; a percentile did.
#
# This guide shows how to take that decision back: see what was decided for you,
# declare the cuts you actually mean, ratify the ones you do not, write the whole
# map out for review, and apply it unchanged to the next dataset.

# %% [markdown]
# ### When to use
#
# Read this when you compare bias results across collections, when a result is
# going into a report someone else has to trust, or when a factor has a boundary
# that means something - a freezing point, a legal altitude limit, a rated range.

# %% [markdown]
# ### What you will need
#
# 1. A Python environment with `dataeval` installed
# 1. No data of your own; this guide builds two small collections in memory

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
import polars as pl

from dataeval import Metadata
from dataeval.bias import Balance

# %% [markdown]
# ## Build two collections of the same thing
#
# The same three factors measured twice: a winter collection and a spring one.
# `temp_c` is a temperature, `haze` an atmospheric index that tracks it loosely,
# and `site` the capture location. Only the season differs.

# %%


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
# ## 1. See what was decided for you
#
# {meth}`.Metadata.encoding` returns the map from a factor's values to its codes:
# a {class}`.BinSpec` where the factor was cut, a {class}`.LevelSpec` where it was
# a category. Every record carries a `provenance` saying **who chose it**.

# %%
winter = Metadata.from_factors(winter_factors, class_labels=winter_labels)
spring = Metadata.from_factors(spring_factors, class_labels=spring_labels)

# Binning is lazy, so touch both now: the warning below is the subject of this guide
# and is worth reading once, here, rather than interleaved through every later cell.
_ = winter.factor_data
_ = spring.factor_data

for name, spec in winter.encoding().items():
    print(f"{name:8} {type(spec).__name__:10} provenance={spec.provenance}")

# %% [markdown]
# `derived` means nobody chose. Compare what that produced on the two
# collections:

# %%
print(f"winter temp_c: {np.round(winter.encoding('temp_c').edges, 1)}")
print(f"spring temp_c: {np.round(spring.encoding('temp_c').edges, 1)}")

# %% [markdown]
# Four bins against five, in different places. Code `2` means one thing in the
# winter result and another in the spring one, so the two `temp_c` rows of a
# {class}`.Balance` output are not measuring the same variable. Nothing warns you
# at comparison time, because by then both are just integers - which is why the
# warning above fires when the cut is *made* instead.

# %%
# `haze` stays automatic for the rest of this guide and would repeat that advice at
# every construction below, so it is silenced after the first reading.
warnings.filterwarnings("ignore", message=".*binned automatically")

# %% [markdown]
# ## 2. Declare the cut you actually mean
#
# `continuous_factor_bins` takes explicit edges - the boundaries that carry your
# meaning - or a plain count when you only care about resolution. `factor_levels`
# is the categorical counterpart: a vocabulary fixed **before** any data is seen,
# so code `i` means `levels[i]` in every dataset that declares it.
#
# Use `-np.inf` and `np.inf` as the outer edges. They are what make the cut safe
# to reapply: a colder-than-anything value falls into the first bin rather than
# creating a new code that means something different from every code before it.

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
# `temp_c` reads `edges` and `site` reads `declared`. `haze` is still `derived` -
# no claim was made about it, and the library will not invent one.
#
# Pass `strict=True` alongside `factor_levels` when the vocabulary is closed. A
# value outside it is then an error naming the value, instead of being quietly
# appended:

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
# ## 3. Ratify the cuts you do not have an opinion about
#
# You will not have a boundary in mind for every factor. `haze` is a case where
# the automatic cut is *fine* - it just needs to stop moving.
# {meth}`.Metadata.accept` fixes a derived placement in place, without changing a
# single code:

# %%
before = winter.factor_data.copy()
winter.accept()
print(f"provenance: { ({n: s.provenance for n, s in winter.encoding().items()}) }")
print(f"codes unchanged: {np.array_equal(before, winter.factor_data)}")

# %% [markdown]
# The distinction `accept` records is the whole point of the vocabulary: `edges`
# and `declared` mean *a person decided this*, `accepted` means *a person looked
# at this and signed it off*, and `derived` means *nobody has looked yet*. Calling
# `accept()` with no arguments ratifies everything still in the last category.

# %% [markdown]
# ## 4. Write it out for review
#
# {meth}`.Metadata.export_encoding` writes the whole map as JSON with sorted keys
# and a fixed indent. That is deliberate: a descriptor is policy, so it belongs in
# version control next to your code, where a change to one factor's cutoff shows
# up in a pull request as a change to one factor's cutoff.

# %%
workspace = Path(mkdtemp(prefix="dataeval-encoding-"))
descriptor = workspace / "encoding.json"
winter.export_encoding(descriptor)

print(json.dumps(json.loads(descriptor.read_text())["factors"]["temp_c"], indent=2))

# %% [markdown]
# Infinities are spelled as the words `"inf"` and `"-inf"`. JSON has no literal
# for them, and a reader sees a word rather than a number that looks like a
# measurement.

# %% [markdown]
# ## 5. Encode the next dataset against it
#
# Hand the descriptor - or the mapping {meth}`.Metadata.encoding` returns - to the
# next collection. The cut is **reapplied, not refitted**: spring is 16 °C warmer,
# and the freezing boundary stays exactly where you put it.

# %%
spring = Metadata.from_factors(spring_factors, class_labels=spring_labels, encoding=descriptor)

print(f"winter temp_c: {winter.encoding('temp_c').edges}")
print(f"spring temp_c: {spring.encoding('temp_c').edges}")

# %% [markdown]
# {attr}`.Metadata.encoding_digest` is a short hash over the whole descriptor. Two
# datasets sharing it were encoded identically, so their results are comparable -
# and it rides along on every evaluator result under
# `result.meta().state["encoding_digest"]`, which is what lets you check that
# months later.

# %%
print(f"winter: {winter.encoding_digest}")
print(f"spring: {spring.encoding_digest}")
print(f"comparable: {winter.encoding_digest == spring.encoding_digest}")

result = Balance().evaluate(spring)
print(f"on the result: {result.meta().state['encoding_digest']}")

# %% [markdown]
# ## 6. Hear about it when a cut stops fitting
#
# A locked cut is reapplied even when the data has moved out from under it, which
# is the intent - re-fitting is what locking it prevents - but it is worth
# hearing about. A cut whose bins the data no longer reaches says so:

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
# A cut that fits is silent. The same mechanism reports the other way a declaration
# goes wrong - an encoding finer than the sample can support, whether it arrived as
# a bin count or as a declared vocabulary, since either one fills a contingency
# table the same way. That report names no replacement count on purpose: the
# resolution that recovers the most association moves with the sample size, so
# there is no number to give.
# [Binning](../concepts/Binning.md#a-fine-cut-costs-the-same-correlation-from-the-other-side)
# has the measurements.
#
# Neither report re-fits anything - if you want that, {meth}`.Metadata.reencode`
# is the explicit request, and it returns a new instance so a result already
# computed under the old codes stays attributable to them.

# %% [markdown]
# ## 7. Choose what the evaluator reads
#
# Recording the cut lets {class}`.Balance` do something it could not before:
# honour it. Its `factor_source` decides, **per factor**, whether to score the
# codes or the values they were cut from.
#
# The default, `"auto"`, reads the codes wherever somebody made a claim - `edges`,
# `count` or `accepted` - and the unbinned values wherever nobody did. A declared
# cutoff is never read past. The `scored_as` column names which of the three
# regimes produced each row:

# %%
mixed = Metadata.from_factors(
    winter_factors,
    class_labels=winter_labels,
    continuous_factor_bins={"temp_c": FREEZING},
    factor_levels={"site": SITES},
)

print(Balance().evaluate(mixed).factors.filter(pl.col("factor1") == "temp_c"))

# %% [markdown]
# `temp_c × site` is `table` - both are coded and `site`'s codes are its own
# values. `temp_c × haze` is `estimator`, because `haze` was never claimed and so
# is read as measured.
#
# The three regimes reach the `factor_correlation_threshold` at different true
# dependences, so a pair sitting near it is worth reading beside this column. See
# [Binning](../concepts/Binning.md) for the measured numbers. Pass
# `factor_source="coded"` to score codes throughout, which is what every release
# before v1.1 did.

# %%
for source in ("auto", "coded", "values"):
    factors = Balance(factor_source=source).evaluate(mixed).factors
    row = factors.filter((pl.col("factor1") == "temp_c") & (pl.col("factor2") == "haze"))
    print(f"{source:8} mi_value={row['mi_value'][0]:.4f}  scored_as={row['scored_as'][0]}")

# %% [markdown]
# The three answers are all correct; they answer different questions. `coded`
# asks how much a three-way temperature band shares with a binned haze index,
# `values` asks it of the measurements and ignores your cutoff, and `auto`
# honours the cutoff on `temp_c` while reading `haze` as measured.
#
# The selector is named for the two representations, not for binning: `factor_data`
# holds *codes*, and bin indices are only one kind of code - `site` is coded and was
# never cut from anything. Whether a factor was binned is the separate question
# `"auto"` consults, and {attr}`.FactorInfo.is_binned` is where it is recorded.

# %% [markdown]
# ## 8. Keep the record with the data
#
# {meth}`.Metadata.save` carries the encoding into the archive, so a restored
# instance reproduces its codes rather than re-deriving them. Without that, a
# round-trip would silently discard the review work above.

# %%
archive = workspace / "winter.dem"
winter.save(archive)
restored = Metadata.load(archive)

print(f"edges:  {restored.encoding('temp_c').edges}")
print(f"digest: {restored.encoding_digest == winter.encoding_digest}")

# %% [markdown]
# What you pass to {meth}`.Metadata.load` still wins. The archive fills in only
# the factors you say nothing about, so one file serves every set of bins you
# might want from it.

# %% [markdown]
# ## What you learned
#
# - {meth}`.Metadata.encoding` shows the map from values to codes, and its
#   `provenance` says who chose it - `derived` means nobody did
# - `continuous_factor_bins` and `factor_levels` declare the cuts and vocabularies
#   that carry your meaning; infinite outer edges keep them safe to reapply
# - {meth}`.Metadata.accept` fixes a derived placement without moving a code, and
#   records that a person signed it off
# - {meth}`.Metadata.export_encoding` writes reviewable JSON for version control,
#   and passing it back as `encoding=` gives the next dataset the same alphabet
# - {attr}`.Metadata.encoding_digest` says whether two results are comparable, and
#   travels on the result itself
# - A locked cut reports when it stops fitting rather than quietly re-fitting
# - {attr}`~dataeval.bias.Balance.factor_source` honours a declared cut and reads
#   unbinned values where nobody declared one; `scored_as` says which happened

# %% [markdown]
# ## Related concepts
#
# - [Binning](../concepts/Binning.md)
# - [Dataset Bias and Coverage](../concepts/DatasetBias.md)
# - [Acting on Results](../concepts/ActingOnResults.md)
#
# ## See also
#
# ### How-to guides
#
# - [How to reason about factor binning across levels](./h2_bin_factors_by_level.py)
# - [How to apply DataEval's statistical outputs to Metadata](./h2_add_intrinsic_factors.py)
#
# ### Tutorials
#
# - [Identify bias and correlations](./tt_identify_bias.py)
