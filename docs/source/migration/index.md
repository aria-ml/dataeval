# Migration

Version-to-version upgrade guides. Each one covers what changed for existing code, what to
change now, and what disappears in the following release.

DataEval retires a deprecated API one minor release after deprecating it, so a guide's
"removed in" notes are the deadline for migrating off it.

:::{toctree}
:hidden:

v1.2
v1.1
:::

:::{list-table}
:widths: 20 80
:header-rows: 0

- - [](v1.2.md)
  - **v1.1 to v1.2** — the removal of everything v1.1 deprecated (metadata level shims, the
    `selection` module, `per_channel` and the renamed preprocessing helpers, the
    `is_discrete` protocol fallback, `discrete_features` and
    `ClassifierUncertaintyExtractor`), and `SourceIndex` becomes an address:
    `SourceIndex.target` is renamed `key` and its third field becomes `level`.
- - [](v1.1.md)
  - **v1.0 to v1.1** — metadata levels, `View` and operations, image statistic scales, and
    chance-corrected `Balance`. Everything deprecated here is removed in v1.2.0.
:::
