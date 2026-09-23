# JATIC Maturity

The JATIC program grades each product on a maturity scale. This page records the
maturity level DataEval holds, the releases that level applies to, and what the
program requires at each level.

Levels, labels, and requirements on this page follow the JATIC Product Standards
Compliance Checklist, version 1.2.

## Maturity Levels

```{list-table}
:widths: 15 85
:header-rows: 1

* - Level
  - Description
* - 0
  - A new product to JATIC.
* - S
  - The mid development phase of the product.
* - I
  - Completion of technical requirements for a final product.
* - II
  - Product is validated by users.
* - III
  - Product is mature. Validated and adopted by many users.
```

## Current Status

```{list-table}
:widths: 35 65
:header-rows: 0

* - Product maturity level
  - Maturity Level I
* - Approved
  - 2026-04-14, following a program assessment of `v1.0.6`
* - Mature release
  - `v1.0.6`
```

## Maturity by Release

A release carries the maturity label `JATIC Maturity I` only if it was assessed
and approved by the Program Direction Group as a mature release. Other releases
carry no label, including releases made after DataEval reached Maturity Level I.

```{list-table}
:widths: 30 25 45
:header-rows: 1

* - Releases
  - Label
  - Basis
* - `v1.1.0` through `v1.1.3`
  - None
  - Not submitted for mature-release approval
* - `v1.0.6`
  - `JATIC Maturity I`
  - Assessed and approved as a mature release, 2026-04-14
* - `v1.0.5` and earlier
  - None
  - Released before DataEval reached Maturity Level I
* - Pre-releases (`-rcN`, `-aN`)
  - None
  - Not mature releases
```

A mature release is supported with CI and hotfixes for at least six months, or
until a newer mature release is issued, and for as long as the product
documentation lists it as a mature release.

## Requirements by Level

Each level adds requirement areas to the level below it. The table lists which
areas of the Program Standards a product must meet at each level.

```{list-table}
:widths: 30 14 14 14 14 14
:header-rows: 1

* - Requirement area
  - 0
  - S (Sandbox)
  - I
  - II
  - III
* - Software (SR, CR, GR, CS, TR, DSOR, IR)
  - Not required
  - Interoperability only
  - Required
  - Required
  - Required
* - Documentation (DR)
  - Not required
  - Governance (DR-1) only
  - Required
  - Required
  - Required
* - Release (RS)
  - Not required
  - Required
  - Required
  - Required
  - Required
* - Validation (VS)
  - Not required
  - Not required
  - Internal validation (VS-1) only
  - Required
  - Required
* - Deployment (DS)
  - Not required
  - Not required
  - Required
  - Required
  - Required
* - Long-term sustainment (LTPS)
  - Not required
  - Not required
  - Not required
  - Not required
  - Required
```

Several requirements set a different threshold at each level:

```{list-table}
:widths: 28 24 24 24
:header-rows: 1

* - Requirement
  - Level I
  - Level II
  - Level III
* - Reference implementation coverage of T&E functions (TR-5-H-1)
  - At least 50%
  - At least 90%
  - At least 90%
* - Minimum score on every qualitative rubric question, graded 1.0 to 7.0 (VS-1-H-1)
  - 4.0
  - 4.4
  - 5.0
* - External usage (VS-2-H-1)
  - None
  - At least 2 DoD programs
  - At least 5 DoD programs or 5 academic or industry organizations, and publications by at least 2 external groups
* - Deploy updates to user enclaves after a mature release (DS-1-H-1)
  - Within 30 business days
  - Within 10 business days
  - Within 10 business days
* - Required deployment environments (DS-2-H-1)
  - Advana
  - Advana, Iron Bank, Linchpin, SUNet
  - Advana, Iron Bank, Linchpin, SUNet
```

A mature release is approved by the Program Direction Group after an assessment
of its functional requirements and standards compliance (RS-5-H-1, RS-5-H-2).

## Assessment Records

Assessments, requirement exceptions, and the verification cross-reference matrix
for each release are kept in the DataEval directory of the ARiA metadata
repository on JATIC GitLab, `jatic/aria/metarepo`, under `DataEval/assessments/`.
Access requires a JATIC GitLab account.
