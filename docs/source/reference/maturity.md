# JATIC Maturity

The JATIC program grades each product on a maturity scale. This page records the
maturity level DataEval holds, the releases that level applies to, and what the
program requires at each level.

Requirements on this page follow draft version 1.3 of the JATIC product
standards: the Software Development Plan (SDP) Requirements v1.3 draft for
software, and the program's draft release, validation, and deployment standards.
Version 1.3 has not been released. A release's maturity is governed by the
standards in force when that release was assessed, not by this draft.

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
  - Product is validated.
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

The SDP Requirements v1.3 draft maps each software requirement area to the
maturity levels at which it applies. At Maturity S, the program requires only
the interoperability standards.

```{list-table}
:widths: 25 15 15 15 15 15
:header-rows: 1

* - Software area
  - 0
  - S
  - I
  - II
  - III
* - General software (SR)
  - Not required
  - Partial
  - Required
  - Required
  - Required
* - Python coding (CR)
  - Not required
  - Partial
  - Required
  - Required
  - Required
* - Interoperability (IR)
  - Not required
  - Required (IR-1, IR-2)
  - Required
  - Required
  - Required
* - Containerization (CS)
  - Not required
  - Partial
  - Required
  - Required
  - Required
* - Testing (TR)
  - Not required
  - Partial
  - Required
  - Required
  - Required
* - DevSecOps (DSOR)
  - Not required
  - Partial
  - Required
  - Required
  - Required
* - GitLab and branching (GR)
  - Not required
  - Partial
  - Required
  - Required
  - Required
```

The remaining product standards apply as follows:

```{list-table}
:widths: 30 14 14 14 14 14
:header-rows: 1

* - Standard
  - 0
  - S
  - I
  - II
  - III
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
  - Internal qualitative validation (VS-1) only
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
* - Public T&E capabilities integrated into CheckMAITE (TR-5-H-1)
  - At least 50%
  - At least 90%
  - At least 90%
* - Minimum score on every qualitative rubric question, scored out of 5.0 (VS-1-H-1)
  - 3.0
  - 3.5
  - 4.0
* - External usage (VS-2-H-1)
  - None
  - At least 2 DoD programs
  - At least 5 DoD programs or 5 academic or industry organizations, and academic
    publications by at least 2 groups outside the developer's organization
* - Demonstrated use on CDAO data holdings in a meaningful scenario (VS-3)
  - Not required
  - Required
  - Required
* - Documented validation of product features (VS-4)
  - Not required
  - Required
  - Required
* - Start deploying updates after a mature release (DS-1-H-1)
  - Within 30 business days
  - Within 10 business days
  - Within 10 business days
* - Required deployment environments (DS-2-H-1)
  - WDP (formerly Advana)
  - WDP, COSMOS, SUNet
  - WDP, COSMOS, SUNet, Iron Bank, Linchpin
```

A mature release is approved by the Program Direction Group after an assessment
of its functional requirements and standards compliance (RS-5-H-1, RS-5-H-2).

## Assessment Records

Assessments, requirement exceptions, and the verification cross-reference matrix
for each release are kept in the DataEval directory of the ARiA metadata
repository on JATIC GitLab, `jatic/aria/metarepo`, under `DataEval/assessments/`.
Access requires a JATIC GitLab account.
