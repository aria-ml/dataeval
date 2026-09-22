# Security Policy

This document describes how to report vulnerabilities in DataEval and how the
maintainers record scanner findings that cannot be resolved in this repository.
The accepted-findings section is the project's response to JATIC SDP
requirement DSOR-3-H-2.

## Reporting a vulnerability

Please do **not** open a public issue for security vulnerabilities. Email
<dataeval@ariacoustics.com> with a description of the issue and its impact,
steps to reproduce, and the affected versions or container tags.

For non-security bugs, follow the regular process in
[CONTRIBUTING.md](CONTRIBUTING.md).

## Container image findings

Published images are scanned by the GitLab Container Scanning analyzer (Trivy)
in the `container_scanning` job; see
[.gitlab/ci/container.yml](.gitlab/ci/container.yml). The gate is set at
`CS_SEVERITY_THRESHOLD: HIGH`, matching CS-2-H-2. All three variants carry zero
HIGH and zero CRITICAL findings.

### Accepted MEDIUM findings

DSOR-3-H-2 requires no MEDIUM findings, which is stricter than CS-2-H-2. The
findings below are present in every published variant and are accepted rather
than resolved. Ubuntu has published no patch for any of them, on any release.

| CVE | Package | Vendor status | Fixed version | Re-evaluate |
| --- | --- | --- | --- | --- |
| CVE-2026-18374 | `libc6`, `libc-bin` | affected | none published | 2026-12-22 |
| CVE-2026-18477 | `tar` | affected | none published | 2026-12-22 |
| CVE-2026-18508 | `tar` | affected | none published | 2026-12-22 |
| CVE-2026-85091 | `zlib1g` | affected | none published | 2026-12-22 |

Justification, verified 2026-09-22 against `ubuntu:24.04` and `ubuntu:26.04`:

- **glibc and zlib cannot be removed.** Every binary in the image links
  `libc.so.6`, and `libz.so.1` is linked by the `opencv-python-headless`,
  `scipy` and `pillow` wheels.
- **tar could be removed, but at a cost that outweighs the finding.** Nothing
  in the runtime calls it; Python's `tarfile` is pure Python over zlib. It is
  marked `Essential: yes` and `dpkg` depends on it, so removal requires
  `--force-remove-essential --force-depends` and leaves `apt` and `dpkg`
  broken for anyone building an image *from* this one. Two accepted MEDIUM
  findings are the better trade against breaking the documented downstream
  workflow.
- **A newer base does not help.** All four CVEs carry the same `affected`
  status on Ubuntu 26.04, which additionally introduces 20 unpatched MEDIUM
  findings in `rust-coreutils`. Ubuntu 24.04 remains the lower-risk base.
- **Self-clearing.** The `apt-get upgrade` layer in the build and prod stages
  picks up the patched packages on the next image build once Ubuntu publishes
  the advisories. No change to this repository will be required.

The deviation from DSOR-3-H-2 is escalated through the routes the standard
publishes: a JATIC Support "help" issue, the team's Program Representative or
the Program's Scrum-of-Scrums, or an `internal-docs` issue labeled "Standards".

### Resolved findings

- `pip` (CVE-2026-13346, CVE-2026-3219, CVE-2026-6357, CVE-2026-8643). The pip
  bundled with the standalone interpreter was unreachable at runtime, because
  uv builds the virtual environment directly and never invokes it. It is now
  removed during the build stage, along with `ensurepip` and the console
  scripts, so that no vulnerable copy remains on disk. See
  [docker/Dockerfile.j2](docker/Dockerfile.j2).
