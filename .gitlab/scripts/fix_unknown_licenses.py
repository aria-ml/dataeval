#!/usr/bin/env python3
"""Inject SPDX licenses into CycloneDX SBOM for packages GitLab marks as unknown.

GitLab's gemnasium-python analyzer does not populate component licenses in the
SBOM. Licenses are enriched server-side via GitLab's license DB after upload.
For packages the DB does not recognize, the UI shows "unknown".

When a component already carries a `licenses` field in the SBOM, GitLab uses
it instead of querying its DB. This script populates that field for known
unknowns.

Run after `/analyzer run` and before artifact upload (CI after_script).
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

# Package name -> SPDX license ID. Evidence is the upstream LICENSE file.
LICENSE_OVERRIDES: dict[str, str] = {
    "llvmlite": "BSD-2-Clause",  # https://github.com/numba/llvmlite/blob/main/LICENSE
    "mpmath": "BSD-3-Clause",  # https://github.com/mpmath/mpmath/blob/master/LICENSE
    "numba": "BSD-2-Clause",  # https://github.com/numba/numba/blob/main/LICENSE
    "numpy": "BSD-3-Clause",  # https://github.com/numpy/numpy/blob/main/LICENSE.txt
    "packaging": "BSD-2-Clause",  # https://github.com/pypa/packaging/blob/main/LICENSE.BSD
    "sympy": "BSD-3-Clause",  # https://github.com/sympy/sympy/blob/master/LICENSE
    "torchvision": "BSD-3-Clause",  # https://github.com/pytorch/vision/blob/main/LICENSE
    "xxhash": "BSD-2-Clause",  # https://github.com/ifduyue/python-xxhash/blob/master/LICENSE
}


def patch_sbom(path: Path) -> int:
    data = json.loads(path.read_text())
    components = data.get("components", [])
    patched = 0
    for comp in components:
        name = (comp.get("name") or "").lower()
        spdx = LICENSE_OVERRIDES.get(name)
        if spdx is None:
            continue
        if comp.get("licenses"):
            continue  # already has license info, skip
        comp["licenses"] = [{"license": {"id": spdx}}]
        patched += 1
        print(f"  patched {name} -> {spdx}")
    if patched:
        path.write_text(json.dumps(data, indent=2))
    return patched


def main() -> int:
    sboms = glob.glob("**/gl-sbom-*.cdx.json", recursive=True)
    if not sboms:
        print("no gl-sbom-*.cdx.json found", file=sys.stderr)
        return 1
    total = 0
    for s in sboms:
        print(f"patching {s}")
        total += patch_sbom(Path(s))
    print(f"total components patched: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
