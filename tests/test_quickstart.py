"""Run the quickstart's code, so that a renamed argument cannot leave the first example raising.

Nothing else executes ``quickstart.md``. Sphinx renders its fences, and ``test_doc_cross_references``
reads the roles around them, so ``Duplicates(cluster_threshold=...)`` stayed in the guide after the
argument became ``cluster_sensitivity`` and every reader who copied it got a ``TypeError``.

The CIFAR-10 step downloads data, so it is parsed but not run. The evaluation step runs unchanged
against the Step 1 example dataset class and a few synthetic images, three of which are near
copies of earlier ones.
"""

import ast
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dataeval.quality import DuplicatesOutput

pytestmark = pytest.mark.infrastructure

QUICKSTART = Path(__file__).resolve().parent.parent / "docs" / "source" / "getting-started" / "quickstart.md"
FENCE = re.compile(r"^```python\n(.*?)^```", re.DOTALL | re.MULTILINE)

BLOCKS = FENCE.findall(QUICKSTART.read_text())
RUNNABLE = [i for i, block in enumerate(BLOCKS) if "download=True" not in block and "Duplicates(" not in block]


def _run(block: str, namespace: dict[str, Any]) -> dict[str, Any]:
    exec(compile(block, QUICKSTART.name, "exec"), namespace)
    return namespace


@pytest.mark.required
def test_the_extractor_finds_the_quickstart_blocks():
    """Guard against the fence pattern silently matching nothing."""
    assert len(BLOCKS) >= 4, f"found {len(BLOCKS)} python blocks; the fence pattern is likely broken"
    assert any("Duplicates(" in block for block in BLOCKS), "the evaluation block is missing"


@pytest.mark.required
@pytest.mark.parametrize("index", range(len(BLOCKS)))
def test_every_python_block_parses(index: int):
    ast.parse(BLOCKS[index])


@pytest.mark.required
@pytest.mark.parametrize("index", RUNNABLE)
def test_the_dataset_blocks_run(index: int):
    """The blocks that define example datasets run as written, with no download."""
    _run(BLOCKS[index], {})


@pytest.mark.required
def test_the_evaluation_block_finds_near_duplicates():
    """The first evaluation in the guide runs as written and reports what it says it reports."""
    define = next(block for block in BLOCKS if "class MyImageClassificationDataset" in block)
    evaluate = next(block for block in BLOCKS if "Duplicates(" in block)

    rng = np.random.default_rng(0)
    images = [rng.random((3, 32, 32), dtype=np.float32) for _ in range(300)]
    copies = ((10, 200), (40, 250), (77, 299))
    for source, target in copies:
        noisy = images[source] + rng.normal(0, 0.002, images[source].shape)
        images[target] = np.clip(noisy, 0, 1).astype(np.float32)
    labels = [i % 10 for i in range(300)]

    namespace = _run(define, {})
    namespace["dataset"] = namespace["MyImageClassificationDataset"](images, labels, 10)
    result = _run(evaluate, namespace)["result"]

    assert isinstance(result, DuplicatesOutput)
    found = {tuple(sorted(indices)) for indices, _ in result.near}
    assert set(copies) <= found, f"the injected near copies {copies} were not all reported: {sorted(found)}"
