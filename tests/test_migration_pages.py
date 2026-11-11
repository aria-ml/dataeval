"""Check that the v1.3 migration guide exists and sits in the toctree of the migration index."""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.infrastructure

MIGRATION = Path(__file__).resolve().parent.parent / "docs" / "source" / "migration"
TOCTREE = re.compile(r"^:::\{toctree\}\n(.*?)^:::", re.DOTALL | re.MULTILINE)


def _toctree_entries() -> list[str]:
    (block,) = TOCTREE.findall((MIGRATION / "index.md").read_text())
    return [line.strip() for line in block.splitlines() if line.strip() and not line.startswith(":")]


@pytest.mark.required
def test_v1_3_page_in_index():
    assert (MIGRATION / "v1.3.md").is_file()
    assert "v1.3" in _toctree_entries()
