"""Run the code on the extrinsic factors page, so that a typo cannot leave a snippet that raises or reads a field wrong.

Nothing else executes ``ExtrinsicFactors.md``. Sphinx renders its fences and no notebook runs them. The docs job has no
ffmpeg, so the byte-level helpers run on packets built by hand and the calls to ``ffmpeg`` and ``ffprobe`` are replaced
by functions that return a constructed stream. The ``ffprobe`` snippets run against a clip that ``ffmpeg`` writes, and
skip when ``ffmpeg`` is not installed.
"""

import ast
import math
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

PAGE = Path(__file__).resolve().parents[2] / "docs" / "source" / "concepts" / "ExtrinsicFactors.md"
FENCE = re.compile(r"^```python\n(.*?)^```", re.DOTALL | re.MULTILINE)
BLOCKS = FENCE.findall(PAGE.read_text())

KEY = bytes.fromhex("060e2b34020b01010e01030101000000")
"""The ST 0601 universal key that the page splits packets on."""


def _block(marker: str) -> str:
    """Return the one code block of the page that contains ``marker``."""
    found = [block for block in BLOCKS if marker in block]
    assert len(found) == 1, f"expected one block containing {marker!r}, found {len(found)}"
    return found[0]


def _run(block: str, namespace: dict[str, Any]) -> dict[str, Any]:
    exec(compile(block, PAGE.name, "exec"), namespace)
    return namespace


def _ber(length: int) -> bytes:
    """Encode a BER length."""
    if length < 0x80:
        return bytes([length])
    size = (length.bit_length() + 7) // 8
    return bytes([0x80 | size]) + length.to_bytes(size, "big")


def _value(*items: tuple[int, bytes]) -> bytes:
    """Build the value of an ST 0601 packet from (item number, bytes) pairs, each below 128."""
    return b"".join(bytes([number]) + _ber(len(raw)) + raw for number, raw in items)


def _latitude(degrees: float) -> bytes:
    """Encode a latitude as the int32 that item 13 holds."""
    return round(degrees * 4294967294 / 180).to_bytes(4, "big", signed=True)


def _stream(*values: bytes, key: bytes = KEY) -> bytes:
    """Frame each packet value with the universal key and a BER length."""
    return b"".join(key + _ber(len(value)) + value for value in values)


def _fake_subprocess(stdout: bytes) -> SimpleNamespace:
    """Stand in for ``subprocess`` so a block that calls ``ffmpeg`` reads ``stdout`` instead."""
    return SimpleNamespace(run=lambda *args, **kwargs: SimpleNamespace(stdout=stdout))


@pytest.mark.required
def test_the_extractor_finds_the_blocks_of_the_page():
    """Guard against the fence pattern silently matching nothing."""
    assert len(BLOCKS) >= 8, f"found {len(BLOCKS)} python blocks; the fence pattern is likely broken"


@pytest.mark.required
@pytest.mark.parametrize("index", range(len(BLOCKS)))
def test_every_python_block_parses(index: int):
    ast.parse(BLOCKS[index])


@pytest.mark.required
def test_the_items_helper_decodes_sensor_latitude():
    """The platform section's helpers split a packet and decode item 13 as the page says."""
    packet = _value((2, (1_700_000_000_000_000).to_bytes(8, "big")), (13, _latitude(35.25)), (14, b"\x00\x00\x00\x01"))
    namespace = _run(_block("def ber("), {"packet": packet})
    assert list(namespace["fields"]) == [2, 13, 14]
    assert int.from_bytes(namespace["fields"][2], "big") == 1_700_000_000_000_000
    assert math.isclose(namespace["lat"], 35.25, abs_tol=1e-6)


@pytest.mark.required
@pytest.mark.parametrize("packet", [_value((2, b"\x00" * 8)), _value((13, b"\x80\x00\x00\x00"))])
def test_latitude_is_none_when_the_item_is_absent_or_flagged_out_of_range(packet: bytes):
    assert _run(_block("def ber("), {"packet": packet})["lat"] is None


@pytest.mark.required
def test_a_long_form_length_does_not_shift_the_items_after_it():
    """An item longer than 127 bytes carries a multi-byte BER length, and the next item must still be found."""
    packet = _value((1, b"\xaa" * 300), (13, _latitude(-12.5)))
    namespace = _run(_block("def ber("), {"packet": packet})
    assert len(namespace["fields"][1]) == 300
    assert math.isclose(namespace["lat"], -12.5, abs_tol=1e-6)


@pytest.mark.required
def test_the_packet_splitter_reads_item_2_of_each_0601_packet():
    """The loop skips a packet with another key and reads the timestamp of the rest."""
    stamps = [1_700_000_000_000_000, 1_700_000_000_066_666]
    stream = (
        _stream(_value((2, stamps[0].to_bytes(8, "big")), (13, _latitude(1.0))))
        + _stream(_value((2, (5).to_bytes(8, "big"))), key=bytes(16))
        + _stream(_value((2, stamps[1].to_bytes(8, "big"))))
    )
    namespace = _run(_block("def ber("), {"packet": b""})
    namespace.update(subprocess=_fake_subprocess(stream), video="clip.mp4")
    assert _run(_block("ST 0601 universal key"), namespace)["timestamps_us"] == stamps


@pytest.mark.required
def test_the_sei_scan_returns_the_12_bytes_after_each_identifier():
    first, second = bytes(range(12)), bytes(range(12, 24))
    raw = b"\x00\x00\x01" + b"MISPmicrosectime" + first + b"\x65\x88" + b"MISPmicrosectime" + second + b"\x00"
    namespace = _run(_block("MISPmicrosectime"), {"subprocess": _fake_subprocess(raw), "video": "clip.mp4"})
    assert namespace["sei"] == [first, second]


@pytest.mark.required
def test_sorting_by_pts_puts_decode_order_timestamps_in_display_order():
    """Decode order puts the P frame before the B frame it is predicted from, so the pairs sort by PTS."""
    packets = [{"pts_time": "0.2"}, {"pts_time": "0.0"}, {"pts_time": "0.1"}]
    block = _block("in_display_order")
    namespace = {"ffprobe": lambda *args: packets, "video": "clip.mp4", "sei": [b"c", b"a", b"b"]}
    assert _run(block, namespace)["in_display_order"] == [(0.0, b"a"), (0.1, b"b"), (0.2, b"c")]
    with pytest.raises(ValueError, match="shorter"):
        _run(block, {**namespace, "sei": [b"a", b"b"]})


@pytest.mark.ffmpeg
@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs ffmpeg and ffprobe")
def test_the_ffprobe_snippets_read_the_properties_of_a_clip(tmp_path: Path):
    """The stream and frame snippets run against a clip with an I frame every fifth frame and no B frames."""
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10", "-t", "2"]
        + ["-c:v", "mpeg4", "-g", "5", "-bf", "0", "-pix_fmt", "yuv420p", str(clip)],
        check=True,
    )
    namespace = _run(_block("def ffprobe("), {"video": str(clip)})
    _run(_block("stream=codec_name"), namespace)
    assert (namespace["codec"], namespace["pix_fmt"], namespace["interlaced"]) == ("mpeg4", "yuv420p", False)
    _run(_block("frame=pict_type,pkt_size"), namespace)
    assert namespace["frame_type"][:6] == ["I", "P", "P", "P", "P", "I"]
    assert len(namespace["bits_per_frame"]) == 20
    assert all(bits > 0 for bits in namespace["bits_per_frame"])
    assert namespace["avg_gop"] == 5.0
    assert namespace["variable_gop"] is False
