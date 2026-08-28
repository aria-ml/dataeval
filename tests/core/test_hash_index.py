"""Tests for Hamming-radius search and grouping over packed perceptual hashes."""

import numpy as np
import pytest

from dataeval.core import _hash_index
from dataeval.core._hash_index import (
    _band_bounds,
    _band_count,
    _band_keys,
    _multi_index,
    _prefer_scan,
    hash_groups,
    hash_neighbors,
    pack_hashes,
)

WIDTHS = [16, 32, 64]  # hex characters: 64-, 128- and 256-bit digests


def brute_pairs(hexes, radius, valid):
    """Independent reference: every pair, compared with Python big integers."""
    values = [int(digest, 16) if digest else 0 for digest in hexes]
    found = {}
    for i in range(len(hexes)):
        if not valid[i]:
            continue
        for j in range(i + 1, len(hexes)):
            if valid[j]:
                distance = bin(values[i] ^ values[j]).count("1")
                if distance <= radius:
                    found[(i, j)] = distance
    return dict(sorted(found.items()))


def brute_groups(pairs, count, valid):
    """Independent reference: connected components of `pairs`, by union-find."""
    parent = list(range(count))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for i, j in pairs:
        a, b = find(i), find(j)
        if a != b:
            parent[a] = b
    components = {}
    for i in range(count):
        if valid[i]:
            components.setdefault(find(i), []).append(i)
    return sorted([sorted(members) for members in components.values() if len(members) > 1])


def random_hexes(rng, count, hex_length, near=0, duplicates=0, entropy_bits=None):
    """Digests with controllable near-duplicate, exact-duplicate and entropy structure."""
    bits = hex_length * 4
    if entropy_bits is not None:
        values = [int(rng.integers(0, 1 << entropy_bits)) for _ in range(max(1, count - duplicates))]
    else:
        values = [
            int.from_bytes(bytes(rng.integers(0, 256, size=hex_length // 2, dtype=np.uint8)), "big")
            for _ in range(max(1, count - near - duplicates))
        ]
        for _ in range(near):
            value = values[int(rng.integers(0, len(values)))]
            for bit in rng.choice(bits, size=int(rng.integers(1, 8)), replace=False):
                value ^= 1 << int(bit)
            values.append(value)
    for _ in range(duplicates):
        values.append(values[int(rng.integers(0, len(values)))])
    rng.shuffle(values)
    return [f"{value:0{hex_length}x}" for value in values[:count]]


@pytest.mark.required
class TestPackHashes:
    @pytest.mark.parametrize("hex_length", WIDTHS)
    def test_round_trips_to_the_same_bits(self, hex_length):
        digest = "a5" * (hex_length // 2)
        codes, valid = pack_hashes([digest])
        packed = int("".join(f"{int(word):016x}" for word in codes[0]), 16)
        assert packed == int(digest, 16) << (codes.shape[1] * 64 - hex_length * 4)
        assert valid.tolist() == [True]

    def test_empty_digest_is_invalid_and_zero(self):
        codes, valid = pack_hashes(["ff00ff00ff00ff00", ""])
        assert valid.tolist() == [True, False]
        assert codes[1].tolist() == [0]

    def test_all_empty_keeps_shape(self):
        codes, valid = pack_hashes(["", ""])
        assert codes.shape == (2, 1)
        assert not valid.any()

    def test_no_hashes(self):
        codes, valid = pack_hashes([])
        assert len(codes) == 0
        assert len(valid) == 0

    def test_mixed_lengths_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            pack_hashes(["ff00", "ff00ff00ff00ff00"])

    def test_non_hex_rejected(self):
        with pytest.raises(ValueError, match="hex-encoded"):
            pack_hashes(["zzzzzzzzzzzzzzzz"])

    def test_matches_core_hamming_distance(self):
        from dataeval.core import hamming_distance

        digests = ["ff00ff00ff00ff00", "ff00ff00ff00ff01", "0123456789abcdef"]
        codes, _ = pack_hashes(digests)
        result = hash_neighbors(codes, radius=64)
        for (i, j), distance in zip(result["pairs"], result["distances"], strict=True):
            assert distance == hamming_distance(digests[i], digests[j])


@pytest.mark.required
class TestBanding:
    @pytest.mark.parametrize(("bits", "bands"), [(64, 7), (64, 8), (256, 7), (13, 7), (64, 1)])
    def test_bounds_partition_every_bit_exactly_once(self, bits, bands):
        bounds = _band_bounds(bits, bands)
        assert len(bounds) == bands
        assert sum(width for _, width in bounds) == bits
        assert [start for start, _ in bounds] == [0, *np.cumsum([w for _, w in bounds])[:-1].tolist()]

    @pytest.mark.parametrize("bits", [64, 128, 256])
    @pytest.mark.parametrize("radius", [0, 1, 6, 12])
    def test_band_width_never_exceeds_a_word(self, bits, radius):
        assert all(width <= 64 for _, width in _band_bounds(bits, _band_count(bits, radius)))

    def test_band_count_satisfies_the_pigeonhole_bound(self):
        for radius in range(0, 20):
            assert _band_count(64, radius) >= radius + 1

    @pytest.mark.parametrize("hex_length", WIDTHS)
    def test_keys_read_the_bits_the_digest_declares(self, hex_length):
        rng = np.random.default_rng(0)
        digests = random_hexes(rng, 32, hex_length)
        codes, _ = pack_hashes(digests)
        bits = hex_length * 4
        for start, width in _band_bounds(bits, _band_count(bits, 6)):
            keys = _band_keys(codes, start, width)
            for row, digest in enumerate(digests):
                expected = (int(digest, 16) >> (bits - start - width)) & ((1 << width) - 1)
                assert int(keys[row]) == expected


@pytest.mark.required
class TestHashNeighbors:
    @pytest.mark.parametrize("radius", [0, 1, 3, 6, 12])
    @pytest.mark.parametrize("hex_length", WIDTHS)
    def test_agrees_with_brute_force(self, radius, hex_length):
        rng = np.random.default_rng(radius * 100 + hex_length)
        digests = random_hexes(rng, 150, hex_length, near=40, duplicates=30)
        codes, valid = pack_hashes(digests)
        result = hash_neighbors(codes, radius=radius, valid=valid)
        found = {(int(i), int(j)): int(d) for (i, j), d in zip(result["pairs"], result["distances"], strict=True)}
        assert found == brute_pairs(digests, radius, valid)

    def test_multi_index_recall_is_exact_against_the_scan(self):
        """The one failure nothing downstream can see: a blocking scheme that drops true pairs."""
        rng = np.random.default_rng(7)
        for trial in range(20):
            hex_length = int(rng.choice(WIDTHS))
            digests = random_hexes(rng, 120, hex_length, near=40, duplicates=20)
            codes, valid = pack_hashes(digests)
            radius = int(rng.integers(1, 13))
            expected = brute_pairs(digests, radius, valid)
            unique = np.unique(codes[valid], axis=0)
            indexed = set()
            for left, right, _distance in _multi_index(unique, hex_length * 4, radius):
                indexed.update(zip(map(int, left), map(int, right), strict=True))
            # Every true pair of *distinct* codes must appear among the candidates the bands
            # proposed; identical codes are handled outside the index.
            lookup = {tuple(row): pos for pos, row in enumerate(unique.tolist())}
            for (i, j), distance in expected.items():
                a, b = lookup[tuple(codes[i].tolist())], lookup[tuple(codes[j].tolist())]
                if a != b:
                    assert (min(a, b), max(a, b)) in indexed, f"trial {trial}: dropped {(i, j)} at {distance}"

    def test_invalid_positions_never_appear(self):
        digests = ["ff00ff00ff00ff00", "", "ff00ff00ff00ff00", ""]
        codes, valid = pack_hashes(digests)
        result = hash_neighbors(codes, radius=6, valid=valid)
        assert result["pairs"].tolist() == [[0, 2]]

    def test_pairs_are_sorted_and_canonical(self):
        rng = np.random.default_rng(1)
        codes, valid = pack_hashes(random_hexes(rng, 80, 16, near=40, duplicates=20))
        pairs = hash_neighbors(codes, radius=8, valid=valid)["pairs"]
        assert (pairs[:, 0] < pairs[:, 1]).all()
        as_list = pairs.tolist()
        assert as_list == sorted(as_list)
        assert len({tuple(p) for p in as_list}) == len(as_list)

    @pytest.mark.parametrize("count", [0, 1])
    def test_degenerate_inputs(self, count):
        codes, valid = pack_hashes(["ff00ff00ff00ff00"] * count)
        result = hash_neighbors(codes, radius=6, valid=valid)
        assert result["pairs"].shape == (0, 2)
        assert result["distances"].shape == (0,)

    def test_negative_radius_rejected(self):
        codes, _ = pack_hashes(["ff00ff00ff00ff00"])
        with pytest.raises(ValueError, match="non-negative"):
            hash_neighbors(codes, radius=-1)

    def test_wrong_shape_rejected(self):
        with pytest.raises(ValueError, match="2D"):
            hash_neighbors(np.zeros(4, dtype=np.uint64), radius=1)

    def test_mismatched_valid_rejected(self):
        codes, _ = pack_hashes(["ff00ff00ff00ff00", "0123456789abcdef"])
        with pytest.raises(ValueError, match="must agree"):
            hash_neighbors(codes, radius=1, valid=np.array([True]))

    def test_refuses_a_quadratic_answer_rather_than_exhausting_memory(self):
        """A thousand identical hashes is half a million pairs; the refusal names the way out."""
        codes, _ = pack_hashes(["deadbeefcafef00d"] * 1000)
        with pytest.raises(ValueError, match="hash_groups"):
            hash_neighbors(codes, radius=0, max_pairs=1000)

    def test_max_pairs_is_not_a_truncation(self):
        codes, _ = pack_hashes(["deadbeefcafef00d"] * 40)
        allowed = hash_neighbors(codes, radius=0, max_pairs=40 * 39 // 2)
        assert len(allowed["pairs"]) == 40 * 39 // 2


@pytest.mark.required
class TestHashGroups:
    @pytest.mark.parametrize("radius", [0, 1, 3, 6, 12])
    @pytest.mark.parametrize("hex_length", WIDTHS)
    def test_agrees_with_brute_force_components(self, radius, hex_length):
        rng = np.random.default_rng(radius * 100 + hex_length)
        digests = random_hexes(rng, 150, hex_length, near=40, duplicates=30)
        codes, valid = pack_hashes(digests)
        result = hash_groups(codes, radius=radius, valid=valid)
        expected = brute_groups(list(brute_pairs(digests, radius, valid)), len(digests), valid)
        assert sorted([sorted(int(x) for x in g) for g in result["groups"]]) == expected

    @pytest.mark.parametrize("radius", [0, 4, 10])
    def test_labels_agree_with_groups(self, radius):
        rng = np.random.default_rng(radius)
        codes, valid = pack_hashes(random_hexes(rng, 120, 16, near=40, duplicates=25))
        result = hash_groups(codes, radius=radius, valid=valid)
        labels = result["labels"]
        for index, group in enumerate(result["groups"]):
            assert (labels[group] == index).all()
        assert (labels == -1).sum() == len(labels) - sum(len(g) for g in result["groups"])

    def test_groups_are_sorted_within_and_between(self):
        rng = np.random.default_rng(2)
        codes, valid = pack_hashes(random_hexes(rng, 100, 16, near=40, duplicates=20))
        groups = hash_groups(codes, radius=6, valid=valid)["groups"]
        for group in groups:
            assert group.tolist() == sorted(group.tolist())
        assert [int(g[0]) for g in groups] == sorted(int(g[0]) for g in groups)

    def test_invalid_positions_are_labelled_out(self):
        codes, valid = pack_hashes(["ff00ff00ff00ff00", "", "ff00ff00ff00ff00"])
        result = hash_groups(codes, radius=6, valid=valid)
        assert [g.tolist() for g in result["groups"]] == [[0, 2]]
        assert result["labels"].tolist() == [0, -1, 0]

    def test_grouping_is_transitive_where_pairing_is_not(self):
        """Two hops of 4 bits are one group at radius 4, though the ends are 8 bits apart."""
        base = 0
        digests = [f"{base:016x}", f"{base ^ 0b1111:016x}", f"{base ^ 0b11111111:016x}"]
        codes, valid = pack_hashes(digests)
        assert len(hash_neighbors(codes, radius=4, valid=valid)["pairs"]) == 2
        assert [g.tolist() for g in hash_groups(codes, radius=4, valid=valid)["groups"]] == [[0, 1, 2]]

    def test_a_large_identical_set_is_one_group_not_a_pair_explosion(self):
        """The case that motivated hash_groups: pairs would be 4.5 million rows, groups is one."""
        rng = np.random.default_rng(5)
        digests = random_hexes(rng, 2000, 16) + ["deadbeefcafef00d"] * 3000
        codes, valid = pack_hashes(digests)
        result = hash_groups(codes, radius=6, valid=valid)
        assert max(len(g) for g in result["groups"]) == 3000

    @pytest.mark.parametrize("count", [0, 1])
    def test_degenerate_inputs(self, count):
        codes, valid = pack_hashes(["ff00ff00ff00ff00"] * count)
        result = hash_groups(codes, radius=6, valid=valid)
        assert result["groups"] == []
        assert len(result["labels"]) == count

    def test_negative_radius_rejected(self):
        codes, _ = pack_hashes(["ff00ff00ff00ff00"])
        with pytest.raises(ValueError, match="non-negative"):
            hash_groups(codes, radius=-1)


@pytest.mark.required
class TestLowEntropyRegime:
    """Digests spanning few bits put much of the corpus in one band, which pruning must survive.

    This is the FMV case, not a synthetic one: consecutive frames of a static camera differ by a
    handful of bits, so their hashes concentrate rather than spread.
    """

    @pytest.mark.parametrize("radius", [1, 6, 12])
    def test_grouping_still_matches_brute_force(self, radius):
        rng = np.random.default_rng(radius)
        digests = random_hexes(rng, 200, 16, duplicates=40, entropy_bits=10)
        codes, valid = pack_hashes(digests)
        result = hash_groups(codes, radius=radius, valid=valid)
        expected = brute_groups(list(brute_pairs(digests, radius, valid)), len(digests), valid)
        assert sorted([sorted(int(x) for x in g) for g in result["groups"]]) == expected

    def test_pruning_does_not_change_the_answer(self):
        """hash_groups prunes candidates hash_neighbors verifies; both must see one truth."""
        rng = np.random.default_rng(9)
        digests = random_hexes(rng, 150, 16, duplicates=20, entropy_bits=9)
        codes, valid = pack_hashes(digests)
        pairs = hash_neighbors(codes, radius=5, valid=valid, max_pairs=10**8)["pairs"]
        grouped = hash_groups(codes, radius=5, valid=valid)["groups"]
        expected = brute_groups([tuple(p) for p in pairs.tolist()], len(digests), valid)
        assert sorted([sorted(int(x) for x in g) for g in grouped]) == expected


@pytest.mark.required
class TestStrategyAgreement:
    """Two search strategies, one answer.

    Which one runs is a cost decision, so nothing downstream can see it -- and nothing
    downstream can see it go wrong either. Every other test here happens to exercise whichever
    strategy its size and radius select; these pin both against each other on the same input.
    """

    @pytest.mark.parametrize("radius", [1, 3, 6, 12])
    @pytest.mark.parametrize("hex_length", WIDTHS)
    def test_the_index_and_the_scan_return_the_same_pairs(self, monkeypatch, radius, hex_length):
        rng = np.random.default_rng(radius * 100 + hex_length)
        digests = random_hexes(rng, 150, hex_length, near=40, duplicates=30)
        codes, valid = pack_hashes(digests)

        monkeypatch.setattr(_hash_index, "_prefer_scan", lambda *_: True)
        scanned = hash_neighbors(codes, radius=radius, valid=valid)
        scanned_groups = hash_groups(codes, radius=radius, valid=valid)

        monkeypatch.setattr(_hash_index, "_prefer_scan", lambda *_: False)
        indexed = hash_neighbors(codes, radius=radius, valid=valid)
        indexed_groups = hash_groups(codes, radius=radius, valid=valid)

        assert indexed["pairs"].tolist() == scanned["pairs"].tolist()
        assert indexed["distances"].tolist() == scanned["distances"].tolist()
        assert [g.tolist() for g in indexed_groups["groups"]] == [g.tolist() for g in scanned_groups["groups"]]

    def test_a_corpus_past_the_threshold_uses_the_index_and_still_agrees(self, monkeypatch):
        """At the size the index is actually chosen for, not just the size that fits a scan."""
        rng = np.random.default_rng(21)
        digests = random_hexes(rng, 4200, 16, near=400, duplicates=200)
        codes, valid = pack_hashes(digests)
        assert not _prefer_scan(len(np.unique(codes[valid], axis=0)), 64, 6)

        indexed = hash_neighbors(codes, radius=6, valid=valid, max_pairs=10**8)
        monkeypatch.setattr(_hash_index, "_prefer_scan", lambda *_: True)
        scanned = hash_neighbors(codes, radius=6, valid=valid, max_pairs=10**8)

        assert len(indexed["pairs"])
        assert indexed["pairs"].tolist() == scanned["pairs"].tolist()
        assert indexed["distances"].tolist() == scanned["distances"].tolist()

    def test_a_radius_spanning_the_digest_links_everything(self):
        """No two codes can differ in more bits than they hold, so the answer is one group."""
        rng = np.random.default_rng(22)
        digests = random_hexes(rng, 60, 16)
        codes, valid = pack_hashes(digests + [""])
        result = hash_groups(codes, radius=64, valid=valid)
        assert [g.tolist() for g in result["groups"]] == [list(range(60))]
        assert result["labels"].tolist() == [0] * 60 + [-1]

    def test_connectivity_survives_being_flushed_in_pieces(self, monkeypatch):
        """Edges are folded into labels and dropped in batches; the batch size must not show."""
        rng = np.random.default_rng(23)
        digests = random_hexes(rng, 200, 16, duplicates=40, entropy_bits=10)
        codes, valid = pack_hashes(digests)
        expected = [g.tolist() for g in hash_groups(codes, radius=6, valid=valid)["groups"]]
        for flush in (1, 7, 100):
            monkeypatch.setattr(_hash_index, "_EDGE_FLUSH", flush)
            assert [g.tolist() for g in hash_groups(codes, radius=6, valid=valid)["groups"]] == expected


@pytest.mark.required
class TestBitsArgument:
    """The significant width of a digest that does not fill its last word."""

    @pytest.mark.parametrize("hex_length", [12, 20, 30])
    def test_a_padded_digest_is_searched_on_its_own_bits(self, hex_length):
        rng = np.random.default_rng(hex_length)
        digests = random_hexes(rng, 120, hex_length, near=30, duplicates=20)
        codes, valid = pack_hashes(digests)
        assert codes.shape[1] * 64 > hex_length * 4  # there really is padding to ignore
        result = hash_neighbors(codes, radius=5, valid=valid, bits=hex_length * 4)
        found = {(int(i), int(j)): int(d) for (i, j), d in zip(result["pairs"], result["distances"], strict=True)}
        assert found == brute_pairs(digests, 5, valid)

    @pytest.mark.parametrize("bits", [0, -1, 65, 4096])
    def test_a_width_the_codes_cannot_hold_is_rejected(self, bits):
        codes, _ = pack_hashes(["ff00ff00ff00ff00", "0123456789abcdef"])
        with pytest.raises(ValueError, match="bits must be between"):
            hash_neighbors(codes, radius=1, bits=bits)
        with pytest.raises(ValueError, match="bits must be between"):
            hash_groups(codes, radius=1, bits=bits)

    def test_codes_without_a_word_are_rejected(self):
        with pytest.raises(ValueError, match="W >= 1"):
            hash_neighbors(np.zeros((3, 0), dtype=np.uint64), radius=1)

    def test_valid_must_be_one_dimensional(self):
        codes, _ = pack_hashes(["ff00ff00ff00ff00", "0123456789abcdef"])
        with pytest.raises(ValueError, match="1D array"):
            hash_neighbors(codes, radius=1, valid=np.ones((2, 1), dtype=np.bool_))
