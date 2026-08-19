"""The protocols a caller implements to hand DataEval its own container."""

from typing import Any

import numpy as np
import pytest

from dataeval import protocols


def _mro(protocol: Any) -> tuple[type, ...]:
    """Read a protocol class's MRO without pyright objecting to the dunder."""
    return protocol.__mro__


@pytest.mark.required
class TestTheMetadataProtocolPair:
    """Codes and measured values are alternative representations, not layers.

    ``runtime_checkable`` tests for the *presence* of a member and never its type, so if
    both protocols named the member ``factor_data`` every coded container would satisfy
    the numeric one too and ``isinstance`` could not tell them apart. ``factor_values`` is
    what discriminates, and it is why the two are not related by inheritance.
    """

    class Coded:
        factor_names = ["a"]
        factor_data = np.zeros((3, 1), dtype=np.int64)
        class_labels = np.zeros(3, dtype=np.intp)
        is_binned = [True]

    class Valued:
        factor_names = ["a"]
        factor_values = np.zeros((3, 1))
        class_labels = np.zeros(3, dtype=np.intp)

    class Both(Coded, Valued): ...

    class Labels:
        class_labels = np.zeros(3, dtype=np.intp)

    @pytest.mark.parametrize(
        ("container", "coded", "valued", "labels"),
        [
            ("Coded", True, False, True),
            ("Valued", False, True, True),
            ("Both", True, True, True),
            ("Labels", False, False, True),
        ],
    )
    def test_dispatch_is_sound_in_both_directions(self, container, coded, valued, labels):
        instance = getattr(self, container)()

        assert isinstance(instance, protocols.CodedMetadataLike) is coded
        assert isinstance(instance, protocols.ValuedMetadataLike) is valued
        assert isinstance(instance, protocols.LabelsLike) is labels
        assert isinstance(instance, protocols.AnyMetadataLike) is (coded or valued)

    def test_metadata_like_is_a_plain_alias(self):
        """A union would keep only the annotation usage; this module's own doctest
        subclasses it, and ``class Mine(A | B)`` is not a class statement that works."""
        assert protocols.MetadataLike is protocols.CodedMetadataLike

        class Mine(protocols.MetadataLike):
            # Properties, as the protocol declares them and as this module's own doctest
            # writes them; plain class attributes would shadow rather than implement.
            @property
            def factor_names(self):
                return ["a"]

            @property
            def factor_data(self):
                return np.zeros((3, 1), dtype=np.int64)

            @property
            def class_labels(self):
                return np.zeros(3, dtype=np.intp)

            @property
            def is_binned(self):
                return [True]

        # `issubclass` is unavailable for a protocol with data members, so the base-class
        # usage is confirmed the way a caller would actually feel it: it subclasses, and
        # what it produces is recognised.
        assert isinstance(Mine(), protocols.CodedMetadataLike)

    def test_the_concrete_metadata_carries_both_channels(self):
        from dataeval import Metadata

        md = Metadata.from_factors({"t": np.arange(9.0)}, class_labels=np.zeros(9, dtype=int))
        assert isinstance(md, protocols.CodedMetadataLike)
        assert isinstance(md, protocols.ValuedMetadataLike)

    def test_both_metadata_protocols_extend_the_labels_one(self):
        """Labels are the part they do not differ on, so they are declared once.

        Nominal as well as structural: a container satisfying either metadata protocol
        already satisfied ``LabelsLike`` by shape, and saying so in the hierarchy is what
        keeps ``class_labels`` from being restated in three places and drifting.
        """
        # Read off the MRO: ``issubclass`` is refused outright for a protocol carrying
        # non-method members, which both of these do.
        assert protocols.LabelsLike in _mro(protocols.CodedMetadataLike)
        assert protocols.LabelsLike in _mro(protocols.ValuedMetadataLike)
        # Siblings, not layers. Sharing a floor is what makes that legible rather than
        # tempting: neither representation is a special case of the other.
        assert protocols.CodedMetadataLike not in _mro(protocols.ValuedMetadataLike)
        assert protocols.ValuedMetadataLike not in _mro(protocols.CodedMetadataLike)

    @pytest.mark.parametrize(
        ("protocol", "members"),
        [
            (
                "CodedMetadataLike",
                {
                    "factor_names": ["a"],
                    "factor_data": np.zeros((3, 1), dtype=np.int64),
                    "class_labels": np.zeros(3, dtype=np.intp),
                    "is_binned": [True],
                },
            ),
            (
                "ValuedMetadataLike",
                {
                    "factor_names": ["a"],
                    "factor_values": np.zeros((3, 1)),
                    "class_labels": np.zeros(3, dtype=np.intp),
                },
            ),
        ],
    )
    def test_every_member_is_still_required(self, protocol, members):
        """Inheriting must not make ``class_labels`` optional.

        ``isinstance`` collects inherited members as well as declared ones, so moving it to
        the base changes nothing about what is checked — this is what says so, and it is
        the one thing the refactor could plausibly have broken.
        """
        declared = getattr(protocols, protocol)
        assert isinstance(type("Whole", (), dict(members))(), declared)

        for absent in members:
            partial = {name: value for name, value in members.items() if name != absent}
            assert not isinstance(type("Partial", (), partial)(), declared), absent
