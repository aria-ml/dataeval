"""The protocols a caller implements to hand DataEval its own container."""

from typing import Annotated, Any, NamedTuple, get_args, get_origin
from unittest.mock import Mock

import maite.protocols.multiobject_tracking
import maite.protocols.object_detection
import numpy as np
import pytest

from dataeval import protocols
from dataeval.protocols import _is_protocol_instance


def _mro(protocol: Any) -> tuple[type, ...]:
    """Read a protocol class's MRO without pyright objecting to the dunder."""
    return protocol.__mro__


class ObjectDetectionTargetTuple(NamedTuple):
    boxes: np.ndarray
    labels: np.ndarray
    scores: np.ndarray


class SingleFrameObjectTrackingTargetTuple(NamedTuple):
    boxes: np.ndarray
    labels: np.ndarray
    scores: np.ndarray
    track_ids: np.ndarray


class MultiobjectTrackingTargetTuple(NamedTuple):
    frame_tracks: tuple[SingleFrameObjectTrackingTargetTuple, ...]


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


@pytest.mark.required
class TestTheTargetProtocolsStayInstanceCheckable:
    """The two target protocols must answer ``isinstance``, not raise on it.

    ``dataeval.protocols`` mirrors MAITE's two target protocols instead of re-exporting
    them, because no MAITE name supports ``isinstance`` across the supported range
    (``maite>=0.9.4``): 0.9.x leaves ``MultiobjectTrackingTarget`` un-``runtime_checkable``
    and 0.10 wraps both in ``Annotated[..., Is[...]]``. Either way ``isinstance`` raises
    :class:`TypeError` for *every* object handed to it, so the whole task-dispatch layer
    (``detect_task``, ``dataeval.utils._validate``, ``ClassFilter``, ``Relabel``, ...)
    fails closed on every dataset rather than on non-conforming ones.

    Mirroring is only safe while the members match, so these pin them to MAITE's.
    """

    @staticmethod
    def _maite_members(alias: Any) -> set[str]:
        """The member names MAITE declares, reached through any ``Annotated`` wrapper."""
        protocol = get_args(alias)[0] if get_origin(alias) is Annotated else alias
        return {name for name in dir(protocol) if not name.startswith("_")}

    @pytest.mark.parametrize(
        ("mirrored", "wrapped"),
        [
            ("ObjectDetectionTarget", maite.protocols.object_detection.ObjectDetectionTarget),
            ("MultiobjectTrackingTarget", maite.protocols.multiobject_tracking.MultiobjectTrackingTarget),
        ],
    )
    def test_members_match_maite(self, mirrored, wrapped):
        """A MAITE release that adds or renames a target member must fail here."""
        declared = getattr(protocols, mirrored)
        assert self._maite_members(wrapped) == {name for name in dir(declared) if not name.startswith("_")}

    @pytest.mark.parametrize("mirrored", ["ObjectDetectionTarget", "MultiobjectTrackingTarget"])
    def test_isinstance_answers_instead_of_raising(self, mirrored):
        """The failure mode being guarded against is a raise, not a wrong answer."""
        declared = getattr(protocols, mirrored)
        assert declared._is_runtime_protocol
        assert isinstance(object(), declared) is False

    def test_the_two_targets_are_told_apart(self):
        """Dispatch depends on a detection target and a tracking target not colliding."""

        class Detection:
            boxes = np.zeros((2, 4))
            labels = np.zeros(2)
            scores = np.zeros(2)

        class Tracking:
            frame_tracks = ()

        assert isinstance(Detection(), protocols.ObjectDetectionTarget)
        assert not isinstance(Detection(), protocols.MultiobjectTrackingTarget)
        assert isinstance(Tracking(), protocols.MultiobjectTrackingTarget)
        assert not isinstance(Tracking(), protocols.ObjectDetectionTarget)
        assert not isinstance(np.zeros(3), protocols.ObjectDetectionTarget)

    def test_a_union_of_targets_is_instance_checkable(self):
        """``isinstance(t, A | B)`` is how ClassFilter and ClassBalance ask the question."""

        class Detection:
            boxes = np.zeros((2, 4))
            labels = np.zeros(2)
            scores = np.zeros(2)

        assert isinstance(Detection(), protocols.ObjectDetectionTarget | protocols.SegmentationTarget)

    def test_real_conforming_targets_satisfy_the_mirror(self):
        """Mirroring is structural, so targets built for MAITE must pass unchanged."""

        detection = ObjectDetectionTargetTuple(boxes=np.zeros((2, 4)), labels=np.zeros(2), scores=np.zeros(2))
        assert isinstance(detection, protocols.ObjectDetectionTarget)
        assert not isinstance(detection, protocols.MultiobjectTrackingTarget)

        frame = SingleFrameObjectTrackingTargetTuple(
            boxes=np.zeros((2, 4)), labels=np.zeros(2), scores=np.zeros(2), track_ids=np.zeros(2)
        )
        tracking = MultiobjectTrackingTargetTuple(frame_tracks=(frame,))
        assert isinstance(tracking, protocols.MultiobjectTrackingTarget)
        assert not isinstance(tracking, protocols.ObjectDetectionTarget)


@pytest.mark.required
class TestProtocolInstanceIsVersionIndependent:
    """``_is_protocol_instance`` must give the same answer on every supported Python.

    ``isinstance`` against a ``@runtime_checkable`` protocol does not: below 3.12 it probes
    members with ``hasattr``, from 3.12 on with :func:`inspect.getattr_static`. Task
    dispatch is decided by that answer, so a divergence there means a dataset is read as a
    different task on 3.11 than on 3.12. These assert the 3.12 answer, so on 3.10 and 3.11
    they fail if dispatch ever falls back to ``isinstance``.
    """

    class Detection:
        boxes = np.zeros((2, 4))
        labels = np.zeros(2)
        scores = np.zeros(2)

    def test_a_conforming_target_passes(self):
        assert _is_protocol_instance(self.Detection(), protocols.ObjectDetectionTarget)

    def test_a_missing_member_fails(self):
        class NoScores:
            boxes = np.zeros((2, 4))
            labels = np.zeros(2)

        assert not _is_protocol_instance(NoScores(), protocols.ObjectDetectionTarget)
        assert not _is_protocol_instance(np.zeros(3), protocols.ObjectDetectionTarget)

    def test_a_bare_stand_in_is_rejected_on_every_version(self):
        """``isinstance`` answers True here below Python 3.12; this must not."""
        stand_in = Mock()
        stand_in.boxes = np.zeros((2, 4))
        stand_in.labels = np.zeros(2)
        stand_in.scores = np.zeros(2)

        assert _is_protocol_instance(stand_in, protocols.ObjectDetectionTarget)
        # ...but nothing was fabricated for a protocol it was never given members for.
        assert not _is_protocol_instance(Mock(), protocols.ObjectDetectionTarget)
        assert not _is_protocol_instance(Mock(), protocols.MultiobjectTrackingTarget)

    def test_a_raising_member_is_seen_not_called(self):
        """Below 3.12 ``isinstance`` calls the getter, so this raises out of the type test."""
        calls = []

        class Guarded:
            @property
            def boxes(self):
                calls.append(1)
                raise ValueError("no boxes above the image level")

            labels = np.zeros(2)
            scores = np.zeros(2)

        assert _is_protocol_instance(Guarded(), protocols.ObjectDetectionTarget)
        assert calls == []

    def test_properties_and_namedtuples_both_pass(self):
        """The two shapes real targets actually take."""

        class Props:
            @property
            def boxes(self):
                return np.zeros((2, 4))

            @property
            def labels(self):
                return np.zeros(2)

            @property
            def scores(self):
                return np.zeros(2)

        assert _is_protocol_instance(Props(), protocols.ObjectDetectionTarget)
        assert _is_protocol_instance(
            ObjectDetectionTargetTuple(boxes=np.zeros((2, 4)), labels=np.zeros(2), scores=np.zeros(2)),
            protocols.ObjectDetectionTarget,
        )
