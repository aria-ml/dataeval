import numpy as np
import pytest

from dataeval.protocols import ObjectDetectionTarget, SegmentationTarget, _is_protocol_instance
from dataeval.utils._mask import MaskedTarget, materialize_target_attrs


@pytest.mark.required
class TestMaterializeTargetAttrs:
    def test_a_target_with_neither_dict_nor_fields_yields_nothing(self):
        """__slots__ without _fields is neither a namedtuple nor a __dict__ carrier."""

        class _Opaque:
            __slots__ = ()

        assert materialize_target_attrs(_Opaque()) == {}

    def test_a_property_backed_target_still_surfaces_its_public_members(self):
        """A target exposing boxes/labels/scores via @property (e.g. maite_datasets'
        CustomObjectDetectionTarget) stores them under private, differently-named instance
        attrs. A blind ``__dict__`` copy surfaces those private names instead of the public
        protocol names, so a proxy built from it fails a static protocol check even though
        live attribute reads (which go through ``__getattribute__``, not ``__dict__``) work fine.
        """

        class _PropertyBacked:
            def __init__(self) -> None:
                self._boxes = np.zeros((1, 4), dtype=np.float32)
                self._labels = np.asarray([0], dtype=np.intp)
                self._scores = np.asarray([0.9], dtype=np.float32)

            @property
            def boxes(self):
                return self._boxes

            @property
            def labels(self):
                return self._labels

            @property
            def scores(self):
                return self._scores

        attrs = materialize_target_attrs(_PropertyBacked())
        assert {"boxes", "labels", "scores"} <= attrs.keys()

        proxy = MaskedTarget(_PropertyBacked(), np.array([True]))
        assert _is_protocol_instance(proxy, ObjectDetectionTarget)

    def test_a_property_backed_segmentation_target_still_surfaces_its_mask(self):
        """ClassFilter wraps segmentation targets in MaskedTarget too, and SegmentationTarget's
        ``mask`` member is not among ObjectDetectionTarget's -- it needs its own probe.
        """

        class _SegPropertyBacked:
            def __init__(self) -> None:
                self._seg_mask = np.zeros((1, 4, 4), dtype=np.bool_)
                self._labels = np.asarray([0], dtype=np.intp)
                self._scores = np.asarray([0.9], dtype=np.float32)

            @property
            def mask(self):
                return self._seg_mask

            @property
            def labels(self):
                return self._labels

            @property
            def scores(self):
                return self._scores

        proxy = MaskedTarget(_SegPropertyBacked(), np.array([True]))
        assert _is_protocol_instance(proxy, SegmentationTarget)
