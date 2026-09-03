import pytest

from dataeval.utils._mask import materialize_target_attrs


@pytest.mark.required
class TestMaterializeTargetAttrs:
    def test_a_target_with_neither_dict_nor_fields_yields_nothing(self):
        """__slots__ without _fields is neither a namedtuple nor a __dict__ carrier."""

        class _Opaque:
            __slots__ = ()

        assert materialize_target_attrs(_Opaque()) == {}
