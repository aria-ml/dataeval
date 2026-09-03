import pytest

from dataeval.utils._internal import PoolWrapper, simplify_type


@pytest.mark.required
class TestCastSimplify:
    @pytest.mark.parametrize(
        ("value", "output"),
        [
            ("123", 123),
            ("12.3", 12.3),
            ("foo", "foo"),
            ([123, "12.3"], [123.0, 12.3]),
            ([123, "foo"], ["123", "foo"]),
            (["123", "456"], [123, 456]),
        ],
    )
    def test_convert_type(self, value, output):
        assert output == simplify_type(value)


@pytest.mark.required
class TestPoolWrapperLifecycle:
    """A real pool must be closed and joined on exit; the single-threaded path has none."""

    def test_a_multiprocess_pool_is_closed_on_exit(self):
        with PoolWrapper(processes=2) as pool:
            assert pool._pool is not None
            assert list(pool.imap_unordered(abs, [-1, -2, -3])) != []
        # close() then join() have run; a closed pool refuses new work.
        with pytest.raises(ValueError, match="Pool not running"):
            pool._pool.apply(abs, (-1,))

    def test_the_single_threaded_path_has_nothing_to_close(self):
        with PoolWrapper(processes=1) as pool:
            assert pool._pool is None
            assert sorted(pool.imap_unordered(abs, [-1, -2])) == [1, 2]
