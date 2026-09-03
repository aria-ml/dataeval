"""Test Metadata class as FeatureExtractor protocol (unbound usage)."""

import numpy as np
import pytest

from dataeval import Metadata
from dataeval.exceptions import NotFittedError
from tests.embeddings.test_embeddings import MockDataset


@pytest.fixture
def mock_ds():
    """Create a simple mock dataset."""
    return MockDataset(
        np.ones((10, 3, 3)),
        np.ones((10, 3)),
        [{str(i): float(i), "category": f"cat_{i % 3}"} for i in range(10)],
    )


@pytest.fixture
def mock_ds2():
    """Create a different mock dataset with same structure."""
    return MockDataset(
        np.ones((5, 3, 3)),
        np.ones((5, 3)),
        [{str(i): float(i + 10), "category": f"cat_{i % 3}"} for i in range(5)],
    )


class TestMetadataFeatureExtractor:
    """Test Metadata as a FeatureExtractor (unbound scenarios)."""

    def test_is_bound_false(self):
        """Test is_bound property returns False for unbound instance."""
        metadata = Metadata()
        assert not metadata.is_bound

    def test_is_bound_true(self, mock_ds):
        """Test is_bound property returns True for bound instance."""
        metadata = Metadata(mock_ds)
        assert metadata.is_bound

    def test_bind_returns_self(self, mock_ds):
        """Test bind() returns self for method chaining."""
        metadata = Metadata()
        result = metadata.bind(mock_ds)
        assert result is metadata
        assert metadata.is_bound

    def test_len_unbound_raises(self):
        """Test __len__ raises ValueError when no dataset is bound."""
        metadata = Metadata()
        with pytest.raises(NotFittedError, match="No dataset bound"):
            _ = len(metadata)

    def test_iter_unbound_raises(self):
        """Test __iter__ raises ValueError when no dataset is bound."""
        metadata = Metadata()
        with pytest.raises(NotFittedError, match="No dataset bound"):
            for _ in metadata:
                pass

    def test_getitem_unbound_raises(self):
        """Test __getitem__ raises ValueError when no dataset is bound."""
        metadata = Metadata()
        with pytest.raises(NotFittedError, match="No dataset bound"):
            _ = metadata[0]

    def test_shape_unbound_raises(self):
        """Test shape property raises ValueError when no dataset is bound."""
        metadata = Metadata()
        with pytest.raises(NotFittedError, match="No dataset bound"):
            _ = metadata.shape

    def test_call_unbound_raises(self):
        """Test __call__ raises ValueError when data is None and no dataset is bound."""
        metadata = Metadata()
        with pytest.raises(NotFittedError, match="No dataset bound"):
            _ = metadata()

    def test_call_with_data_unbound(self, mock_ds):
        """Test __call__ with data argument on unbound instance."""
        metadata = Metadata(continuous_factor_bins={"0": 5})
        result = metadata(mock_ds)
        assert result.shape[0] == 10
        assert len(result.shape) == 2

    def test_call_bound_no_args(self, mock_ds):
        """Test __call__ without arguments uses bound dataset."""
        metadata = Metadata(mock_ds, continuous_factor_bins={"0": 5})
        result = metadata()
        assert result.shape[0] == 10

    def test_call_same_dataset(self, mock_ds):
        """Test __call__ with same dataset (by identity) returns cached data."""
        metadata = Metadata(mock_ds, continuous_factor_bins={"0": 5})
        result1 = metadata()
        result2 = metadata(mock_ds)
        np.testing.assert_array_equal(result1, result2)

    def test_call_different_dataset(self, mock_ds, mock_ds2):
        """Test __call__ with different dataset creates new computation."""
        metadata = Metadata(mock_ds, continuous_factor_bins={"0": 5})
        result1 = metadata()
        result2 = metadata(mock_ds2)
        assert result1.shape[0] == 10
        assert result2.shape[0] == 5

    def test_bind_clears_state(self, mock_ds, mock_ds2):
        """Test bind() clears cached state."""
        metadata = Metadata(mock_ds, continuous_factor_bins={"0": 5})
        # Access to cache state
        _ = metadata()

        # Bind new dataset
        metadata.bind(mock_ds2)

        # Verify state was cleared by checking we get new data
        result = metadata()
        assert result.shape[0] == 5


class TestMetadataErrorCases:
    """Test error handling in Metadata."""

    def test_exclude_and_include_raises(self, mock_ds):
        """Test both exclude and include raises ValueError."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            Metadata(mock_ds, exclude=["a"], include=["b"])

    def test_getitem_slice_unbound_raises(self):
        """Test __getitem__ with slice raises when unbound."""
        metadata = Metadata()
        with pytest.raises(NotFittedError, match="No dataset bound"):
            _ = metadata[0:5]

    def test_getitem_string_unbound_raises(self):
        """Test __getitem__ with string raises when unbound."""
        metadata = Metadata()
        with pytest.raises(NotFittedError, match="No dataset bound"):
            _ = metadata["some_factor"]

    def test_getitem_invalid_factor_raises(self, mock_ds):
        """Test __getitem__ with invalid factor name raises KeyError."""
        metadata = Metadata(mock_ds)
        with pytest.raises(KeyError, match="not found"):
            _ = metadata["nonexistent_factor"]

    def test_getitem_invalid_type_raises(self, mock_ds):
        """Test __getitem__ with invalid type raises TypeError."""
        metadata = Metadata(mock_ds)
        with pytest.raises(TypeError, match="Index must be"):
            _ = metadata[1.5]  # type: ignore


class TestMetadataInherited:
    """Test the inherited property."""

    def test_inherited_default(self, mock_ds):
        """Test inherited defaults to True."""
        metadata = Metadata(mock_ds)
        assert metadata.inherited

    def test_inherited_setter_triggers_rebuild(self, get_od_dataset):
        """Test setting inherited triggers factor rebuild."""
        od_ds = get_od_dataset(10, 2)

        metadata = Metadata(od_ds)
        initial_factors = set(metadata.factor_names)

        # Dropping inherited factors should rebuild
        metadata.inherited = False
        filtered_factors = set(metadata.factor_names)

        # Should have fewer or equal factors when filtering to view-native only
        assert len(filtered_factors) <= len(initial_factors)

    def test_constructor_arg_matches_the_setter(self, get_od_dataset):
        """The constructor argument and the toggle reach the same state."""
        od_ds = get_od_dataset(10, 2)

        constructed = Metadata(od_ds, inherited=False)
        toggled = Metadata(od_ds)
        toggled.inherited = False

        assert constructed.inherited is False
        assert list(constructed.factor_names) == list(toggled.factor_names)

    def test_new_carries_the_flag_forward(self, get_od_dataset):
        """new() reproduces the config, which has to include this one."""
        od_ds = get_od_dataset(10, 2)

        metadata = Metadata(od_ds, inherited=False)

        assert metadata.new(od_ds).inherited is False


class TestMetadataItemCount:
    """Test item_count property."""

    def test_item_count_no_trigger_when_nonzero(self, mock_ds):
        """Test item_count property doesn't trigger structure when count is already set."""
        metadata = Metadata(mock_ds)
        # Count is set during init
        count = metadata.item_count
        assert count == 10


class TestMetadataFitOnFirstCall:
    """Test the fit-on-first-call contract shared with :class:`BoVWExtractor`.

    An unbound ``Metadata`` used as a feature extractor must *fit* on the first call
    and only *transform* afterwards. Deriving a fresh encoding per call silently
    compares bin codes that mean different things on either side.
    """

    def test_first_call_fits_the_instance(self, mock_ds):
        """The first call binds, so the extractor can describe what it produced."""
        extractor = Metadata()
        with pytest.raises(NotFittedError):
            _ = extractor.factor_names

        extractor(mock_ds)

        assert list(extractor.factor_names)
        assert extractor.is_bound

    def test_encoding_is_frozen_by_the_first_call(self, mock_ds, mock_ds2):
        """A second dataset is encoded against the first one's cuts, not its own."""
        extractor = Metadata()
        extractor(mock_ds)
        fitted = extractor.encoding()

        extractor(mock_ds2)

        assert extractor.encoding() == fitted

    def test_second_dataset_reuses_the_fitted_encoding(self, mock_ds, mock_ds2):
        """The transform of a second dataset matches an explicit shared encoding."""
        extractor = Metadata()
        extractor(mock_ds)
        shared = Metadata(mock_ds2, encoding=extractor.encoding())

        np.testing.assert_array_equal(np.asarray(extractor(mock_ds2)), np.asarray(shared.factor_data))

    def test_columns_are_stable_across_calls(self, mock_ds, mock_ds2):
        """Both sides must produce the same columns in the same order."""
        extractor = Metadata()
        first = np.asarray(extractor(mock_ds))
        names = list(extractor.factor_names)
        second = np.asarray(extractor(mock_ds2))

        assert first.shape[1] == second.shape[1] == len(names)

    def test_fitting_preserves_an_explicit_view(self, get_od_dataset):
        """``bind`` clears an explicit view; the fit path must not.

        Regression test: fitting through ``bind`` alone reset ``view="unit"`` to the
        instance-level default, which silently pulled per-detection factors into the
        extracted columns.
        """
        dataset = get_od_dataset(10, 2)
        extractor = Metadata(view="unit")

        extractor(dataset)

        assert extractor.view == "unit"
        assert list(extractor.factor_names) == list(Metadata(dataset, view="unit").factor_names)

    def test_new_carries_the_fitted_encoding(self, mock_ds, mock_ds2):
        """``new`` shares the fitted encoding, which is what it exists to do.

        Its own contract is that a derived instance is "configured identically", so
        once the first call has fitted an encoding, ``new`` must carry it. Building a
        fresh ``Metadata`` is the way to ask for an independent fit.
        """
        extractor = Metadata()
        extractor(mock_ds)

        derived = extractor.new(mock_ds2)

        assert derived.encoding() == extractor.encoding()

    def test_bound_instance_transforms_other_data(self, mock_ds, mock_ds2):
        """A bound instance already counts as fitted, so it transforms rather than refits."""
        metadata = Metadata(mock_ds)
        _ = metadata.factor_data
        fitted = metadata.encoding()

        metadata(mock_ds2)

        assert metadata.encoding() == fitted


@pytest.mark.required
class TestATransformYieldsTheFittedFactorSet:
    """Which columns a dataset yields is a property of that dataset, so the two sides of a
    comparison can disagree about them -- and a feature-wise detector lines its columns up
    positionally, with nothing to notice that they describe different factors."""

    @staticmethod
    def _mixed_on_the_reference():
        """A column mixed on the reference and clean on the stream measured against it.

        The shape that started this: only the earlier campaign carried the sentinel, so the
        reference held the column back and the later stream read it as an ordinary factor.
        """
        reference = MockDataset(
            np.ones((6, 3, 3)),
            np.ones((6, 3)),
            [{"a": 1.0 if i % 2 else "N", "z": float(i)} for i in range(6)],
        )
        stream = MockDataset(
            np.ones((4, 3, 3)),
            np.ones((4, 3)),
            [{"a": float(i + 5), "z": float(i)} for i in range(4)],
        )
        return reference, stream

    def test_the_two_datasets_do_read_different_factor_sets(self):
        """The premise: without this, there is nothing for the transform to reconcile."""
        reference, stream = self._mixed_on_the_reference()
        assert Metadata(reference).factor_names == ["z"]
        assert Metadata(stream).factor_names == ["a", "z"]

    def test_a_factor_the_reference_never_had_is_left_out(self):
        """It carries no recorded encoding, so a code in its column would mean nothing."""
        reference, stream = self._mixed_on_the_reference()
        extractor = Metadata(reference)

        transformed = np.asarray(extractor(stream))

        assert transformed.shape == (4, 1)

    def test_the_columns_are_the_fitted_factors_and_in_that_order(self):
        """Read by name rather than by position, so the column that arrives as feature f is
        the factor the reference called feature f.

        Checked against the stream read under the *reference's* encoding, which is the only
        thing the codes are meaningful against -- a column re-cut from the stream's own draw
        would disagree with this one even where the factor is right.
        """
        reference = MockDataset(
            np.ones((6, 3, 3)),
            np.ones((6, 3)),
            [{"a": 1.0 if i % 2 else "N", "m": float(i), "z": float(10 - i)} for i in range(6)],
        )
        stream = MockDataset(
            np.ones((4, 3, 3)),
            np.ones((4, 3)),
            [{"a": float(100 + i), "m": float(i), "z": float(10 - i)} for i in range(4)],
        )
        extractor = Metadata(reference)
        assert extractor.factor_names == ["m", "z"]

        transformed = np.asarray(extractor(stream))

        # `a` sorts ahead of both, so lining the columns up positionally would put it where
        # `m` belongs and shift `z` off the end entirely.
        derived = extractor.new(stream)
        position = {name: i for i, name in enumerate(derived.factor_names)}
        assert position["a"] == 0
        columns = np.asarray(derived.factor_data)
        expected = columns[:, [position["m"], position["z"]]]

        assert transformed.shape == (4, 2)
        assert np.array_equal(transformed, expected)

    def test_a_fitted_factor_the_new_data_lacks_raises(self):
        """There is nothing honest to put in its column, and reading the rest in order would
        compare each against a different factor."""
        reference = MockDataset(
            np.ones((4, 3, 3)),
            np.ones((4, 3)),
            [{"a": float(i), "z": float(i)} for i in range(4)],
        )
        stream = MockDataset(np.ones((3, 3, 3)), np.ones((3, 3)), [{"z": float(i)} for i in range(3)])
        extractor = Metadata(reference)
        assert "a" in extractor.factor_names

        with pytest.raises(ValueError, match=r"yields no column for \['a'\]"):
            extractor(stream)
