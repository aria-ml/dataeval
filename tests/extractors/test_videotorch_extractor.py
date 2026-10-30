"""Tests for VideoTorchExtractor."""

from collections.abc import Iterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from dataeval import Embeddings
from dataeval.extractors import VideoTorchExtractor
from dataeval.protocols import FeatureExtractor

# The extractor is agnostic to frame resolution, so tiny frames are used to keep
# the mock model's linear layer and the test videos small.
FRAME_SHAPE = (8, 8, 3)  # (H, W, C)
FRAME_SIZE = 8 * 8 * 3
NUM_FRAMES = 4
HIDDEN_SIZE = 8


def make_video(n_frames: int) -> np.ndarray:
    """Create a random video of shape (n_frames, H, W, C)."""
    return np.random.rand(n_frames, *FRAME_SHAPE).astype(np.float32)


@dataclass(frozen=True)
class StreamFrame:
    """A decoded frame as a MAITE video stream yields it: channel-first pixels plus timing."""

    pixels: np.ndarray
    time_s: float
    pts: int
    frame_index: int


def frames_of(video: np.ndarray, pulled: list[int] | None = None) -> Iterator[StreamFrame]:
    """Yield a (T, H, W, C) video's frames channel first, as a generator: no len, no indexing.

    ``pulled``, when given, counts the frames handed out so far in its only element.
    """
    for i, frame in enumerate(video):
        if pulled is not None:
            pulled[0] += 1
        yield StreamFrame(np.ascontiguousarray(np.moveaxis(frame, -1, 0)), i / 30, i, i)


class ReplayableStream:
    """A stream that decodes afresh on every walk, as a file-backed reader does."""

    def __init__(self, video: np.ndarray) -> None:
        self.video = video
        self.walks = 0

    def __iter__(self) -> Iterator[StreamFrame]:
        self.walks += 1
        return frames_of(self.video)


# Mock VideoMAE-like model for testing
class MockVideoModel(torch.nn.Module):
    """Mock video transformer model that mimics HuggingFace structure."""

    def __init__(self, hidden_size: int = HIDDEN_SIZE, num_frames: int = NUM_FRAMES):
        super().__init__()
        self.config = type("Config", (), {"num_frames": num_frames, "hidden_size": hidden_size})()
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(num_frames * FRAME_SIZE, hidden_size),
        )

    def forward(self, pixel_values):
        hidden_states = self.encoder(pixel_values)
        # Return structure similar to HuggingFace BaseModelOutput
        return type(
            "ModelOutput",
            (),
            {
                "last_hidden_state": hidden_states.unsqueeze(1),  # (batch, 1, hidden_size)
                "pooler_output": None,
            },
        )()


class MockProcessor:
    """Mock HuggingFace processor for testing."""

    def __call__(self, frames, return_tensors="pt"):
        # Convert list of frames to tensor
        # Stack frames: (num_frames, H, W, C) -> (1, num_frames, C, H, W)
        frames_array = np.stack([np.array(f) for f in frames]) if isinstance(frames, list) else np.array(frames)

        # Convert to tensor and rearrange dimensions
        tensor = torch.from_numpy(frames_array).float()
        if tensor.ndim == 4:  # (T, H, W, C)
            tensor = tensor.permute(3, 0, 1, 2)  # (C, T, H, W)

        return type("ProcessorOutput", (), {"pixel_values": tensor.unsqueeze(0)})()


@pytest.fixture
def processor():
    """Create a mock processor."""
    return MockProcessor()


@pytest.fixture
def extractor(processor):
    """Create a simple extractor for testing."""
    model = MockVideoModel()
    return VideoTorchExtractor(model, processor=processor, device="cpu", pooling="first", clip_aggregation="mean")


@pytest.mark.required
class TestVideoTorchExtractorInit:
    """Test VideoTorchExtractor initialization."""

    def test_init_basic(self, processor):
        """Test basic initialization."""
        extractor = VideoTorchExtractor(MockVideoModel(), processor=processor)
        assert extractor.device is not None
        assert extractor.layer_name is None
        assert extractor.use_output is True
        assert extractor.pooling == "first"
        assert extractor.clip_aggregation == "mean"
        assert extractor.num_frames == NUM_FRAMES

    def test_init_with_device(self, processor):
        """Test initialization with specified device."""
        extractor = VideoTorchExtractor(MockVideoModel(), processor=processor, device="cpu")
        assert extractor.device == torch.device("cpu")

    def test_init_auto_num_frames_from_config(self, processor):
        """Test that num_frames is automatically extracted from model.config."""
        extractor = VideoTorchExtractor(MockVideoModel(num_frames=8), processor=processor)
        assert extractor.num_frames == 8

    def test_init_manual_num_frames(self, processor):
        """Test that an explicit num_frames overrides model.config."""
        extractor = VideoTorchExtractor(MockVideoModel(), processor=processor, num_frames=2)
        assert extractor.num_frames == 2

    def test_init_without_num_frames_raises(self, processor):
        """Test that missing num_frames raises ValueError."""
        model = torch.nn.Sequential(torch.nn.Flatten())  # No config
        with pytest.raises(ValueError, match="num_frames must be provided"):
            VideoTorchExtractor(model, processor=processor)

    def test_init_with_layer_name(self, processor):
        """Test initialization with layer extraction."""
        extractor = VideoTorchExtractor(MockVideoModel(), processor=processor, layer_name="encoder.0")
        assert extractor.layer_name == "encoder.0"

    def test_init_with_invalid_layer_raises(self, processor):
        """Test that invalid layer name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid layer"):
            VideoTorchExtractor(MockVideoModel(), processor=processor, layer_name="nonexistent")

    @pytest.mark.parametrize("pooling", ["mean", "first", "last", "none"])
    def test_init_with_pooling_options(self, processor, pooling):
        """Test initialization with different pooling strategies."""
        extractor = VideoTorchExtractor(MockVideoModel(), processor=processor, pooling=pooling)
        assert extractor.pooling == pooling

    def test_init_with_invalid_pooling_raises(self, processor):
        """Test that invalid pooling raises ValueError."""
        with pytest.raises(ValueError, match="Invalid pooling"):
            VideoTorchExtractor(MockVideoModel(), processor=processor, pooling="invalid")

    @pytest.mark.parametrize("aggregation", ["mean", "max"])
    def test_init_with_clip_aggregation_options(self, processor, aggregation):
        """Test initialization with different clip aggregation strategies."""
        extractor = VideoTorchExtractor(MockVideoModel(), processor=processor, clip_aggregation=aggregation)
        assert extractor.clip_aggregation == aggregation

    def test_init_with_invalid_clip_aggregation_raises(self, processor):
        """Test that invalid clip_aggregation raises ValueError."""
        with pytest.raises(ValueError, match="Invalid clip_aggregation"):
            VideoTorchExtractor(MockVideoModel(), processor=processor, clip_aggregation="invalid")

    def test_init_with_transforms(self, processor):
        """Test initialization with transforms."""

        class MockTransform:
            def __call__(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2

        extractor = VideoTorchExtractor(MockVideoModel(), processor=processor, transforms=MockTransform())
        assert len(extractor._transforms) == 1


@pytest.mark.required
class TestVideoTorchExtractorCall:
    """Test VideoTorchExtractor.__call__ method."""

    @pytest.mark.parametrize(
        ("frame_counts", "expected_videos"),
        [
            pytest.param([NUM_FRAMES], 1, id="single_clip"),
            pytest.param([NUM_FRAMES * 3], 1, id="multiple_clips"),
            pytest.param([NUM_FRAMES + 1], 1, id="trailing_incomplete_clip_skipped"),
            pytest.param([NUM_FRAMES, NUM_FRAMES * 3, NUM_FRAMES * 5], 3, id="mixed_lengths"),
            pytest.param([], 0, id="no_videos"),
        ],
    )
    def test_call(self, extractor, frame_counts, expected_videos):
        """Test that each video yields exactly one embedding, in the order given."""
        result = extractor([make_video(n) for n in frame_counts])

        assert isinstance(result, np.ndarray)
        assert result.shape[0] == expected_videos
        if expected_videos:
            assert result.shape[1] == HIDDEN_SIZE

    @pytest.mark.parametrize("frame_counts", [[NUM_FRAMES - 1], [NUM_FRAMES, NUM_FRAMES - 1]])
    def test_call_rejects_a_video_with_no_complete_clip(self, extractor, frame_counts):
        """A video with no embedding cannot be dropped: the result is read by position."""
        with pytest.raises(ValueError, match="shorter than one clip"):
            extractor([make_video(n) for n in frame_counts])

    def test_call_returns_one_row_per_video_in_order(self, extractor):
        """Row i belongs to video i -- what Embeddings relies on when it stores by index."""
        videos = [np.full((NUM_FRAMES, *FRAME_SHAPE), i, dtype=np.float32) for i in range(4)]
        result = extractor(videos)

        assert result.shape == (4, HIDDEN_SIZE)
        for i, video in enumerate(videos):
            np.testing.assert_allclose(result[i], extractor([video])[0], rtol=1e-6)

    def test_call_with_generator(self, extractor):
        """Test that extractor works with generators (lazy loading)."""

        def video_generator():
            for _ in range(3):
                yield make_video(NUM_FRAMES)

        result = extractor(video_generator())
        assert result.shape == (3, HIDDEN_SIZE)

    def test_call_very_long_video(self, extractor):
        """Test that videos with many clips are aggregated into a single embedding."""
        # 101 clips exercises the long-video code path
        result = extractor([make_video(NUM_FRAMES * 101)])
        assert result.shape == (1, HIDDEN_SIZE)


@pytest.mark.required
class TestVideoTorchExtractorStreams:
    """A MAITE video stream embeds as the same frames do in a (T, H, W, C) array."""

    @pytest.mark.parametrize("n_clips", [1, 3])
    def test_stream_matches_array(self, extractor, n_clips):
        """Whole clips read from a stream give the embedding the same frames give as an array."""
        video = make_video(NUM_FRAMES * n_clips)
        np.testing.assert_allclose(extractor([frames_of(video)]), extractor([video]), rtol=1e-6)

    def test_stream_drops_a_trailing_partial_clip(self, extractor):
        """Frames past the last whole clip are left out, as they are from an array."""
        video = make_video(NUM_FRAMES * 2 + 1)
        result = extractor([frames_of(video)])

        np.testing.assert_allclose(result, extractor([video]), rtol=1e-6)
        np.testing.assert_allclose(result, extractor([video[: NUM_FRAMES * 2]]), rtol=1e-6)

    def test_stream_shorter_than_one_clip_raises(self, extractor):
        """A stream with no whole clip has no embedding, and the positional result cannot skip it."""
        streams = [frames_of(make_video(NUM_FRAMES)), frames_of(make_video(NUM_FRAMES - 1))]
        with pytest.raises(ValueError, match="index 1 is shorter than one clip"):
            extractor(streams)

    def test_stream_is_read_one_clip_at_a_time(self):
        """A generator, never measured or indexed: each clip is embedded before the next is read."""
        pulled = [0]
        pulled_at_each_clip = []

        class RecordingProcessor(MockProcessor):
            def __call__(self, frames, return_tensors="pt"):
                pulled_at_each_clip.append(pulled[0])
                return super().__call__(frames, return_tensors)

        extractor = VideoTorchExtractor(MockVideoModel(), processor=RecordingProcessor(), device="cpu")
        stream = frames_of(make_video(NUM_FRAMES * 3), pulled)
        assert not hasattr(stream, "__len__")
        assert not hasattr(stream, "__getitem__")

        assert extractor([stream]).shape == (1, HIDDEN_SIZE)
        assert pulled_at_each_clip == [NUM_FRAMES, NUM_FRAMES * 2, NUM_FRAMES * 3]

    def test_a_replayable_stream_is_walked_once_per_call(self, extractor):
        """The stream is left for its caller to walk again, and walking it again gives the same embedding."""
        stream = ReplayableStream(make_video(NUM_FRAMES * 2))
        first, second = extractor([stream]), extractor([stream])

        assert stream.walks == 2
        np.testing.assert_allclose(first, second, rtol=1e-6)

    @pytest.mark.parametrize("pixels_shape", [(1, 8, 8), (8, 8)], ids=["one_channel", "no_channel_axis"])
    def test_single_channel_stream_reads_as_a_single_channel_array(self, pixels_shape):
        """A one-channel frame, with or without its channel axis, reaches the pipeline as (T, H, W, 1) does."""
        video = np.random.rand(NUM_FRAMES, 8, 8, 1).astype(np.float32)
        stream = (StreamFrame(frame.reshape(pixels_shape), i / 30, i, i) for i, frame in enumerate(video))
        clip_shapes = []

        def record(clip: torch.Tensor) -> torch.Tensor:
            clip_shapes.append(tuple(clip.shape))
            return clip

        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(NUM_FRAMES * 8 * 8, HIDDEN_SIZE))
        extractor = VideoTorchExtractor(model, transforms=record, device="cpu", num_frames=NUM_FRAMES)

        result = extractor([stream])
        assert clip_shapes == [(NUM_FRAMES, 8, 8, 1)]
        np.testing.assert_allclose(result, extractor([video]), rtol=1e-6)
        assert clip_shapes == [(NUM_FRAMES, 8, 8, 1)] * 2

    @pytest.mark.parametrize("tracking_datum", [True, False], ids=["stream_target_metadata", "bare_stream"])
    def test_embeddings_of_a_video_stream_dataset(self, extractor, tracking_datum):
        """Embeddings hands each datum's stream to the extractor and stores one row per video."""
        videos = [make_video(NUM_FRAMES * (i + 1)) for i in range(5)]

        class StreamDataset:
            def __len__(self) -> int:
                return len(videos)

            def __getitem__(self, idx: int) -> Any:
                stream = ReplayableStream(videos[idx])
                if not tracking_datum:
                    return stream
                empty_frame = SimpleNamespace(boxes=np.zeros((0, 4)), labels=np.zeros(0, dtype=int))
                return stream, SimpleNamespace(frame_tracks=[empty_frame] * len(videos[idx])), {"id": idx}

        embeddings = Embeddings(StreamDataset(), extractor=extractor, batch_size=2)

        np.testing.assert_allclose(embeddings[:], extractor(videos), rtol=1e-6)


@pytest.mark.required
class TestVideoTorchExtractorClipAggregation:
    """Test different clip aggregation strategies."""

    @pytest.mark.parametrize("aggregation", ["mean", "max"])
    def test_aggregation_shape(self, processor, aggregation):
        """Test aggregation across clips returns one embedding per video."""
        model = MockVideoModel()
        extractor = VideoTorchExtractor(model, processor=processor, device="cpu", clip_aggregation=aggregation)

        result = extractor([make_video(NUM_FRAMES * 3)])
        assert result.shape == (1, HIDDEN_SIZE)

    def test_mean_vs_max_different_results(self, processor):
        """Test that mean and max produce different results."""
        model = MockVideoModel()
        extractor_mean = VideoTorchExtractor(model, processor=processor, device="cpu", clip_aggregation="mean")
        extractor_max = VideoTorchExtractor(model, processor=processor, device="cpu", clip_aggregation="max")

        # Same video for both, with distinct per-clip content so mean and max diverge
        video = np.concatenate([np.full((NUM_FRAMES, *FRAME_SHAPE), i, dtype=np.float32) for i in range(3)])

        result_mean = extractor_mean([video])
        result_max = extractor_max([video])

        assert not np.allclose(result_mean, result_max)


class MultiTokenVideoModel(MockVideoModel):
    """Mock model whose last_hidden_state carries several tokens per clip."""

    n_tokens = 3

    def forward(self, pixel_values) -> Any:
        hidden_states = self.encoder(pixel_values)
        sequence = hidden_states.unsqueeze(1).expand(-1, self.n_tokens, -1)
        return type("ModelOutput", (), {"last_hidden_state": sequence, "pooler_output": None})()


@pytest.mark.required
class TestVideoTorchExtractorPooling:
    """Test different pooling strategies within clips."""

    @pytest.mark.parametrize("pooling", ["mean", "first", "last", "none"])
    def test_pooling(self, processor, pooling):
        """Test that each pooling strategy yields one embedding per video."""
        model = MockVideoModel()
        extractor = VideoTorchExtractor(model, processor=processor, device="cpu", pooling=pooling)

        result = extractor([make_video(NUM_FRAMES)])
        assert result.shape == (1, HIDDEN_SIZE)

    @pytest.mark.parametrize(
        ("pooling", "expected_width"),
        [("mean", HIDDEN_SIZE), ("first", HIDDEN_SIZE), ("last", HIDDEN_SIZE), ("none", HIDDEN_SIZE * 3)],
    )
    def test_pooling_over_a_real_sequence_stays_two_dimensional(self, processor, pooling, expected_width):
        """A multi-token sequence still yields one row per video, whatever the strategy."""
        extractor = VideoTorchExtractor(MultiTokenVideoModel(), processor=processor, device="cpu", pooling=pooling)

        result = extractor([make_video(NUM_FRAMES), make_video(NUM_FRAMES * 2)])
        assert result.shape == (2, expected_width)


@pytest.mark.required
class TestVideoTorchExtractorLayerExtraction:
    """Test layer extraction functionality."""

    def test_extract_intermediate_layer_output(self, processor):
        """Test extracting output from intermediate layer."""
        model = MockVideoModel()
        extractor = VideoTorchExtractor(
            model,
            processor=processor,
            layer_name="encoder.1",  # Linear layer
            device="cpu",
        )

        result = extractor([make_video(NUM_FRAMES)])
        assert result.shape == (1, HIDDEN_SIZE)

    def test_extract_intermediate_layer_input(self, processor):
        """Test extracting input to intermediate layer."""
        model = MockVideoModel()
        extractor = VideoTorchExtractor(
            model, processor=processor, layer_name="encoder.1", use_output=False, device="cpu"
        )

        result = extractor([make_video(NUM_FRAMES)])
        # Input to linear layer should be flattened size
        assert result.shape == (1, NUM_FRAMES * FRAME_SIZE)


@pytest.mark.required
class TestVideoTorchExtractorTransforms:
    """Test transform functionality."""

    def test_transforms_applied(self, processor):
        """Test that transforms are applied during extraction."""

        class DoubleTransform:
            def __call__(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2

        model = MockVideoModel()
        extractor_no_transform = VideoTorchExtractor(model, processor=processor, device="cpu")
        extractor_with_transform = VideoTorchExtractor(
            model, processor=processor, transforms=DoubleTransform(), device="cpu"
        )

        # Use constant input for predictable results
        video = np.ones((NUM_FRAMES, *FRAME_SHAPE), dtype=np.float32)

        result_no_transform = extractor_no_transform([video])
        result_with_transform = extractor_with_transform([video])

        # Results should differ due to transform
        assert not np.allclose(result_no_transform, result_with_transform)


@pytest.mark.required
class TestVideoTorchExtractorRepr:
    """Test __repr__ method."""

    def test_repr_basic(self, processor):
        """Test basic repr."""
        extractor = VideoTorchExtractor(MockVideoModel(), processor=processor, device="cpu")
        repr_str = repr(extractor)

        assert "VideoTorchExtractor" in repr_str
        assert "cpu" in repr_str
        assert f"num_frames={NUM_FRAMES}" in repr_str
        assert "clip_aggregation='mean'" in repr_str
        assert "processor=True" in repr_str

    def test_repr_with_layer_name(self, processor):
        """Test repr includes layer name when set."""
        extractor = VideoTorchExtractor(MockVideoModel(), processor=processor, layer_name="encoder.0", device="cpu")
        assert "layer_name='encoder.0'" in repr(extractor)


@pytest.mark.required
class TestVideoTorchExtractorProtocol:
    """Test that VideoTorchExtractor conforms to FeatureExtractor protocol."""

    def test_protocol_conformance(self, extractor):
        """Test that VideoTorchExtractor implements FeatureExtractor protocol."""
        assert isinstance(extractor, FeatureExtractor)
        assert callable(extractor)

    def test_returns_array_protocol(self, extractor):
        """Test that __call__ returns Array-like object."""
        result = extractor([make_video(NUM_FRAMES)])

        # Should be numpy array (implements Array protocol)
        assert isinstance(result, np.ndarray)
        assert hasattr(result, "shape")
        assert hasattr(result, "dtype")


class TestVideoTorchExtractorIntegration:
    """Integration tests with Embeddings class."""

    def test_works_with_embeddings_class(self, extractor):
        """Test that VideoTorchExtractor can be used with Embeddings."""

        # Create mock video dataset
        class MockVideoDataset:
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                # Return (video, label, metadata) tuple
                return make_video(NUM_FRAMES), idx % 3, {"video_id": idx}

        embeddings = Embeddings(MockVideoDataset(), extractor=extractor, batch_size=4)

        # Test basic operations
        assert len(embeddings) == 10

        # Test single embedding access
        assert embeddings[0].shape == (HIDDEN_SIZE,)

        # Test batch access
        assert embeddings[0:5].shape == (5, HIDDEN_SIZE)

        # Test full array access
        assert embeddings[:].shape == (10, HIDDEN_SIZE)

    def test_embeddings_with_long_videos(self, extractor):
        """Test Embeddings with videos that have multiple clips."""

        class MockLongVideoDataset:
            def __len__(self):
                return 5

            def __getitem__(self, idx):
                # Videos with varying lengths (1 to 5 clips)
                return make_video(NUM_FRAMES * (idx + 1)), idx, {}

        embeddings = Embeddings(MockLongVideoDataset(), extractor=extractor, batch_size=2)

        # All videos should produce embeddings despite different lengths
        assert embeddings[:].shape == (5, HIDDEN_SIZE)

    def test_embeddings_compute_and_cache(self, extractor):
        """Test that Embeddings properly caches video embeddings."""

        class CountingVideoDataset:
            def __init__(self):
                self.access_count = 0

            def __len__(self):
                return 3

            def __getitem__(self, idx):
                self.access_count += 1
                return make_video(NUM_FRAMES), 0, {}

        dataset = CountingVideoDataset()
        embeddings = Embeddings(dataset, extractor=extractor, batch_size=2)

        # First access
        _ = embeddings[:]
        first_count = dataset.access_count

        # Second access should use cache
        _ = embeddings[:]
        second_count = dataset.access_count

        # Access count is 4: 1 during Embeddings initialization (dataset validation) + 3 during first compute access
        assert first_count == second_count == 4
