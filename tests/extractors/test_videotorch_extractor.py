"""Tests for VideoTorchExtractor."""

import numpy as np
import pytest
import torch

from dataeval import Embeddings
from dataeval.extractors import VideoTorchExtractor
from dataeval.protocols import FeatureExtractor


# Mock VideoMAE-like model for testing
class MockVideoModel(torch.nn.Module):
    """Mock video transformer model that mimics HuggingFace structure."""

    def __init__(self, hidden_size: int = 768, num_frames: int = 16):
        super().__init__()
        self.config = type("Config", (), {"num_frames": num_frames, "hidden_size": hidden_size})()
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(num_frames * 224 * 224 * 3, hidden_size),
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


@pytest.mark.required
class TestVideoTorchExtractorInit:
    """Test VideoTorchExtractor initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        model = MockVideoModel()
        processor = MockProcessor()
        extractor = VideoTorchExtractor(model, processor=processor)
        assert extractor.device is not None
        assert extractor.layer_name is None
        assert extractor.use_output is True
        assert extractor.pooling == "first"
        assert extractor.clip_aggregation == "mean"
        assert extractor.num_frames == 16

    def test_init_with_device(self):
        """Test initialization with specified device."""
        model = MockVideoModel()
        processor = MockProcessor()
        extractor = VideoTorchExtractor(model, processor=processor, device="cpu")
        assert extractor.device == torch.device("cpu")

    def test_init_auto_num_frames_from_config(self):
        """Test that num_frames is automatically extracted from model.config."""
        model = MockVideoModel(num_frames=32)
        processor = MockProcessor()
        extractor = VideoTorchExtractor(model, processor=processor)
        assert extractor.num_frames == 32

    def test_init_manual_num_frames(self):
        """Test initialization with manually specified num_frames."""
        model = MockVideoModel()
        processor = MockProcessor()
        extractor = VideoTorchExtractor(model, processor=processor, num_frames=8)
        assert extractor.num_frames == 8

    def test_init_without_num_frames_raises(self):
        """Test that missing num_frames raises ValueError."""
        model = torch.nn.Sequential(torch.nn.Flatten())  # No config
        processor = MockProcessor()
        with pytest.raises(ValueError, match="num_frames must be provided"):
            VideoTorchExtractor(model, processor=processor)

    def test_init_with_layer_name(self):
        """Test initialization with layer extraction."""
        model = MockVideoModel()
        processor = MockProcessor()
        extractor = VideoTorchExtractor(model, processor=processor, layer_name="encoder.0")
        assert extractor.layer_name == "encoder.0"

    def test_init_with_invalid_layer_raises(self):
        """Test that invalid layer name raises ValueError."""
        model = MockVideoModel()
        processor = MockProcessor()
        with pytest.raises(ValueError, match="Invalid layer"):
            VideoTorchExtractor(model, processor=processor, layer_name="nonexistent")

    def test_init_with_pooling_options(self):
        """Test initialization with different pooling strategies."""
        model = MockVideoModel()
        processor = MockProcessor()

        for pooling in ["mean", "first", "last", "none"]:
            extractor = VideoTorchExtractor(model, processor=processor, pooling=pooling)
            assert extractor.pooling == pooling

    def test_init_with_invalid_pooling_raises(self):
        """Test that invalid pooling raises ValueError."""
        model = MockVideoModel()
        processor = MockProcessor()
        with pytest.raises(ValueError, match="Invalid pooling"):
            VideoTorchExtractor(model, processor=processor, pooling="invalid")

    def test_init_with_clip_aggregation_options(self):
        """Test initialization with different clip aggregation strategies."""
        model = MockVideoModel()
        processor = MockProcessor()

        for aggregation in ["mean", "max"]:
            extractor = VideoTorchExtractor(model, processor=processor, clip_aggregation=aggregation)
            assert extractor.clip_aggregation == aggregation

    def test_init_with_invalid_clip_aggregation_raises(self):
        """Test that invalid clip_aggregation raises ValueError."""
        model = MockVideoModel()
        processor = MockProcessor()
        with pytest.raises(ValueError, match="Invalid clip_aggregation"):
            VideoTorchExtractor(model, processor=processor, clip_aggregation="invalid")

    def test_init_with_transforms(self):
        """Test initialization with transforms."""

        class MockTransform:
            def __call__(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2

        model = MockVideoModel()
        processor = MockProcessor()
        extractor = VideoTorchExtractor(model, processor=processor, transforms=MockTransform())
        assert len(extractor._transforms) == 1


@pytest.mark.required
class TestVideoTorchExtractorCall:
    """Test VideoTorchExtractor.__call__ method."""

    @pytest.fixture
    def extractor(self):
        """Create a simple extractor for testing."""
        model = MockVideoModel(hidden_size=128, num_frames=16)
        processor = MockProcessor()
        return VideoTorchExtractor(model, processor=processor, device="cpu", pooling="first", clip_aggregation="mean")

    def test_call_single_short_video(self, extractor):
        """Test extracting features from a single short video (one clip)."""
        # Video with exactly 16 frames (one clip)
        video = np.random.rand(16, 224, 224, 3).astype(np.float32)
        result = extractor([video])

        assert result.shape[0] == 1  # One video
        assert result.shape[1] == 128  # Hidden size
        assert isinstance(result, np.ndarray)

    def test_call_single_long_video(self, extractor):
        """Test extracting features from a single long video (multiple clips)."""
        # Video with 48 frames (3 complete clips)
        video = np.random.rand(48, 224, 224, 3).astype(np.float32)
        result = extractor([video])

        assert result.shape[0] == 1  # One video
        assert result.shape[1] == 128  # Hidden size

    def test_call_multiple_videos(self, extractor):
        """Test extracting features from multiple videos."""
        videos = [
            np.random.rand(16, 224, 224, 3).astype(np.float32),
            np.random.rand(32, 224, 224, 3).astype(np.float32),
            np.random.rand(16, 224, 224, 3).astype(np.float32),
        ]
        result = extractor(videos)

        assert result.shape[0] == 3  # Three videos
        assert result.shape[1] == 128  # Hidden size

    def test_call_with_incomplete_clips(self, extractor):
        """Test that incomplete clips are skipped."""
        # Video with 20 frames (1 complete clip + 4 incomplete frames)
        video = np.random.rand(20, 224, 224, 3).astype(np.float32)
        result = extractor([video])

        # Should only process the complete clip
        assert result.shape[0] == 1
        assert result.shape[1] == 128

    def test_call_empty_list(self, extractor):
        """Test extracting features from empty list."""
        result = extractor([])
        assert result.shape[0] == 0

    def test_call_with_generator(self, extractor):
        """Test that extractor works with generators (lazy loading)."""

        def video_generator():
            for _ in range(3):
                yield np.random.rand(16, 224, 224, 3).astype(np.float32)

        result = extractor(video_generator())
        assert result.shape[0] == 3
        assert result.shape[1] == 128


@pytest.mark.required
class TestVideoTorchExtractorClipAggregation:
    """Test different clip aggregation strategies."""

    def test_mean_aggregation(self):
        """Test mean aggregation across clips."""
        model = MockVideoModel(hidden_size=64, num_frames=8)
        processor = MockProcessor()
        extractor = VideoTorchExtractor(model, processor=processor, device="cpu", clip_aggregation="mean")

        # Video with 24 frames (3 clips)
        video = np.random.rand(24, 224, 224, 3).astype(np.float32)
        result = extractor([video])

        assert result.shape == (1, 64)

    def test_max_aggregation(self):
        """Test max aggregation across clips."""
        model = MockVideoModel(hidden_size=64, num_frames=8)
        processor = MockProcessor()
        extractor = VideoTorchExtractor(model, processor=processor, device="cpu", clip_aggregation="max")

        # Video with 24 frames (3 clips)
        video = np.random.rand(24, 224, 224, 3).astype(np.float32)
        result = extractor([video])

        assert result.shape == (1, 64)

    # NOTE: since the test video is random pixels, mean and max might be pretty close
    def test_mean_vs_max_different_results(self):
        """Test that mean and max produce different results."""
        model = MockVideoModel(hidden_size=32, num_frames=8)
        processor = MockProcessor()

        extractor_mean = VideoTorchExtractor(model, processor=processor, device="cpu", clip_aggregation="mean")
        extractor_max = VideoTorchExtractor(model, processor=processor, device="cpu", clip_aggregation="max")

        # Same video for both
        video = np.random.rand(48, 224, 224, 3).astype(np.float32)

        result_mean = extractor_mean([video])
        result_max = extractor_max([video])

        # Results should be different
        assert not np.allclose(result_mean, result_max)


@pytest.mark.required
class TestVideoTorchExtractorPooling:
    """Test different pooling strategies within clips."""

    def test_first_pooling(self):
        """Test first token (CLS) pooling."""
        model = MockVideoModel(hidden_size=32, num_frames=16)
        processor = MockProcessor()
        extractor = VideoTorchExtractor(model, processor=processor, device="cpu", pooling="first")

        video = np.random.rand(16, 224, 224, 3).astype(np.float32)
        result = extractor([video])
        assert result.shape == (1, 32)

    def test_mean_pooling(self):
        """Test mean pooling."""
        model = MockVideoModel(hidden_size=32, num_frames=16)
        processor = MockProcessor()
        extractor = VideoTorchExtractor(model, processor=processor, device="cpu", pooling="mean")

        video = np.random.rand(16, 224, 224, 3).astype(np.float32)
        result = extractor([video])
        assert result.shape == (1, 32)

    def test_last_pooling(self):
        """Test last token pooling."""
        model = MockVideoModel(hidden_size=32, num_frames=16)
        processor = MockProcessor()
        extractor = VideoTorchExtractor(model, processor=processor, device="cpu", pooling="last")

        video = np.random.rand(16, 224, 224, 3).astype(np.float32)
        result = extractor([video])
        assert result.shape == (1, 32)


@pytest.mark.required
class TestVideoTorchExtractorLayerExtraction:
    """Test layer extraction functionality."""

    def test_extract_intermediate_layer_output(self):
        """Test extracting output from intermediate layer."""
        model = MockVideoModel(hidden_size=128, num_frames=16)
        processor = MockProcessor()
        extractor = VideoTorchExtractor(
            model,
            processor=processor,
            layer_name="encoder.1",  # Linear layer
            device="cpu",
        )

        video = np.random.rand(16, 224, 224, 3).astype(np.float32)
        result = extractor([video])
        assert result.shape[0] == 1
        assert result.shape[1] == 128

    def test_extract_intermediate_layer_input(self):
        """Test extracting input to intermediate layer."""
        model = MockVideoModel(hidden_size=128, num_frames=16)
        processor = MockProcessor()
        extractor = VideoTorchExtractor(
            model, processor=processor, layer_name="encoder.1", use_output=False, device="cpu"
        )

        video = np.random.rand(16, 224, 224, 3).astype(np.float32)
        result = extractor([video])
        assert result.shape[0] == 1
        # Input to linear layer should be flattened size
        assert result.shape[1] == 16 * 224 * 224 * 3


@pytest.mark.required
class TestVideoTorchExtractorTransforms:
    """Test transform functionality."""

    def test_transforms_applied(self):
        """Test that transforms are applied during extraction."""

        class DoubleTransform:
            def __call__(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2

        model = MockVideoModel(hidden_size=64, num_frames=16)
        processor = MockProcessor()

        extractor_no_transform = VideoTorchExtractor(model, processor=processor, device="cpu")
        extractor_with_transform = VideoTorchExtractor(
            model, processor=processor, transforms=DoubleTransform(), device="cpu"
        )

        # Use constant input for predictable results
        video = np.ones((16, 224, 224, 3), dtype=np.float32)

        result_no_transform = extractor_no_transform([video])
        result_with_transform = extractor_with_transform([video])

        # Results should differ due to transform
        assert not np.allclose(result_no_transform, result_with_transform)


@pytest.mark.required
class TestVideoTorchExtractorEdgeCases:
    """Test edge cases and error handling."""

    # TODO: adjust this behavior later (some models allow up to 90+ frames in a clip which is a lot to skip)
    def test_video_with_only_incomplete_clip(self):
        """Test video shorter than num_frames produces no output."""
        model = MockVideoModel(num_frames=16)
        processor = MockProcessor()
        extractor = VideoTorchExtractor(model, processor=processor, device="cpu")

        # Video with only 8 frames (less than required 16)
        video = np.random.rand(8, 224, 224, 3).astype(np.float32)

        # Should skip this video and return empty
        result = extractor([video])
        assert result.shape[0] == 0

    def test_very_long_video(self):
        """Test that very long videos are processed correctly."""
        model = MockVideoModel(hidden_size=32, num_frames=16)
        processor = MockProcessor()
        extractor = VideoTorchExtractor(model, processor=processor, device="cpu", clip_aggregation="mean")

        # Very long video (100 clips)
        video = np.random.rand(1600, 224, 224, 3).astype(np.float32)
        result = extractor([video])

        assert result.shape == (1, 32)

    def test_mixed_video_lengths(self):
        """Test processing videos of different lengths."""
        model = MockVideoModel(hidden_size=32, num_frames=8)
        processor = MockProcessor()
        extractor = VideoTorchExtractor(model, processor=processor, device="cpu")

        videos = [
            np.random.rand(8, 224, 224, 3).astype(np.float32),  # 1 clip
            np.random.rand(24, 224, 224, 3).astype(np.float32),  # 3 clips
            np.random.rand(40, 224, 224, 3).astype(np.float32),  # 5 clips
        ]

        result = extractor(videos)
        assert result.shape == (3, 32)


@pytest.mark.required
class TestVideoTorchExtractorRepr:
    """Test __repr__ method."""

    def test_repr_basic(self):
        """Test basic repr."""
        model = MockVideoModel()
        processor = MockProcessor()
        extractor = VideoTorchExtractor(model, processor=processor, device="cpu")
        repr_str = repr(extractor)

        assert "VideoTorchExtractor" in repr_str
        assert "cpu" in repr_str
        assert "num_frames=16" in repr_str
        assert "clip_aggregation='mean'" in repr_str

    def test_repr_with_layer_name(self):
        """Test repr includes layer name when set."""
        model = MockVideoModel()
        processor = MockProcessor()
        extractor = VideoTorchExtractor(model, processor=processor, layer_name="encoder.0", device="cpu")
        repr_str = repr(extractor)
        assert "layer_name='encoder.0'" in repr_str

    def test_repr_with_processor(self):
        """Test repr indicates processor is present."""
        model = MockVideoModel()
        processor = MockProcessor()
        extractor = VideoTorchExtractor(model, processor=processor, device="cpu")
        repr_str = repr(extractor)
        assert "processor=True" in repr_str


@pytest.mark.required
class TestVideoTorchExtractorProtocol:
    """Test that VideoTorchExtractor conforms to FeatureExtractor protocol."""

    def test_protocol_conformance(self):
        """Test that VideoTorchExtractor implements FeatureExtractor protocol."""
        model = MockVideoModel()
        processor = MockProcessor()
        extractor = VideoTorchExtractor(model, processor=processor)

        assert isinstance(extractor, FeatureExtractor)
        assert callable(extractor)

    def test_returns_array_protocol(self):
        """Test that __call__ returns Array-like object."""
        model = MockVideoModel(hidden_size=32, num_frames=16)
        processor = MockProcessor()
        extractor = VideoTorchExtractor(model, processor=processor, device="cpu")

        video = np.random.rand(16, 224, 224, 3).astype(np.float32)
        result = extractor([video])

        # Should be numpy array (implements Array protocol)
        assert isinstance(result, np.ndarray)
        assert hasattr(result, "shape")
        assert hasattr(result, "dtype")


class TestVideoTorchExtractorIntegration:
    """Integration tests with Embeddings class."""

    def test_works_with_embeddings_class(self):
        """Test that VideoTorchExtractor can be used with Embeddings."""

        # Create mock video dataset
        class MockVideoDataset:
            def __init__(self, n_videos=10, n_frames=16):
                self.n_videos = n_videos
                self.n_frames = n_frames

            def __len__(self):
                return self.n_videos

            def __getitem__(self, idx):
                # Return (video, label, metadata) tuple
                video = np.random.rand(self.n_frames, 224, 224, 3).astype(np.float32)
                label = idx % 3  # Mock labels
                metadata = {"video_id": idx}
                return video, label, metadata

        # Create extractor
        model = MockVideoModel(hidden_size=64, num_frames=16)
        processor = MockProcessor()
        extractor = VideoTorchExtractor(model, processor=processor, device="cpu", clip_aggregation="mean")

        # Create dataset
        dataset = MockVideoDataset(n_videos=10, n_frames=16)

        # Create Embeddings instance
        embeddings = Embeddings(dataset, extractor=extractor, batch_size=4)

        # Test basic operations
        assert len(embeddings) == 10

        # Test single embedding access
        single = embeddings[0]
        assert single.shape == (64,)

        # Test batch access
        batch = embeddings[0:5]
        assert batch.shape == (5, 64)

        # Test full array access
        all_embeddings = embeddings[:]
        assert all_embeddings.shape == (10, 64)

    def test_embeddings_with_long_videos(self):
        """Test Embeddings with videos that have multiple clips."""

        class MockLongVideoDataset:
            def __len__(self):
                return 5

            def __getitem__(self, idx):
                # Videos with varying lengths (multiple clips)
                n_frames = 32 + (idx * 16)  # 32, 48, 64, 80, 96 frames
                video = np.random.rand(n_frames, 224, 224, 3).astype(np.float32)
                return video, idx, {}

        model = MockVideoModel(hidden_size=32, num_frames=16)
        processor = MockProcessor()
        extractor = VideoTorchExtractor(model, processor=processor, device="cpu", clip_aggregation="mean")

        dataset = MockLongVideoDataset()
        embeddings = Embeddings(dataset, extractor=extractor, batch_size=2)

        # All videos should produce embeddings despite different lengths
        all_embeddings = embeddings[:]
        assert all_embeddings.shape == (5, 32)

    def test_embeddings_compute_and_cache(self):
        """Test that Embeddings properly caches video embeddings."""

        class CountingVideoDataset:
            def __init__(self):
                self.access_count = 0

            def __len__(self):
                return 3

            def __getitem__(self, idx):
                self.access_count += 1
                video = np.random.rand(16, 224, 224, 3).astype(np.float32)
                return video, 0, {}

        dataset = CountingVideoDataset()
        model = MockVideoModel(hidden_size=32, num_frames=16)
        processor = MockProcessor()
        extractor = VideoTorchExtractor(model, processor=processor, device="cpu")

        embeddings = Embeddings(dataset, extractor=extractor, batch_size=2)

        # First access
        _ = embeddings[:]
        first_count = dataset.access_count

        # Second access should use cache
        _ = embeddings[:]
        second_count = dataset.access_count

        # Access count should be the same (cached)
        assert first_count == second_count == 3
