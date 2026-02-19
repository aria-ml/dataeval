"""PyTorch-based video feature extractor for pretrained PyTorch-based models."""

__all__ = ["VideoTorchExtractor"]

import logging
import traceback
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from torch.amp.autocast_mode import autocast

from dataeval.config import get_device
from dataeval.protocols import Array, DeviceLike, Transform

_logger = logging.getLogger(__name__)


class VideoTorchExtractor:
    """
    Extracts embeddings from videos using a pretrained model.

    Videos are automatically split into non-overlapping clips of the required length,
    processed separately, then aggregated into a single embedding per video.

    Encapsulates all PyTorch-specific logic for video feature extraction:

    - Model management (PreTrainedModel, e.g. from HuggingFace transformers)
    - Processor/feature extractor integration
    - Device handling
    - Transform pipeline (applied after processor)
    - Layer hooking for intermediate layer extraction
    - Clip-based processing for long videos
    - Clip aggregation strategies

    Implements the :class:`~dataeval.protocols.FeatureExtractor` protocol.

    Parameters
    ----------
    model : torch.nn.Module
        Pretrained torch model for video feature extraction
        (e.g., VideoMAEModel from transformers).
    processor : Any, optional
        HuggingFace processor or feature extractor for preprocessing videos.
        When None, videos must be preprocessed externally.
    transforms : Transform or Sequence[Transform] or None, default None
        Additional preprocessing transforms to apply after the processor.
        When None, only the processor is used.
    device : DeviceLike or None, default None
        Device for computation. When None, uses DataEval's configured device.
    layer_name : str or None, default None
        Layer to extract embeddings from. When None, uses model output.
    use_output : bool, default True
        If True, captures layer output; if False, captures layer input.
        Only used when layer_name is specified.
    pooling : str, default "first"
        Pooling strategy for sequence outputs within each clip. Options:
        - "mean": Average pool across temporal dimension
        - "first": Use first token (CLS token for BERT-style models)
        - "last": Use last token
        - "none": Return full sequence
    num_frames : int or None, default None
        Number of frames per clip. When None, automatically extracted from
        model.config.num_frames. Must be set if model doesn't have this config.
    clip_aggregation : str, default "mean"
        Strategy for aggregating clip embeddings into video embedding. Options:
        - "mean": Average all clip embeddings
        - "max": Max pool across clip embeddings (strongest activation for each feature preserved)


    Example
    -------
    Basic usage with VideoMAE:

    >>> # from transformers import VideoMAEImageProcessor, VideoMAEModel
    >>> from dataeval import Embeddings
    >>> from dataeval.extractors import VideoTorchExtractor
    >>>
    >>> # video_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-large")
    >>> # video_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-large")
    >>> device = "cuda" if torch.cuda.is_available() else "cpu"
    >>> extractor = VideoTorchExtractor(
    ...     video_model, processor=video_processor, device=device, pooling="first", clip_aggregation="mean"
    ... )
    >>> embeddings = Embeddings(video_dataset, extractor=extractor, batch_size=4)

    Extracting from an intermediate layer:

    >>> extractor = VideoTorchExtractor(
    ...     video_model,
    ...     processor=video_processor,
    ...     device=device,
    ...     layer_name="encoder.1",
    ...     use_output=True,
    ...     pooling="first",
    ...     clip_aggregation="mean",
    ... )
    """

    device: torch.device

    def __init__(
        self,
        model: torch.nn.Module,
        processor: Any | None = None,
        transforms: Transform[torch.Tensor] | Iterable[Transform[torch.Tensor]] | None = None,
        device: DeviceLike | None = None,
        layer_name: str | None = None,
        use_output: bool = True,
        pooling: str = "first",
        num_frames: int | None = None,
        clip_aggregation: str = "mean",
        use_amp: bool = False,
    ) -> None:
        self.device = get_device(device)
        self._processor = processor
        self._transforms = self._normalize_transforms(transforms)
        self._layer_name = layer_name
        self._use_output = use_output
        self._pooling = pooling
        self._clip_aggregation = clip_aggregation
        self._use_amp = use_amp

        # Validate pooling strategy
        valid_pooling = {"mean", "first", "last", "none"}
        if pooling not in valid_pooling:
            raise ValueError(f"Invalid pooling '{pooling}'. Must be one of {valid_pooling}")

        # Validate clip aggregation
        valid_aggregation = {"mean", "max"}
        if clip_aggregation not in valid_aggregation:
            raise ValueError(f"Invalid clip_aggregation '{clip_aggregation}'. Must be one of {valid_aggregation}")

        # Setup model
        self._model = model.to(self.device).eval()

        # Ensure cuDNN is properly initialized
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        # Get num_frames from model config if not provided
        if num_frames is None:
            config = getattr(model, "config", None)
            self._num_frames = getattr(config, "num_frames", None) if config is not None else None
            _logger.debug(f"Using num_frames={self._num_frames} from model.config")

            if self._num_frames is None:
                raise ValueError("num_frames must be provided")

        elif num_frames is not None:
            self._num_frames = num_frames
        else:
            raise ValueError("num_frames must be provided or available in model.config.num_frames")

        # Setup hook for intermediate layer extraction
        self._captured_output: Any = None
        if layer_name is not None:
            target_layer = self._get_valid_layer(layer_name, model)
            target_layer.register_forward_hook(self._hook_fn)
            _logger.debug(f"Capturing {'output' if use_output else 'input'} data from layer {layer_name}.")

    @property
    def layer_name(self) -> str | None:
        """Return the layer name for intermediate extraction, if set."""
        return self._layer_name

    @property
    def use_output(self) -> bool:
        """Return whether output (True) or input (False) is captured from the layer."""
        return self._use_output

    @property
    def pooling(self) -> str:
        """Return the pooling strategy."""
        return self._pooling

    @property
    def clip_aggregation(self) -> str:
        """Return the clip aggregation strategy."""
        return self._clip_aggregation

    @property
    def num_frames(self) -> int | None:
        """Return the number of frames per clip."""
        return self._num_frames

    def _normalize_transforms(
        self, transforms: Transform[torch.Tensor] | Iterable[Transform[torch.Tensor]] | None
    ) -> list[Transform[torch.Tensor]]:
        """Normalize transforms to a list."""
        if transforms is None:
            return []
        if isinstance(transforms, Transform):
            return [transforms]
        return list(transforms)

    def _hook_fn(self, _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: Any) -> None:
        """Forward hook to capture layer input or output."""
        if self._use_output:
            # Handle different output types (tensor, tuple, dict, BaseModelOutput)
            if isinstance(output, torch.Tensor):
                self._captured_output = output.detach().clone()
            elif isinstance(output, tuple):
                self._captured_output = output[0].detach().clone()
            elif hasattr(output, "last_hidden_state"):
                self._captured_output = output.last_hidden_state.detach().clone()
            elif isinstance(output, dict) and "last_hidden_state" in output:
                self._captured_output = output["last_hidden_state"].detach().clone()
            else:
                self._captured_output = output
        else:
            self._captured_output = inputs[0].detach().clone()

    def _get_valid_layer(self, layer_name: str, model: torch.nn.Module) -> torch.nn.Module:
        """Validate and return the target layer for hook registration."""
        modules_dict = dict(model.named_modules())

        if layer_name not in modules_dict:
            formatted_layers = "\n".join(f"  {layer}" for layer in modules_dict)
            raise ValueError(f"Invalid layer '{layer_name}'. Available layers are:\n{formatted_layers}")

        return modules_dict[layer_name]

    def _apply_pooling(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply pooling strategy to sequence embeddings.

        Parameters
        ----------
        embeddings : torch.Tensor
            Tensor of shape (batch_size, sequence_length, hidden_dim)
            or (batch_size, hidden_dim).

        Returns
        -------
        torch.Tensor
            Pooled embeddings.
        """
        # If already 2D, no pooling needed
        if embeddings.ndim == 2:
            return embeddings

        if self._pooling == "mean":
            return embeddings.mean(dim=1)
        if self._pooling == "first":
            return embeddings[:, 0]
        if self._pooling == "last":
            return embeddings[:, -1]
        if self._pooling == "none":
            # Flatten batch and sequence dimensions
            batch_size, seq_len, hidden_dim = embeddings.shape
            return embeddings.reshape(batch_size * seq_len, hidden_dim)
        raise ValueError(f"Unknown pooling strategy: {self._pooling}")

    def _preprocess_clip(self, clip_frames: list) -> torch.Tensor:
        """
        Preprocess a single clip through processor and transforms.

        Parameters
        ----------
        clip_frames : list
            List of frames for a single clip (length = num_frames).

        Returns
        -------
        torch.Tensor
            Preprocessed clip tensor ready for the model.
        """
        if self._processor is not None:
            # Process the clip
            processed = self._processor(clip_frames, return_tensors="pt")

            # Extract tensor from processor output
            if hasattr(processed, "pixel_values"):
                tensor = processed.pixel_values.squeeze(0)  # Remove batch dim
            elif "pixel_values" in processed:
                tensor = processed["pixel_values"].squeeze(0)
            else:
                tensor = processed
        else:
            # Convert to tensor if no processor
            tensor = torch.as_tensor(clip_frames)

        # Apply additional transforms
        for transform in self._transforms:
            tensor = transform(tensor)

        return tensor.contiguous()

    def _split_video_into_clips(self, video: Any) -> list[list]:
        """
        Split a video into non-overlapping clips.

        Parameters
        ----------
        video : Any
            Video data as numpy array of shape (num_frames, height, width, channels)
            or list of frames.

        Returns
        -------
        list[list]
            List of clips, where each clip is a list of frames.
        """
        if self._num_frames is None:
            raise RuntimeError("num_frames was not properly initialized")

        # Convert to list of frames if numpy array
        video_frames = list(video) if isinstance(video, np.ndarray) and video.ndim == 4 else video

        # Split into non-overlapping clips
        clips = []
        for start_idx in range(0, len(video_frames), self._num_frames):
            clip_frames = video_frames[start_idx : start_idx + self._num_frames]

            # Only keep complete clips
            if len(clip_frames) == self._num_frames:
                clips.append(clip_frames)
            else:
                _logger.debug(f"Skipping incomplete clip with {len(clip_frames)}/{self._num_frames} frames")

        return clips

    def _aggregate_clips_incremental(self, clips: list[list]) -> torch.Tensor:
        """
        Process and aggregate clips incrementally to minimize memory usage.

        This does slow things down, and could be sped up a bit
        by iterating through batches rather than individual clips.

        Parameters
        ----------
        clips : list[list]
            List of clips, where each clip is a list of frames.

        Returns
        -------
        torch.Tensor
            Aggregated video embedding.
        """
        if len(clips) == 0:
            raise ValueError("No clips to process")

        running_aggregate = None
        n_clips = len(clips)

        for clip_idx, clip_frames in enumerate(clips):
            # Preprocess and extract embedding for this clip
            clip_tensor = self._preprocess_clip(clip_frames)
            clip_embedding = self._extract_clip_embedding(clip_tensor)

            # Update running aggregate
            if running_aggregate is None:
                running_aggregate = clip_embedding.clone()  # Keep on GPU
            else:
                if self._clip_aggregation == "mean":
                    running_aggregate.add_(clip_embedding)  # In-place addition
                elif self._clip_aggregation == "max":
                    running_aggregate = torch.maximum(running_aggregate, clip_embedding)

            # delete no-longer-needed references
            del clip_tensor, clip_embedding

            # clear cache periodically for very long videos
            if n_clips > 500 and (clip_idx + 1) % 100 == 0:
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                _logger.debug(f"Processed {clip_idx + 1}/{n_clips} clips")
        if running_aggregate is None:
            raise RuntimeError("Unexpected error: running_aggregate is None after processing clips")

        # Finalize aggregation
        if self._clip_aggregation == "mean":
            running_aggregate.div_(n_clips)  # In-place division

        return running_aggregate

    def _extract_clip_embedding(self, clip_tensor: torch.Tensor) -> torch.Tensor:  # noqa: C901
        """
        Extract embedding from a single preprocessed clip.

        Parameters
        ----------
        clip_tensor : torch.Tensor
            Preprocessed clip tensor.

        Returns
        -------
        torch.Tensor
            Clip embedding of shape (embedding_dim,).
        """
        clip_tensor = clip_tensor.unsqueeze(0).to(self.device)  # Add batch dim

        with torch.no_grad():
            if self._use_amp and self.device.type == "cuda":
                with autocast("cuda"):
                    if self._layer_name:
                        _ = self._model(clip_tensor)
                        output = self._captured_output
                    else:
                        model_output = self._model(clip_tensor)

                        if isinstance(model_output, torch.Tensor):
                            output = model_output
                        elif hasattr(model_output, "last_hidden_state"):
                            output = model_output.last_hidden_state
                        elif hasattr(model_output, "pooler_output") and model_output.pooler_output is not None:
                            output = model_output.pooler_output
                        elif isinstance(model_output, dict) and "last_hidden_state" in model_output:
                            output = model_output["last_hidden_state"]
                        elif isinstance(model_output, tuple):
                            output = model_output[0]
                        else:
                            raise ValueError(f"Unsupported model output type: {type(model_output)}")
            else:
                if self._layer_name:
                    _ = self._model(clip_tensor)
                    output = self._captured_output
                else:
                    model_output = self._model(clip_tensor)

                    if isinstance(model_output, torch.Tensor):
                        output = model_output
                    elif hasattr(model_output, "last_hidden_state"):
                        output = model_output.last_hidden_state
                    elif hasattr(model_output, "pooler_output") and model_output.pooler_output is not None:
                        output = model_output.pooler_output
                    elif isinstance(model_output, dict) and "last_hidden_state" in model_output:
                        output = model_output["last_hidden_state"]
                    elif isinstance(model_output, tuple):
                        output = model_output[0]
                    else:
                        raise ValueError(f"Unsupported model output type: {type(model_output)}")

        # Apply pooling to get single embedding per clip
        return self._apply_pooling(output).squeeze(0)  # Remove batch dim

    def __call__(self, data: Any) -> Array:
        """
        Extract features from a batch of videos.

        Each video is split into non-overlapping clips, processed separately,
        then aggregated into a single embedding per video.

        Parameters
        ----------
        data : Any
            Iterable of videos to extract features from. Each video should be
            in a format compatible with the processor (e.g., numpy array of shape
            (num_frames, height, width, channels) or list of PIL Images).

        Returns
        -------
        Array
            Embeddings array of shape (n_videos, embedding_dim).
        """
        all_video_embeddings = []

        for video_idx, video in enumerate(data):
            try:
                # Split video into clips
                clips = self._split_video_into_clips(video)

                if not clips:
                    _logger.warning("No valid clips extracted from video")
                    continue

                # Log for very long videos
                if len(clips) > 100:
                    _logger.info(f"Processing long video {video_idx + 1} with {len(clips)} clips")

                # Process clips incrementally
                video_embedding = self._aggregate_clips_incremental(clips)
                all_video_embeddings.append(video_embedding.cpu())

                # Log progress periodically
                if (video_idx + 1) % 10 == 0:
                    _logger.info(f"Processed {video_idx + 1} videos")

            except Exception as e:  # noqa: BLE001
                _logger.warning(f"Failed to process video {video_idx + 1}: {e}")
                _logger.debug(traceback.format_exc())
                continue

        if not all_video_embeddings:
            return np.empty((0,), dtype=np.float32)

        # Stack all video embeddings
        return torch.stack(all_video_embeddings).numpy()

    def __repr__(self) -> str:
        layer_info = f", layer_name={self._layer_name!r}" if self._layer_name else ""
        processor_info = ", processor=True" if self._processor is not None else ""
        pooling_info = f", pooling={self._pooling!r}"
        clip_info = f", num_frames={self._num_frames}, clip_aggregation={self._clip_aggregation!r}"
        return f"VideoTorchExtractor(device={self.device}{layer_info}{processor_info}{pooling_info}{clip_info})"
