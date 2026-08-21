"""PyTorch-based video feature extractor for pretrained PyTorch-based models."""

__all__ = ["VideoTorchExtractor"]

from collections.abc import Callable, Iterable
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
from torch.amp.autocast_mode import autocast

from dataeval._log import get_logger
from dataeval.config import get_device
from dataeval.extractors._torch import get_valid_layer, normalize_transforms
from dataeval.protocols import Array, DeviceLike, Transform

_logger = get_logger(__name__)

# One source of truth for the strategy names: the constructor validates against these keys
# rather than against a second literal set that could drift from what is implemented.
_POOLING: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "mean": lambda embeddings: embeddings.mean(dim=1),
    "first": lambda embeddings: embeddings[:, 0],
    "last": lambda embeddings: embeddings[:, -1],
    # Keep every position, as one row per batch entry.
    "none": lambda embeddings: embeddings.reshape(embeddings.shape[0], -1),
}

# ``add_`` accumulates into the running aggregate and ``maximum`` returns a fresh tensor;
# both answer to ``combine(running, clip) -> running``, so the loop needs no branch.
_CLIP_AGGREGATION: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "mean": torch.Tensor.add_,
    "max": torch.maximum,
}

# Beyond this many clips a video is long enough that releasing cached CUDA blocks now and
# then is worth the synchronization, and that progress is worth reporting.
_LONG_VIDEO_CLIPS = 100
_CACHE_RELEASE_CLIPS = 500
_CACHE_RELEASE_EVERY = 100


def _tensor_of(model_output: Any) -> Any:  # noqa: C901
    """Pull the embedding tensor out of whatever shape a model or a layer returned.

    A bare tensor, a HuggingFace ``BaseModelOutput``, a plain dict and a tuple are all in
    circulation, so the tensor is found by asking rather than by assuming. ``None`` comes
    back when none of those shapes fit, leaving what to do about it to the caller: the
    forward path refuses such an output, while the layer hook keeps it as it came.
    """
    if isinstance(model_output, torch.Tensor):
        return model_output
    if getattr(model_output, "last_hidden_state", None) is not None:
        return model_output.last_hidden_state
    if getattr(model_output, "pooler_output", None) is not None:
        return model_output.pooler_output
    if isinstance(model_output, dict) and "last_hidden_state" in model_output:
        return model_output["last_hidden_state"]
    if isinstance(model_output, tuple) and model_output:
        return model_output[0]
    return None


def _resolve_num_frames(model: torch.nn.Module, num_frames: int | None) -> int:
    """Settle on the clip length, from the argument or from the model's own config."""
    if num_frames is None:
        config = getattr(model, "config", None)
        num_frames = getattr(config, "num_frames", None) if config is not None else None
        if num_frames is None:
            raise ValueError("num_frames must be provided or available in model.config.num_frames")
        _logger.debug(f"Using num_frames={num_frames} from model.config")
    if num_frames < 1:
        raise ValueError(f"num_frames must be at least 1; got {num_frames}.")
    return int(num_frames)


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
        - "none": Keep the whole sequence, flattened into one vector per clip
    num_frames : int or None, default None
        Number of frames per clip. When None, automatically extracted from
        model.config.num_frames. Must be set if model doesn't have this config.
    clip_aggregation : str, default "mean"
        Strategy for aggregating clip embeddings into video embedding. Options:
        - "mean": Average all clip embeddings
        - "max": Max pool across clip embeddings (strongest activation for each feature preserved)
    use_amp : bool, default False
        Run the forward pass under ``torch.autocast`` mixed precision. Only takes
        effect on a CUDA device; ignored everywhere else.

    Raises
    ------
    ValueError
        When `pooling` or `clip_aggregation` names an unknown strategy, when `num_frames`
        is neither given nor available from ``model.config``, or when `layer_name` names
        no submodule of `model`.


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
        self._transforms = normalize_transforms(transforms)
        self._layer_name = layer_name
        self._use_output = use_output
        self._pooling = pooling
        self._clip_aggregation = clip_aggregation
        self._use_amp = use_amp

        # Validate pooling strategy
        if pooling not in _POOLING:
            raise ValueError(f"Invalid pooling '{pooling}'. Must be one of {set(_POOLING)}")

        # Validate clip aggregation
        if clip_aggregation not in _CLIP_AGGREGATION:
            raise ValueError(f"Invalid clip_aggregation '{clip_aggregation}'. Must be one of {set(_CLIP_AGGREGATION)}")

        # Setup model
        self._model = model.to(self.device).eval()
        self._num_frames = _resolve_num_frames(model, num_frames)

        # Setup hook for intermediate layer extraction
        self._captured_output: Any = None
        if layer_name is not None:
            target_layer = get_valid_layer(layer_name, model)
            target_layer.register_forward_hook(self._hook_fn)
            _logger.debug(f"Capturing {'output' if use_output else 'input'} data from layer {layer_name}.")

    @property
    def layer_name(self) -> str | None:
        """Layer name for intermediate extraction, if set."""
        return self._layer_name

    @property
    def use_output(self) -> bool:
        """Whether output (True) or input (False) is captured from the layer."""
        return self._use_output

    @property
    def pooling(self) -> str:
        """Pooling strategy applied within each clip."""
        return self._pooling

    @property
    def clip_aggregation(self) -> str:
        """Strategy used to aggregate clip embeddings into a video embedding."""
        return self._clip_aggregation

    @property
    def num_frames(self) -> int:
        """Number of frames per clip."""
        return self._num_frames

    def _hook_fn(self, _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: Any) -> None:
        """Forward hook to capture layer input or output."""
        captured = _tensor_of(output) if self._use_output else inputs[0]
        if captured is None:
            # Unrecognised output kept as it came, rather than discarded: the layer was
            # named by the caller, who knows what it produces better than this does.
            self._captured_output = output
        else:
            self._captured_output = captured.detach().clone()

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
            Pooled embeddings, batch dimension intact.

        Notes
        -----
        The dimension being pooled over is dimension 1 whatever follows it, so a layer
        capture with spatial dimensions -- ``(batch, channels, height, width)`` -- pools
        over channels and keeps the rest, which the caller then flattens. Unpacking the
        shape into exactly three names instead would refuse such a capture outright.
        """
        # If already 2D, no pooling needed
        if embeddings.ndim == 2:
            return embeddings

        return _POOLING[self._pooling](embeddings)

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
            # np.asarray first: torch.as_tensor on a list of arrays copies frame by frame
            # and warns about it, while one stacked array converts without a copy.
            tensor = torch.as_tensor(np.asarray(clip_frames))

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
        if not clips:
            raise ValueError("No clips to process")

        n_clips = len(clips)
        combine = _CLIP_AGGREGATION[self._clip_aggregation]
        # Cloned so the accumulator owns its memory: the pooled embedding can be a view of
        # the model's output, which the in-place ``mean`` combine would otherwise write to.
        running_aggregate = self._embed_clip(clips[0]).clone()

        for clip_idx, clip_frames in enumerate(clips[1:], start=2):
            clip_embedding = self._embed_clip(clip_frames)
            running_aggregate = combine(running_aggregate, clip_embedding)
            del clip_embedding  # no longer needed
            self._relieve_memory(clip_idx, n_clips)

        # Finalize aggregation
        if self._clip_aggregation == "mean":
            running_aggregate.div_(n_clips)  # In-place division

        return running_aggregate

    def _embed_clip(self, clip_frames: list) -> torch.Tensor:
        """Preprocess one clip and run it through the model."""
        return self._extract_clip_embedding(self._preprocess_clip(clip_frames))

    def _relieve_memory(self, processed: int, n_clips: int) -> None:
        """Release cached CUDA blocks now and then while working through a very long video."""
        if n_clips <= _CACHE_RELEASE_CLIPS or processed % _CACHE_RELEASE_EVERY:
            return
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        _logger.debug(f"Processed {processed}/{n_clips} clips")

    def _forward(self, clip_tensor: torch.Tensor) -> torch.Tensor:
        """Run one clip through the model and return the tensor to pool.

        With a hooked layer the value comes from the hook rather than from the return
        value, so the capture is cleared first: a forward pass that never reaches the
        layer would otherwise hand back the *previous* clip's activations as if they were
        this one's, and every clip after it would read as a copy of the last one that ran.
        """
        if self._layer_name:
            self._captured_output = None
            self._model(clip_tensor)
            if self._captured_output is None:
                raise RuntimeError(
                    f"Layer {self._layer_name!r} did not run during the forward pass, so there is "
                    "nothing to extract from it.",
                )
            return self._captured_output

        model_output = self._model(clip_tensor)
        output = _tensor_of(model_output)
        if output is None:
            raise ValueError(f"Unsupported model output type: {type(model_output)}")
        return output

    def _extract_clip_embedding(self, clip_tensor: torch.Tensor) -> torch.Tensor:
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

        # autocast is CUDA-only here, so the enabled case is the only one that needs it and
        # the disabled case costs nothing rather than entering a no-op autocast.
        amp = autocast("cuda") if self._use_amp and self.device.type == "cuda" else nullcontext()
        with torch.no_grad(), amp:
            output = self._forward(clip_tensor)

        # One row per clip whatever the pooled rank: the batch dimension is always 1 here,
        # and a caller indexes the result by video, so anything left over is flattened in.
        return self._apply_pooling(output).reshape(-1)

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
            Embeddings array of shape (n_videos, embedding_dim), one row per video in
            `data`, in the order they were given.

        Raises
        ------
        ValueError
            When a video is too short to fill even one clip, so it has no embedding.

        Notes
        -----
        The result is addressed **positionally** -- :class:`~dataeval.Embeddings` assigns
        row *i* to the *i*-th index of the batch it asked for -- so a video that produced
        no embedding cannot simply be left out: every later video's embedding would land
        on the wrong item, silently. A video with no complete clip is therefore an error
        rather than a skip, and so is a failure anywhere in the forward pass.
        """
        all_video_embeddings = []

        for video_idx, video in enumerate(data):
            # Split video into clips
            clips = self._split_video_into_clips(video)

            if not clips:
                raise ValueError(
                    f"Video at index {video_idx} is shorter than one clip of {self._num_frames} "
                    "frames, so it yields no embedding, and the result is read by position. "
                    "Drop the video from the dataset, or lower num_frames.",
                )

            # Log for very long videos
            if len(clips) > _LONG_VIDEO_CLIPS:
                _logger.info(f"Processing long video {video_idx + 1} with {len(clips)} clips")

            # Process clips incrementally
            video_embedding = self._aggregate_clips_incremental(clips)
            all_video_embeddings.append(video_embedding.cpu())

        if not all_video_embeddings:
            return np.empty((0,), dtype=np.float32)

        _logger.info(f"Processed {len(all_video_embeddings)} videos")
        # Stack all video embeddings
        return torch.stack(all_video_embeddings).numpy()

    def __repr__(self) -> str:
        layer_info = f", layer_name={self._layer_name!r}" if self._layer_name else ""
        processor_info = ", processor=True" if self._processor is not None else ""
        pooling_info = f", pooling={self._pooling!r}"
        clip_info = f", num_frames={self._num_frames}, clip_aggregation={self._clip_aggregation!r}"
        return f"VideoTorchExtractor(device={self.device}{layer_info}{processor_info}{pooling_info}{clip_info})"
