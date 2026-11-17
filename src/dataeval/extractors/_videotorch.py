"""PyTorch-based video feature extractor for pretrained PyTorch-based models."""

__all__ = ["VideoTorchExtractor"]

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import islice
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.amp.autocast_mode import autocast

from dataeval._log import get_logger
from dataeval.config import get_device
from dataeval.extractors._torch import _capture, get_valid_layer, normalize_transforms, warn_if_pulled_to_cpu
from dataeval.protocols import Array, DeviceLike, Transform
from dataeval.utils._array import as_numpy
from dataeval.utils.preprocessing import normalize_image_shape

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

# Past this many clips a video is long enough that releasing cached CUDA blocks now and
# then is worth the synchronization, and that progress is worth reporting. A stream's
# length is unknown until it ends, so both are decided by the clips read so far.
_LONG_VIDEO_CLIPS = 100
_CACHE_RELEASE_CLIPS = 500
_CACHE_RELEASE_EVERY = 100


def _frame_of(frame: Any) -> Any:
    """Return a stream frame's pixels as an ``(H, W, C)`` array, and any other frame as it came.

    A MAITE frame carries its pixels channel first, where an array video is channel last, so
    the stream frame is turned to match what the processor already receives from an array. A
    frame with no channel axis is read as one channel, as every other reader of stream frames
    reads it, so it arrives as a ``(T, H, W, 1)`` array video's frame does.
    """
    pixels = getattr(frame, "pixels", None)
    if pixels is None:
        return frame
    return np.moveaxis(normalize_image_shape(as_numpy(pixels)), 0, -1)


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


def _layer_value(captured: Any) -> Any:
    """Return the tensor in what a hooked layer gave, or what it gave as it came.

    An unrecognized output is kept rather than discarded: the layer was named by the caller,
    who knows what it produces better than this does.
    """
    tensor = _tensor_of(captured)
    return captured if tensor is None else tensor


@dataclass
class _Running:
    """One video's clip aggregate while its clips are pushed, and how many there were."""

    aggregate: torch.Tensor | None = None
    n_clips: int = 0


def _is_sliced(video: Any) -> bool:
    """Whether a video is cut into clips by slicing, as an array is unless it is a rank-4 NumPy array, not walked."""
    return isinstance(video, Array) and not (isinstance(video, np.ndarray) and video.ndim == 4)


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
    processed separately, then aggregated into a single embedding per video. A video
    can be a (num_frames, height, width, channels) numpy array, a list of frames, or
    a MAITE video stream (:obj:`~dataeval.protocols.VideoStream`) such as a
    multi-object tracking dataset yields, so ``Embeddings(tracking_dataset,
    extractor=VideoTorchExtractor(...))`` embeds each sequence. Frames past the last
    complete clip are left out.

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
        Device for computation. When None, uses the device set with
        :func:`~dataeval.config.set_device`, otherwise PyTorch's default device, which is the CPU
        unless you changed it. The model is moved to this device, and each clip is copied to it,
        so clips on a GPU are copied back to the host unless ``device`` names that GPU. A warning
        says so the first time it happens.
    layer_name : str or None, default None
        Layer to extract embeddings from. When None, uses model output.
    use_output : bool, default True
        If True, captures layer output; if False, captures layer input, as the layer
        receives it, before an in-place layer such as ``ReLU(inplace=True)`` can overwrite
        it. Only used when layer_name is specified. The layer is hooked only while the
        model runs, so the extractor leaves nothing registered on the model.
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
        self._warned_host_copy = False

        # Validate pooling strategy
        if pooling not in _POOLING:
            raise ValueError(f"Invalid pooling '{pooling}'. Must be one of {set(_POOLING)}")

        # Validate clip aggregation
        if clip_aggregation not in _CLIP_AGGREGATION:
            raise ValueError(f"Invalid clip_aggregation '{clip_aggregation}'. Must be one of {set(_CLIP_AGGREGATION)}")

        # Setup model
        self._model = model.to(self.device).eval()
        self._num_frames = _resolve_num_frames(model, num_frames)

        # The layer to capture, validated now and hooked only for the length of each forward pass
        self._layers: dict[str, torch.nn.Module] = {}
        if layer_name is not None:
            self._layers[layer_name] = get_valid_layer(layer_name, model)
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

    def _split_video_into_clips(self, video: Any) -> Iterator[Any]:
        """
        Yield a video's complete, non-overlapping clips, one at a time.

        Parameters
        ----------
        video : Any
            Video as a numpy array of shape (num_frames, height, width, channels), a
            list of frames, or a MAITE video stream.

        Yields
        ------
        Any
            One clip of ``num_frames`` frames: a list of frames, or for an array of
            any rank but 4 a slice of it. A trailing incomplete clip is skipped.
        """
        n = self._num_frames
        if _is_sliced(video):
            # Sliced as is: the layout an array of any other rank has always reached the processor in.
            chunks: Iterator[Any] = (video[start : start + n] for start in range(0, len(video), n))
        else:
            # One pass, n frames at a time: a stream is never measured, indexed or held whole.
            frames = iter(video)
            chunks = iter(lambda: [_frame_of(frame) for frame in islice(frames, n)], [])

        for clip_frames in chunks:
            if len(clip_frames) == n:
                yield clip_frames
            else:
                _logger.debug(f"Skipping incomplete clip with {len(clip_frames)}/{n} frames")

    def _aggregate_clips_incremental(self, clips: Iterator[Any], video_idx: int) -> torch.Tensor:
        """
        Process and aggregate clips incrementally to minimize memory usage.

        Only the clip in hand and the running aggregate are held, so a stream is never
        read further ahead than one clip. This does slow things down, and could be sped
        up a bit by iterating through batches rather than individual clips.

        Parameters
        ----------
        clips : Iterator
            The video's clips, each a sequence of frames.
        video_idx : int
            Position of the video in the batch, for messages.

        Returns
        -------
        torch.Tensor
            Aggregated video embedding.

        Raises
        ------
        ValueError
            When `clips` is empty: the video is too short to fill even one clip.
        """
        running = self._start()
        for clip_frames in clips:
            self._push(running, clip_frames)
        return self._finish(running, video_idx)

    def _start(self) -> "_Running":
        """Begin one video's running aggregate, with no clip in it yet."""
        return _Running()

    def _push(self, running: "_Running", clip_frames: Any) -> None:
        """Embed one clip and fold it into `running` with the `clip_aggregation` combine."""
        embedding = self._embed_clip(clip_frames)
        running.n_clips += 1
        if running.aggregate is None:
            # Cloned so the accumulator owns its memory: the pooled embedding can be a view of
            # the model's output, which the in-place ``mean`` combine would otherwise write to.
            running.aggregate = embedding.clone()
            return
        running.aggregate = _CLIP_AGGREGATION[self._clip_aggregation](running.aggregate, embedding)
        self._relieve_memory(running.n_clips)

    def _finish(self, running: "_Running", video_idx: int) -> torch.Tensor:
        """Return the video's embedding from `running`.

        Raises
        ------
        ValueError
            When no clip was pushed: the video is too short to fill even one clip.
        """
        if running.aggregate is None:
            raise ValueError(
                f"Video at index {video_idx} is shorter than one clip of {self._num_frames} "
                "frames, so it yields no embedding, and the result is read by position. "
                "Drop the video from the dataset, or lower num_frames.",
            )

        if running.n_clips > _LONG_VIDEO_CLIPS:
            _logger.info(f"Processed long video {video_idx + 1} with {running.n_clips} clips")

        # Finalize aggregation
        if self._clip_aggregation == "mean":
            running.aggregate.div_(running.n_clips)  # In-place division

        return running.aggregate

    def _embed_clip(self, clip_frames: list) -> torch.Tensor:
        """Preprocess one clip and run it through the model."""
        return self._extract_clip_embedding(self._preprocess_clip(clip_frames))

    def _relieve_memory(self, processed: int) -> None:
        """Release cached CUDA blocks now and then while working through a very long video."""
        if processed <= _CACHE_RELEASE_CLIPS or processed % _CACHE_RELEASE_EVERY:
            return
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        _logger.debug(f"Processed {processed} clips")

    def _forward(self, clip_tensor: torch.Tensor) -> torch.Tensor:
        """Run one clip through the model and return the tensor to pool.

        With a hooked layer the value comes from the hook rather than from the return
        value. Each clip captures afresh, so a forward pass that never reaches the layer
        raises rather than handing back the previous clip's activations as this one's.
        """
        if self._layers:
            captured = _capture(
                self._model, clip_tensor, self._layers, use_output=self._use_output, convert=_layer_value
            )
            return next(iter(captured.values()))

        model_output = self._model(clip_tensor)
        output = _tensor_of(model_output)
        if output is None:
            raise ValueError(f"Unsupported model output type: {type(model_output)}")
        return output

    def _warn_if_pulled_to_cpu(self, clip: torch.Tensor) -> None:
        """Warn once per extractor that a clip on an accelerator is copied to a CPU model, as the default does."""
        if not self._warned_host_copy:
            self._warned_host_copy = warn_if_pulled_to_cpu("VideoTorchExtractor", clip, self.device)

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
        clip_tensor = clip_tensor.unsqueeze(0)  # Add batch dim
        self._warn_if_pulled_to_cpu(clip_tensor)
        clip_tensor = clip_tensor.to(self.device)

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
            Iterable of videos to extract features from. Each video is one of:

            - a numpy array of shape (num_frames, height, width, channels);
            - a list of frames in a format the processor accepts, such as
              (height, width, channels) arrays or PIL Images;
            - a MAITE video stream (:obj:`~dataeval.protocols.VideoStream`), as a
              multi-object tracking dataset holds in its image slot: any iterable
              of frames with a ``pixels`` attribute in (channels, height, width)
              order. Each frame's pixels reach the processor as a (height, width,
              channels) array, a frame with no channel axis as one channel. Each
              call walks the stream once, in order, one clip at a time: it is never
              asked for its length or indexed, and no more than one clip of it is
              held in memory.

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
            clips = self._split_video_into_clips(video)
            video_embedding = self._aggregate_clips_incremental(clips, video_idx)
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


def _is_mirrored(extractor: object) -> bool:
    """Whether `extractor` can share a walk of each video: a `VideoTorchExtractor` whose call is its own."""
    return type(extractor).__call__ is VideoTorchExtractor.__call__


def _note(error: Exception, key: str, dataset_index: int) -> None:
    error.add_note(f"Raised by the extractor {key!r} for the video at dataset index {dataset_index}.")


def _push_noted(extractor: VideoTorchExtractor, running: _Running, clip: Any, key: str, dataset_index: int) -> None:
    try:
        extractor._push(running, clip)
    except Exception as error:
        _note(error, key, dataset_index)
        raise


def _own_copy(item: Any) -> Any:
    """Return a copy of a NumPy array or a tensor, and anything else as it is."""
    if isinstance(item, np.ndarray):
        return item.copy()
    if isinstance(item, torch.Tensor):
        return item.clone()
    return item


def _clip_for(extractor: VideoTorchExtractor, window: list[Any]) -> list[Any]:
    """Return the frames one extractor's clip is made from, copied where it could write into frames others share.

    Without a processor the frames are stacked into a new array before anything else sees them. A processor is given
    the frames themselves, and the other extractors' windows hold the same ones.
    """
    return window if extractor._processor is None else [_own_copy(frame) for frame in window]


def _feed(
    extractors: Mapping[str, VideoTorchExtractor],
    windows: dict[str, list[Any]],
    running: Mapping[str, _Running],
    frame: Any,
    dataset_index: int,
) -> None:
    """Add one frame to each extractor's clip window, and push each window that fills."""
    for key, extractor in extractors.items():
        windows[key].append(frame)
        if len(windows[key]) == extractor.num_frames:
            _push_noted(extractor, running[key], _clip_for(extractor, windows[key]), key, dataset_index)
            windows[key] = []


def _walk_once(
    extractors: Mapping[str, VideoTorchExtractor], video: Any, running: Mapping[str, _Running], dataset_index: int
) -> None:
    """Walk `video` once, converting each frame once and feeding it to every extractor's own clip window."""
    windows: dict[str, list[Any]] = {key: [] for key in extractors}
    for frame in video:
        _feed(extractors, windows, running, _frame_of(frame), dataset_index)
    for key, window in windows.items():
        if window:
            _logger.debug(f"Skipping incomplete clip with {len(window)}/{extractors[key].num_frames} frames")


def _push_sliced(
    extractors: Mapping[str, VideoTorchExtractor], video: Any, running: Mapping[str, _Running], dataset_index: int
) -> None:
    """Push the clips of a video that is sliced, not walked: it is in memory, so each extractor slices it alone."""
    for key, extractor in extractors.items():
        _push_slices(extractor, video, running[key], key, dataset_index)


def _push_slices(extractor: VideoTorchExtractor, video: Any, running: _Running, key: str, dataset_index: int) -> None:
    """Push one extractor's clips of a sliced video, cut from its own copy of a NumPy array or a tensor.

    The copy keeps a transform that writes in place from changing the video for the next extractor: a clip sliced from
    an array reaches the model without a copy of its own. It lives only while this extractor runs. An array of
    another type is sliced as it is, as `Embeddings.compute_multi` passes it.
    """
    for clip in extractor._split_video_into_clips(_own_copy(video)):
        _push_noted(extractor, running, clip, key, dataset_index)


def _embed_video(
    extractors: Mapping[str, VideoTorchExtractor], video: Any, position: int, dataset_index: int
) -> dict[str, torch.Tensor]:
    """Embed one video with every extractor, walking it at most once."""
    running = {key: extractor._start() for key, extractor in extractors.items()}
    if _is_sliced(video):
        _push_sliced(extractors, video, running, dataset_index)
    else:
        _walk_once(extractors, video, running, dataset_index)
    finished: dict[str, torch.Tensor] = {}
    for key, extractor in extractors.items():
        try:
            finished[key] = extractor._finish(running[key], position).cpu()
        except ValueError as error:
            _note(error, key, dataset_index)
            raise
    return finished


def _mirror(
    extractors: Mapping[str, VideoTorchExtractor], videos: Sequence[Any], indices: Sequence[int]
) -> dict[str, NDArray[np.float32]]:
    """Embed `videos` with every extractor, walking each video once, and return one row per video for each key.

    A video that is sliced into clips rather than walked, an array, is in memory already, so each extractor slices it as
    it does alone. Each result equals the extractor's own call on `videos`.
    """
    rows: dict[str, list[torch.Tensor]] = {key: [] for key in extractors}
    for position, video in enumerate(videos):
        for key, row in _embed_video(extractors, video, position, indices[position]).items():
            rows[key].append(row)
    return {key: torch.stack(r).numpy() if r else np.empty((0,), dtype=np.float32) for key, r in rows.items()}
