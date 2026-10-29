# Recommended extrinsic metadata factors

DataEval reads datasets through the {term}`MAITE<Modular AI Trustworthy Engineering (MAITE)>` interface, which means
every frame has already been decoded by the time DataEval sees it. Anything that lives in the video container rather
than in the pixels (the codec, the group-of-pictures structure, the position and orientation of the sensor, the
wall-clock time of capture) is out of reach unless you extract it yourself. For multi-object tracking,
MAITE's `VideoFrame` does carry each frame's presentation time (`time_s` and `pts`), but not the wall-clock time of
capture or the precision timestamps described below.

That information is worth extracting. Much of it describes the collection and processing pipeline, and when a pipeline
treats some videos differently from others, a model can learn those differences instead of the intended task. This page
recommends which factors to extract, gives the known factor names that DataEval checks, and shows where in a video file
each factor can be found. To serve the factors from video files you read, see
[How to wrap video files as a multi-object tracking dataset](../notebooks/h2_wrap_video_files.py).

## Why extrinsic factors matter

An **extrinsic** factor describes how the data was collected and processed, not what it depicts. Extrinsic factors are
easy to overlook because they have no obvious connection to the task, but they are what a model can use as a
shortcut.

Suppose an older camera records most of the videos of class A, while class B is recorded by a newer camera. The two
cameras leave different compression artifacts, so a model can separate the classes by the artifacts and never learn the
objects. Nothing in the labels or the pixel statistics will say so. A factor that records the camera, or the codec
settings, will.

The factors recommended here cover the collection and processing pipeline in three categories (compression, platform
and time) and exclude information that is redundant or that nobody could act on.

## Recommended factors

Below is a summary of the extrinsic factors recommended for `Metadata`, their levels, and their types. `timestamp`,
`video_duration`, `frame_rate`, `latitude` and `longitude` are [known factor names](#known-factor-names).

| Category    | Factor                                            | Level                              | Type                  |
| ----------- | ------------------------------------------------- | ---------------------------------- | --------------------- |
| Compression | Codec name                                        | sequence                           | categorical           |
| Compression | Interlacing                                       | sequence                           | categorical           |
| Compression | Chroma subsampling and bit depth                  | sequence                           | categorical           |
| Compression | GOP length                                        | sequence                           | numeric               |
| Compression | Variable GOP                                      | sequence                           | categorical           |
| Compression | Frame independence (`I`, `P`, `B`)                | unit                               | categorical           |
| Compression | Level of compression (bits per frame)             | unit                               | numeric               |
| Platform    | Camera type, platform type                        | sequence                           | categorical           |
| Platform    | Weather                                           | sequence (static) or unit (moving) | numeric / categorical |
| Platform    | Location: `latitude`, `longitude`, country, biome | sequence (static) or unit (moving) | numeric / categorical |
| Platform    | Position and motion                               | sequence (static) or unit (moving) | numeric               |
| Time        | `timestamp`                                       | unit                               | numeric               |
| Time        | `video_duration`, `frame_rate`                    | sequence                           | numeric               |

The three categories are described in detail below. Each code snippet shows where the information can be found, not a
complete pipeline; adapt it to your data source.

## Compression

Compression introduces visual artifacts that a model can learn as shortcuts. DataEval's
[image statistics](ImageStatistics.md) measure some of their effects on the frames themselves, such as `sharpness` and
`contrast`, but computing them over a large dataset can be expensive. The factors below cover the common sources of
artifacts and are cheap to extract, so they work as a first pass for deciding whether the image statistics are worth
computing.

| Factor                                  | Description                                          | Level    |
| --------------------------------------- | ---------------------------------------------------- | -------- |
| Codec name                              | Name of the main video codec (for example H.264)     | sequence |
| Interlacing                             | Whether the video is interlaced                      | sequence |
| Chroma subsampling and bit depth        | Pixel format, such as `yuv420p` or `yuv422p10le`     | sequence |
| GOP length                              | Average length of each group of pictures             | sequence |
| Variable GOP                            | Whether the GOP length varies                        | sequence |
| Frame independence                      | Frame type: `I`, `P` or `B`                          | unit     |
| Level of compression                    | Compressed size of each frame, in bits               | unit     |

All of these can be read with `ffprobe`. The helper below is used by the snippets on this page:

```python
import json, subprocess


def ffprobe(path, entries, kind, stream=None):
    """Return ffprobe's `kind` list ("streams", "frames" or "packets") for the requested entries."""
    cmd = ["ffprobe", "-v", "error", "-show_entries", entries, "-of", "json", path]
    if stream:
        cmd[3:3] = ["-select_streams", stream]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return json.loads(out)[kind]
```

Codec, interlacing, chroma subsampling and bit depth are stream properties:

```python
s = ffprobe(video, "stream=codec_name,pix_fmt,field_order", "streams", "v:0")[0]
codec = s["codec_name"]  # codec name
pix_fmt = s["pix_fmt"]  # chroma subsampling and bit depth
interlaced = s.get("field_order", "unknown") not in ("progressive", "unknown")  # interlacing
```

Frame type, compressed size, and the GOP statistics derived from them are frame properties:

```python
frames = ffprobe(video, "frame=pict_type,pkt_size", "frames", "v:0")
frame_type = [f["pict_type"] for f in frames]  # frame independence
bits_per_frame = [8 * int(f["pkt_size"]) for f in frames]  # level of compression

i_frames = [i for i, t in enumerate(frame_type) if t == "I"]
gaps = [b - a for a, b in zip(i_frames, i_frames[1:])]
avg_gop = sum(gaps) / len(gaps) if gaps else None  # GOP length
variable_gop = len(set(gaps)) > 1  # variable GOP
```

## Platform

Platform factors describe the physical context of the sensor: whether it is fixed or mounted on a vehicle, and where it
is. What you can record depends on your collection systems and on your problem, and recording the same information at
several granularities (coordinates, city, country, biome) can expose patterns that a single granularity hides. Latitude
and longitude are the preferred location factors, because coarser descriptions such as country and city can be derived
from them. For aerial platforms, altitude is an especially useful factor, along with weather descriptors such as cloud
cover.

Many of these factors remain constant throughout the recording of a video when the platform is static, so they are
stored at the `sequence` level for video (and the `unit` level for images). When the platform moves, it is recommended
to record them for each frame, at the `unit` level for both video and images.

Suggested factors:

- camera type
- platform type (aerial, ground, static, worn)
- weather, or sensor-based proxies such as air temperature, pressure, wind speed and humidity
- location as `latitude` and `longitude` (preferred), country, or biome (ocean, forest, city, and so on)
- position and motion: ground speed or air speed, pitch and roll angles, altitude

### MISB ST 0601

MISB ST 0601 is the Motion Imagery Standards Board's standard for geospatial and platform metadata on unmanned systems.
Some of its items are redundant for bias detection, but it is a useful reference when designing a collection pipeline.
The following items are recommended when extracting from an ST 0601-compliant video:

| Factor                                         | Item      | Units | KLV format             |
| ---------------------------------------------- | --------- | ----- | ---------------------- |
| Platform True Airspeed                         | 8         | m/s   | uint8                  |
| Platform Designation                           | 10        | none  | utf8                   |
| Image Source Sensor                            | 11        | none  | utf8                   |
| Latitude (Frame Center / Sensor)               | 23 / 13   | °     | int32                  |
| Longitude (Frame Center / Sensor)              | 24 / 14   | °     | int32                  |
| Wind Speed                                     | 36        | m/s   | uint8                  |
| Static Pressure                                | 37        | mbar  | uint16                 |
| Outside Air Temperature                        | 39        | °C    | int8                   |
| Relative Humidity                              | 55        | %     | uint8                  |
| Platform Ground Speed                          | 56        | m/s   | uint8                  |
| UAS Datalink LS Version Number                 | 65        | none  | uint8                  |
| Platform Pitch Angle (Full / range limited)    | 90 / 6    | °     | int32 / int16          |
| Platform Roll Angle (Full / range limited)     | 91 / 7    | °     | int32 / int16          |
| Altitude (AGL / MSL)                           | 113 / 15  | m     | IMAPB / uint16         |
| Zoom Percentage                                | 134       | %     | IMAPB, variable length |
| Leap Seconds                                   | 136       | s     | int, up to 4 bytes     |
| Correction Offset                              | 137       | µs    | int, up to 8 bytes     |

The known factors `latitude` and `longitude` are the sensor's position (items 13 and 14), not the frame center's.

ST 0601 data travels in its own stream inside the container, encoded as KLV (key-length-value). Each packet starts with
a 16-byte universal key and a BER length, and its value is a sequence of items, each an item number, a BER length and
the item's bytes. The helpers below split a packet's value, `packet`, into `{item: raw bytes}`; the time section shows
how to cut the stream into packets. Decoding an item then depends on its format in the table above; sensor latitude
(item 13) is shown as an example:

```python
def ber(buf, pos):
    """Read a BER length; return (value, next position)."""
    n = buf[pos]
    if n < 0x80:
        return n, pos + 1
    k = n & 0x7F
    return int.from_bytes(buf[pos + 1 : pos + 1 + k], "big"), pos + 1 + k


def items(packet):
    """Yield (item number, raw bytes) for each item in the value of a 0601 packet."""
    pos = 0
    while pos < len(packet):
        item = 0
        while True:
            b = packet[pos]
            pos += 1
            item = (item << 7) | (b & 0x7F)
            if not b & 0x80:
                break
        n, pos = ber(packet, pos)
        yield item, packet[pos : pos + n]
        pos += n


fields = dict(items(packet))
v = fields.get(13)  # item 13: sensor latitude

# latitude-specific value processing
lat = None if v in (None, b"\x80\x00\x00\x00") else int.from_bytes(v, "big", signed=True) * 180 / 4294967294
```

:::{important}
In a MISB-compliant system an item is sent only when its value changes, so not every item appears in every packet. To
build a per-frame table, carry the last known value forward.
:::

## Time

Time is stored at the `unit` level. How precisely it needs to be known depends on the medium. For images, a coarse label
such as day, night or dawn/dusk is often enough. For video, a timestamp for each frame is recommended. Without
timestamps, provide the video's total duration or average frame rate as `sequence`-level factors.

Sub-second precision rarely matters when evaluating a dataset for bias, but it matters a great deal for frame
alignment. Metadata streams are stored per packet, not per frame, and the two do not correspond one to one. If frames
and metadata are misaligned, results from metadata analysis can be misleading.

### Timestamp types

Which timestamps exist depends on the data source. The examples on this page assume a MISB-compliant video, which offers
two kinds.

**Presentation timestamps (PTS)** are the container's clock. Every stream in a container has comparable PTS, which makes
them useful for aligning video to metadata. They are not immutable, however: transcoding or remuxing can change them and
break the alignment. MPEG-TS streams also commonly start at a non-zero offset.

**Precision timestamps** are an optional element that MISB recommends. They are typically supplied by a GPS receiver and
attached to each stream as it is recorded. They are designed to be immutable, and any manual correction is stored in a
separate correction offset (item 137, above).

### Aligning metadata to frames

Match the precision timestamp of each frame in the video bitstream to the precision timestamp of each KLV packet in the
data stream. If either stream lacks precision timestamps, fall back to matching on PTS, which any standard video
collection process provides.

The data stream is usually, but not always, stream `0:d:0`. Check its position before extracting from it.

In H.264, precision timestamps in the video stream are carried in user data unregistered SEI messages that
begin with the 16-byte identifier `MISPmicrosectime`. MPEG-2 carries the same identifier in picture user data instead.
The snippet below handles H.264. Decode the 12 bytes that follow each identifier according to ST 0604:

```python
import re

raw = subprocess.run(
    [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        video,
        "-map",
        "0:v:0",
        "-c",
        "copy",
        "-bsf:v",
        "h264_mp4toannexb",
        "-f",
        "h264",
        "-",
    ],
    capture_output=True,
    check=True,
).stdout
ident = b"MISPmicrosectime"
sei = [raw[m.end() : m.end() + 12] for m in re.finditer(re.escape(ident), raw)]
```

In the data stream, each ST 0601 packet carries its precision timestamp as item 2, a 64-bit count of
microseconds. Split the stream into KLV packets by the 16-byte key and BER length, and read item 2 from each packet's
value with the `items()` helper from the platform section:

```python
raw = subprocess.run(
    ["ffmpeg", "-v", "error", "-i", video, "-map", "0:d:0", "-c", "copy", "-f", "data", "-"],
    capture_output=True,
    check=True,
).stdout
key = bytes.fromhex("060e2b34020b01010e01030101000000")  # ST 0601 universal key
timestamps_us, pos = [], 0
while pos + 17 <= len(raw):
    length, start = ber(raw, pos + 16)
    if raw[pos : pos + 16] == key:
        timestamps_us.append(int.from_bytes(dict(items(raw[start : start + length]))[2], "big"))
    pos = start + length
```

Both streams report PTS on the container clock, in seconds:

```python
frame_pts = [
    float(f["best_effort_timestamp_time"]) for f in ffprobe(video, "frame=best_effort_timestamp_time", "frames", "v:0")
]
klv_pts = [float(p["pts_time"]) for p in ffprobe(video, "packet=pts_time", "packets", "d:0")]
```

:::{important}
Modes of `ffmpeg` and `ffprobe` that do not decode the video (`-c copy`, `-show_packets`) return results in decode
order, which differs from display order. The SEI timestamps above are in decode order too. To line them up with
ordinary frame numbering, sort by PTS. This assumes one SEI timestamp per video packet, and `strict=True` raises if the
counts differ:

```python
vpkts = ffprobe(video, "packet=pts_time", "packets", "v:0")  # in decode order
in_display_order = sorted(zip((float(p["pts_time"]) for p in vpkts), sei, strict=True))
```

:::

## Known factor names

A few of the recommended factors have a name that DataEval gives one fixed meaning, listed in
{data}`dataeval.types.KNOWN_FACTORS`. Names are lowercase and matched against the final column name character for
character, so `TIMESTAMP` or `gps_latitude` is an ordinary factor. Analytics will build on these names, so providing
them now means your metadata is ready for them.

| Name             | Meaning                             | Unit                          | Levels             | Valid values          |
| ---------------- | ----------------------------------- | ----------------------------- | ------------------ | --------------------- |
| `timestamp`      | Wall-clock capture time of the unit | seconds since Unix epoch, UTC | `unit`             | any finite number     |
| `frame_rate`     | Average frame rate of the video     | frames per second             | `sequence`         | greater than 0        |
| `video_duration` | Total duration of the video         | seconds                       | `sequence`         | 0 or greater          |
| `latitude`       | Latitude of the sensor (WGS 84)     | decimal degrees               | `sequence`, `unit` | from -90 to 90        |
| `longitude`      | Longitude of the sensor (WGS 84)    | decimal degrees               | `sequence`, `unit` | from -180 to 180      |

DataEval checks the type, range and level of a known factor. Every value must be a real number (integers count,
booleans and text do not) within the range above, and the factor must be at one of the listed levels. Missing values
(`None` or NaN) are not checked; infinity is invalid.

When a column of the dataset's own metadata fails a check, it is held back: it is not a factor, and
{attr}`.Metadata.unusable` and {attr}`.Metadata.dropped_factors` list it with the reason `invalid_known_factor`. Fix it
with {meth}`.Metadata.repair`. A column that mixes text and numbers is reported as `mixed_types`, as for any factor.
A known factor at a level it is not defined at, such as a `frame_rate` given per frame, cannot be repaired, since a
repair changes values and never their level. Supply it at a listed level instead: fix it at the source, or pass its
values to {meth}`.Metadata.add_factors` at that level, such as one `frame_rate` per video with `level="sequence"`.

{meth}`.Metadata.add_factors`, {meth}`.Metadata.from_factors`, {meth}`.Metadata.agg` and {meth}`.Metadata.aggregate`
raise a `ValueError` instead.

Using a known name is a promise that the data follows the definition above. The checks cannot verify units: a
`timestamp` in milliseconds since the stream started passes them. If a factor means something else, give it a
different name.

A dataset's reported `video_duration` and `frame_rate` can be checked against the frames themselves.
{func}`.video_stats` measures both from frame timestamps, and
`md.add_factors(video_stats(ds), return_collisions=True, collision_tolerance={"video_duration": 0.01})` attaches the
measured values and returns a report of where they disagree with the reported ones. With the default
`overwrite=False`, a measured factor whose name the dataset already reports is stored under a new name, such as
`video_duration_added`. The report holds a {class}`.FactorCollision` for each factor that disagrees on at least one
video. Here `video_duration` differs when the two values are more than 1% apart, and a reported `frame_rate`, which the
tolerance does not name, differs on any difference. Each entry's `differing` frame lists those videos by `item_index`,
with the `existing` value, the `incoming` one and their `relative_difference`. A video without a reported value, or one
that could not be measured, is not compared.

A `timestamp` column of ISO 8601 text or of datetimes is held back, and `repair([ParseDateTime("timestamp")])` reads it
as seconds since the Unix epoch, which passes. A period such as `every="hour"` is not a capture time, and its text labels
keep the column held back. Give a column you mean to read as periods a different name, such as `capture_time`. With
{meth}`.Metadata.add_factors` or {meth}`.Metadata.from_factors`, convert datetimes to epoch seconds first, or use
another name to keep them as datetimes. For a NumPy array, `(stamps - np.datetime64("1970-01-01")) / np.timedelta64(1, "s")`
turns a missing `NaT` into NaN, a missing value. Casting with `astype("int64")` instead turns it into a large negative
number that passes as a real time.

## Related concept pages

- [Metadata levels](MetadataLevels.md) — the sequence, unit, track and instance levels, and why a factor is stored once
at its own level.
- [Binning](Binning.md) — how continuous factors such as altitude and timestamps are discretized, and how to declare
cut points that come from the world.
- [Image statistics](ImageStatistics.md) — factors DataEval can compute from the decoded frames and add to `Metadata` on
demand, as a complement to the factors on this page.
- [Acting on results](ActingOnResults.md) — how to diagnose and respond to a correlation between a metadata factor and
the class label.
