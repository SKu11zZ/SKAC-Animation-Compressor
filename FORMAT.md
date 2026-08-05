# `.skac` animation formats

This document describes the first public SKAC animation container. Multi-byte integers
use little-endian byte order. A file contains a fixed prefix, UTF-8 JSON metadata, and
one compressed binary payload.

## Fixed prefix

The prefix uses the little-endian structure `8s HH II QQ II`:

| Field | Type | Meaning |
| --- | --- | --- |
| magic | 8 bytes | ASCII `SKACANIM` |
| major | uint16 | incompatible format version; currently `1` |
| minor | uint16 | backward-compatible format version; currently `0` |
| flags | uint32 | payload flags; bit 0 means zlib |
| metadata bytes | uint32 | exact JSON byte length |
| payload bytes | uint64 | exact compressed payload length |
| raw payload bytes | uint64 | expected decompressed payload length |
| metadata CRC | uint32 | CRC-32 of the JSON bytes |
| payload CRC | uint32 | CRC-32 of the compressed payload |

Readers reject unknown major versions, unsupported flags, mismatched lengths, trailing
bytes, invalid UTF-8, CRC failures, oversized declarations, and inconsistent skeleton
or track tables.

## Metadata

The canonical JSON object uses schema `skac.animation` version `1.0.0`. It records:

- frame count, frame time, and duration;
- joint names, parent indices, rest offsets, BVH channel declarations, and end sites;
- a SHA-256 signature of the canonical skeleton object;
- rotation and translation quantization settings;
- exact track tables, translation bounds, and payload section lengths;
- the selected quality preset and payload compressor.

Local translations are stored in the source skeleton's coordinate system. A BVH
position channel is represented as its rest offset plus the animated channel value.
The container preserves the source coordinate convention; coordinate conversion is an
input/output adapter responsibility, not an implicit codec operation.

The format does not store meshes, materials, textures, local file paths, or application
custom properties. Joint names originate from the input skeleton and should be treated
as asset data when deciding whether a generated file may be shared.

## Rotation payload

Only joints with declared rotation channels are stored. Before quantization, each joint
track is reduced independently. The encoder recursively compares source samples with
shortest-path spherical interpolation between retained keys and inserts the worst frame
until the preset's angular-error limit is satisfied.

Each track stores a uint32 key count, delta-coded unsigned variable-length frame
indices, and its packed key values. Every normalized key quaternion uses smallest-three
coding:

1. select the component with the largest absolute value;
2. flip the quaternion when that component is negative;
3. store the omitted-component index in two bits;
4. quantize the other three components over `[-1/sqrt(2), +1/sqrt(2)]`.

The decoder reconstructs the positive omitted component, normalizes the quaternion, and
spherically interpolates the omitted frames. The current low, medium, and high presets
use 10, 13, or 16 bits per stored component with key-reduction limits of 1.0, 0.25,
or 0.05 degrees respectively.

The reference decoder resolves all interpolation segments for a track in one vectorized
pass. This keeps the file format unchanged while avoiding a small NumPy call for every
keyframe interval.

## Translation payload

Only position channels declared by the source skeleton are stored. Each scalar track is
reduced against linear interpolation using a preset fraction of that track's value
range. It then stores a uint32 key count, delta-coded frame indices, and uniformly
quantized key values. Per-clip minimum and maximum bounds remain in metadata. Static
joint offsets remain in the skeleton object.

## Compression and compatibility

The concatenated bit streams are compressed with zlib and protected by CRC-32. CRC is
for accidental corruption detection, not authenticity. Applications that need trusted
distribution should sign the complete file separately.

Minor versions may add metadata fields without changing existing decoding semantics.
Changing payload interpretation, transform conventions, or required fields requires a
new major version.

## SKAC v2 Beta

SKAC v2 is the deterministic, chunked evolution of the same Codec. It does not require
a model and does not change the meaning of an authored clip. Use
`skac encode --format-version 2` to write it; the Python reader and C++17 runtime accept
both v1 and v2.

The fixed prefix keeps the same 44-byte layout with major version `2` and flags value
`2`. The payload is a concatenation of independently zlib-compressed chunks. Canonical
metadata contains an ordered directory with each chunk's id, role, required flag,
offset, compressed and raw sizes, plus CRC-32 values for both representations. The
container also retains a CRC over the complete compressed payload.

The implemented required roles are:

- `segment.index`: the authoritative contiguous frame ranges and their chunk ids;
- `base.rotation`: per-segment rotation tracks with an independent bit width per joint;
- `base.translation`: per-segment scalar translation tracks with an independent bit
  width and bounds per component.

Every rotation and translation track begins at relative frame zero. Frame indices are
delta-coded varuints. Rotation values retain smallest-three coding; translation values
retain bounded uniform quantization. Segments are decoded independently and written into
the final immutable pose buffer. The current runtime opens chunks independently but
keeps the fully reconstructed clip resident for fast sampling.

For v2, the CLI's `rotation_bits` and `translation_bits` settings are upper bounds for
the adaptive candidate set rather than one uniform width applied to every track.

Unknown required chunk roles are rejected. Unknown optional roles may be skipped.
Future refinement, semantics, contact, or generated-motion data must remain optional;
the base layer always decodes without a model. Model weights and their storage, memory,
hardware, and latency are reported separately from Codec results.
