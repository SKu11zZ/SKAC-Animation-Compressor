# `.skac` animation format 1.0

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
