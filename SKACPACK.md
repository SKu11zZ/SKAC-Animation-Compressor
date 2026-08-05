# SKAC Pack v1

[English](#english) | [中文](#chinese)

<a id="english"></a>

## English

SKAC Pack v1 is the public animation-library container. It stores validated `.skac`
animations behind stable entry IDs, keeps a content-addressed blob table, and stores an
identical animation only once. Pack creation is deterministic: the same named inputs
produce the same bytes regardless of command-line order.

The first implementation performs exact whole-animation deduplication. Shared track and
segment dictionaries are the next compatible Pack milestone; the current implementation
does not claim partial-motion sharing yet.

### Prefix

Multi-byte values are little-endian. The prefix uses `8s HH II Q II`:

| Field | Type | Meaning |
| --- | --- | --- |
| magic | 8 bytes | ASCII `SKACPACK` |
| major | uint16 | incompatible version; currently `1` |
| minor | uint16 | compatible version; currently `0` |
| flags | uint32 | currently `0` |
| metadata bytes | uint32 | canonical UTF-8 JSON length |
| payload bytes | uint64 | concatenated unique blob length |
| metadata CRC | uint32 | CRC-32 of metadata bytes |
| payload CRC | uint32 | CRC-32 of payload bytes |

Metadata schema `skac.pack` version `1.0.0` contains two sorted tables:

- `entries`: public ID, blob SHA-256, timing, and skeleton signature;
- `blobs`: SHA-256, byte offset, and byte length for each unique `.skac` file.

Every read validates prefix lengths, both CRCs, canonical tables, contiguous blob
ranges, SHA-256 values, embedded `.skac` containers, and entry metadata. Entry IDs are
restricted to 1-128 ASCII letters, digits, dots, underscores, and hyphens. They are
identifiers, not paths.

### CLI

```text
skac pack-create --clip idle=idle.skac --clip walk=walk.skac -o library.skacpack
skac pack-inspect library.skacpack
skac pack-extract library.skacpack walk -o restored.skac
```

Agent protocol v1 exposes the same operations as `pack_create`, `pack_inspect`, and
`pack_extract`. Agent paths remain relative to its workspace.

<a id="chinese"></a>

## 中文

SKAC Pack v1 是公开的动画库容器。它用稳定的动画 ID 管理多个 `.skac`，以
SHA-256 建立内容寻址表，并自动合并完全相同的动画。相同输入无论命令顺序如何，都会生成
完全一致的文件。

当前版本先完成整条动画的精确去重，还没有声称能够共享动画内部的局部片段。下一阶段会在
保持兼容的前提下加入共享 Track 与 Segment 字典。读取时会检查长度、CRC、SHA-256、
连续数据范围、内嵌 `.skac` 和索引元数据；任何损坏或不一致都会被拒绝。
