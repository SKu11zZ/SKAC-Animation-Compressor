# Agent CLI

[English](#english) · [中文](#chinese)

<a id="english"></a>

## English

`skac-agent` is the machine-facing entry point. It uses a versioned JSON contract,
writes only JSON to standard output, returns stable exit codes, and never asks an
interactive question. It currently exposes the tested BVH workflow; the experimental
FBX bridge is intentionally outside protocol v1.

Install the package, then discover the contract:

```text
skac-agent capabilities --pretty
```

Without installing the console script, use
`python -m skac_codec.agent capabilities --pretty` instead.

Every asset path in a request must be relative to `--workspace`. Absolute paths and
paths that leave the workspace are rejected. Existing outputs are preserved unless
the request explicitly sets `"overwrite": true`.

Single request:

```text
skac-agent run --workspace ./job --request ./encode-request.json --pretty
```

`encode-request.json`:

```json
{
  "protocol": "skac.agent.v1",
  "request_id": "encode-001",
  "operation": "encode",
  "arguments": {
    "input": "motions/walk.bvh",
    "output": "artifacts/walk.skac",
    "quality": "high",
    "format_version": 2
  }
}
```

The response has one envelope for every operation:

```json
{
  "ok": true,
  "operation": "encode",
  "protocol": "skac.agent.v1",
  "request_id": "encode-001",
  "result": {
    "artifact": {
      "bytes": 12345,
      "media_type": "application/vnd.skac.animation",
      "path": "artifacts/walk.skac"
    }
  }
}
```

The real response also includes Codec settings, compression rate, and reconstruction
errors. An error response sets `"ok": false` and contains a stable `error.code` plus a
human-readable message. Exit code `0` means success, `2` means invalid input, and `3`
means that a valid operation could not be completed.

Use `--request -` (the default) for stdin. Add `--jsonl` to process one request per
line; one bad line does not stop later jobs. JSONL responses are also one line each.

Protocol v1 operations:

- `encode`: BVH to `.skac`; `format_version` selects v1 or v2, and v2 optionally accepts
  `min_segment_frames` and `max_segment_frames`;
- `inspect`: read `.skac` metadata without creating an asset;
- `pack_create`: build a deterministic `.skacpack` from a JSON object of entry IDs and
  workspace-relative `.skac` paths;
- `pack_inspect`: fully validate and inspect a `.skacpack`;
- `pack_extract`: restore one named `.skac` entry from a `.skacpack`;
- `adaptive_plan`: build and reconstruct-check a SKAC v2 perceptual segmentation and
  bit-allocation plan, then write JSON and a self-contained SVG;
- `decode`: `.skac` to the source BVH skeleton, or to a target skeleton when both
  `target` and `profile` are supplied;
- `profile`: freeze one source-to-target skeleton profile;
- `runtime_skeleton`: export the target-skeleton JSON consumed by the native runtime;
- `quality_gate_same`: gate Codec reconstruction and direct source-character decode;
  it accepts no target or Profile fields;
- `quality_gate_different`: gate Codec decode plus frozen target-character playback;
  it requires `target` and includes Profile coverage and runtime checks.

`quality_gate` remains as a compatibility operation that classifies the case from the
two skeleton signatures. New Agent integrations should use the two explicit operations.
Every gate always writes JSON plus a self-contained SVG report.

The exact required and optional fields are returned by `capabilities`. Unknown fields
are rejected so a misspelled option cannot silently change a run.

[跳到中文](#chinese)

---

<a id="chinese"></a>

## 中文

`skac-agent` 是专门给 Agent 和自动化程序用的入口。它只在标准输出里写 JSON，
协议带版本，退出码固定，也不会执行到一半弹出交互问题。v1 先只开放已经验证过的
BVH 流程，实验性的 FBX 桥不放进稳定协议。

安装项目后，可以先让 Agent 查询能力：

```text
skac-agent capabilities --pretty
```

请求里的素材路径必须是 `--workspace` 下面的相对路径。绝对路径和越过工作目录的
路径会被拒绝。输出文件如果已经存在，默认也不会覆盖，除非请求里明确写
`"overwrite": true`。

执行单条请求：

```text
skac-agent run --workspace ./job --request ./encode-request.json --pretty
```

请求和响应格式见上面的完整示例。每条请求固定包含：

- `protocol`：目前是 `skac.agent.v1`；
- `request_id`：调用方自己的任务 ID，响应会原样带回；
- `operation`：`encode`、`inspect`、`decode`、`profile`、`runtime_skeleton`、`quality_gate_same` 或
  `quality_gate_different`；两套门槛都会同时生成 JSON 和 SVG；
- `arguments`：该操作需要的路径和选项。

`encode` 可通过 `format_version` 选择 v1 或 v2；v2 还可设置 `min_segment_frames` 和
`max_segment_frames`。`quality_gate_same` 只接受源动画，不允许传目标骨架或 Profile 字段；
`quality_gate_different` 必须传目标骨架，并会检查 Profile 覆盖和运行时。旧的
`quality_gate` 只留作兼容，会按两份骨架签名自动分类。新的 Agent 接入直接用两个明确操作。

退出码 `0` 表示成功，`2` 表示请求本身不合法，`3` 表示请求合法但执行失败。
错误响应不会往标准输出里混入 Python traceback，Agent 可以直接按 JSON 解析。

如果要批处理，给 `run` 加 `--jsonl`，输入文件每行放一个请求。某一行失败不会挡住
后面的任务，输出也保持一行一个 JSON。具体字段不要硬猜，直接读取
`capabilities` 返回值；拼错或多写字段会被明确拒绝。

[Jump to English](#english)
