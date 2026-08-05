# Native Runtime Beta / 原生运行时 Beta

[English](#english) · [中文](#chinese)

<a id="english"></a>

## English

The Runtime Beta plays a complete `.skac` file without starting Python. It consists of
a C++17 decoder with a stable C ABI, native frozen-Profile playback, a Unity C# package,
and an Unreal runtime plugin. No compiled binaries are stored in this repository.

### What works now

- parse and validate SKAC v1 and independently checksummed SKAC v2 chunks;
- inflate through built-in zlib or a host callback;
- decode the complete clip into immutable track data;
- query frame rate, duration, joint names, and parents;
- sample local transforms by frame or time with clamp/loop behavior;
- compile Profile 2.0 once and sample directly into a different target skeleton;
- share one immutable decoder across multiple per-character playback instances;
- reuse caller-owned pose buffers during playback;
- use the same public decoder from Unity and Unreal.

The C ABI returns quaternion `xyzw` plus translation `xyz`. Decoder instances are
read-only after opening. Each retargeter owns reusable scratch buffers and represents
one playback instance; separate retargeters may share a decoder and run on separate
threads. Diagnostics are thread-local.

### Beta limits

- v1 inflates as one stream; v2 inflates per chunk, but both currently retain the fully
  reconstructed clip in memory rather than a moving streaming window;
- source coordinate convention and units are preserved;
- Unity and Unreal adapters do not yet drive a character automatically;
- the target skeleton is an explicit generated JSON companion, not embedded in `.skac`;
- contact correction is intentionally outside the real-time Profile path;
- the public repository provides source adapters, not prebuilt engine binaries.

Profile building remains an offline Python step. Runtime playback only consumes its
frozen mapping, evaluation order, basis quaternions, and root scale.

### Prepare different-character playback

```text
python -m skac_codec profile motion.skac target.bvh -o target.skac-profile.json
python -m skac_codec runtime-skeleton target.bvh -o target.runtime-skeleton.json
```

Open the `.skac` decoder once, then call `skac_retargeter_create` with those two JSON
documents. Create one retargeter per playing character, reuse its output buffer, and
keep the shared decoder alive until every retargeter is closed.

### Build the native library

```text
cmake -S native -B build/native \
  -DSKAC_RUNTIME_BUILD_SHARED=ON \
  -DSKAC_RUNTIME_WITH_ZLIB=ON
cmake --build build/native --config Release
ctest --test-dir build/native -C Release --output-on-failure
```

For Unreal, the adapter compiles the native source into the module and uses Unreal's
zlib implementation through the host callback. For Unity, build a shared library with
zlib and place the result in the project's platform-specific native plugin directory.

### Verification

`native/tests/runtime_smoke.cpp` exercises the ABI and sampling path. The second check
compares a compiled native probe against the Python format-1.0 decoder, transform by
transform:

```text
python -m tools.verify_native_runtime --probe build/native/skac_runtime_probe
python -m tools.verify_native_v2_runtime --probe build/native/skac_runtime_v2_probe
```

The comparison uses a generated public test motion and writes temporary metadata and
payload files outside the release tree.

The Release benchmark measures time-interpolated sampling after warmup:

```text
python -m tools.run_native_runtime_benchmark \
  --benchmark build/native/skac_runtime_benchmark \
  --output reports/native_runtime_profile_beta.json \
  --visual reports/native_runtime_profile_beta.svg
```

The committed report uses a deterministic 300-frame fixture with 65 source and 67
target joints. It is local regression evidence, not a cross-machine ranking.

[Jump to Chinese](#chinese)

---

<a id="chinese"></a>

## 中文

Runtime Beta 可以不启动 Python，直接在引擎侧播放完整 `.skac` 文件。它由 C++17 解码
核心、稳定的 C ABI、原生冻结 Profile 播放、Unity C# Package 和 Unreal Runtime Plugin
组成。仓库只提交源码，不提交编译后的动态库。

### 现在能做什么

- 解析并校验 SKAC v1，以及带独立校验的 SKAC v2 分块；
- 使用内置 zlib，或者把解压交给宿主引擎；
- 一次载入完整动画并生成只读轨道数据；
- 查询帧数、时长、关节名和父子关系；
- 按帧或按时间采样，支持截断与循环；
- 一次编译 Profile 2.0，随后直接采样到不同目标骨骼；
- 多个角色播放实例共享同一个只读 Decoder；
- 播放时重复使用调用方提供的姿态缓冲区；
- Unity 和 Unreal 共用同一个公开解码核心。

C ABI 输出四元数 `xyzw` 和位移 `xyz`。Decoder 打开后只读；每个 Retargeter 拥有可复用
的临时缓冲区，代表一个播放实例。不同 Retargeter 可以共享 Decoder 并在不同线程运行，
错误信息按线程保存。

### Beta 的边界

- v1 仍按整段解压，v2 会逐块解压；两者目前都会把完整重建结果保留在内存中，还不是只
  保留滑动窗口的流式播放；
- 保留源动画的坐标系和单位；
- Unity、Unreal 适配层暂不自动驱动角色；
- 目标骨骼使用显式生成的 JSON 配套文件，暂不嵌入 `.skac`；
- 接触修正刻意不进入实时 Profile 路径；
- 公开仓库提供适配源码，不提供预编译引擎二进制。

Profile Builder 仍然是离线 Python 步骤。运行时只消费冻结的映射、计算顺序、基变换
四元数和根位移缩放。

### 准备不同角色播放

```text
python -m skac_codec profile motion.skac target.bvh -o target.skac-profile.json
python -m skac_codec runtime-skeleton target.bvh -o target.runtime-skeleton.json
```

先打开一次 `.skac` Decoder，再把这两个 JSON 交给 `skac_retargeter_create`。每个正在播放
的角色建立一个 Retargeter，重复使用输出缓冲区，并在所有 Retargeter 关闭之前保留共享
Decoder。

### 编译原生库

```text
cmake -S native -B build/native \
  -DSKAC_RUNTIME_BUILD_SHARED=ON \
  -DSKAC_RUNTIME_WITH_ZLIB=ON
cmake --build build/native --config Release
ctest --test-dir build/native -C Release --output-on-failure
```

Unreal 适配层会把原生源码直接编进模块，并通过回调使用 Unreal 自带的 zlib。Unity 需要
先编译带 zlib 的动态库，再把它放到项目对应平台的原生插件目录。

### 怎么验证

`native/tests/runtime_smoke.cpp` 覆盖 ABI 和采样入口。第二项检查会把原生 Probe 与 Python
格式 1.0 解码器逐个变换进行比较：

```text
python -m tools.verify_native_runtime --probe build/native/skac_runtime_probe
python -m tools.verify_native_v2_runtime --probe build/native/skac_runtime_v2_probe
```

比较使用程序生成的公开测试动画，临时元数据和 Payload 会写在发布目录之外。

Release 基准会在预热后测量带时间插值的采样：

```text
python -m tools.run_native_runtime_benchmark \
  --benchmark build/native/skac_runtime_benchmark \
  --output reports/native_runtime_profile_beta.json \
  --visual reports/native_runtime_profile_beta.svg
```

仓库内报告使用确定性生成的 300 帧夹具，源骨骼 65 个关节、目标骨骼 67 个关节。它只作为
本机回归证据，不拿来做跨机器排名。

[Jump to English](#english)
