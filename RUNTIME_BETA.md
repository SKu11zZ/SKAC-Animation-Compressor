# Native Runtime Beta / 原生运行时 Beta

[English](#english) · [中文](#chinese)

<a id="english"></a>

## English

The Runtime Beta is the first engine-facing path for playing a complete `.skac` file
without starting Python. It consists of a C++17 decoder with a stable C ABI, a Unity
C# package, and an Unreal runtime plugin. No compiled binaries are stored in this
repository.

### What works now

- parse and validate a format-1.0 `.skac` container;
- inflate through built-in zlib or a host callback;
- decode the complete clip into immutable track data;
- query frame rate, duration, joint names, and parents;
- sample local transforms by frame or time with clamp/loop behavior;
- reuse caller-owned pose buffers during playback;
- use the same public decoder from Unity and Unreal.

The C ABI returns quaternion `xyzw` plus translation `xyz`. Decoder instances are
read-only after opening. Separate decoder instances can be sampled on separate
threads; diagnostics are thread-local.

### Beta limits

- same-character playback only in the native layer;
- whole-clip loading, not chunked or streaming decode;
- source coordinate convention and units are preserved;
- Unity and Unreal adapters do not yet drive a character automatically;
- the Python Profile 2.0 cross-skeleton path is not yet compiled into this runtime;
- the public repository provides source adapters, not prebuilt engine binaries.

These limits matter: the existing Python Profile workflow and the native playback
Beta are related parts of the project, but they are not yet one production runtime.

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
```

The comparison uses a generated public test motion and writes temporary metadata and
payload files outside the release tree.

[Jump to Chinese](#chinese)

---

<a id="chinese"></a>

## 中文

Runtime Beta 是第一条不启动 Python、直接在引擎侧播放完整 `.skac` 文件的路径。它由
C++17 解码核心、稳定的 C ABI、Unity C# Package 和 Unreal Runtime Plugin 组成。仓库只
提交源码，不提交编译后的动态库。

### 现在能做什么

- 解析并校验格式 1.0 的 `.skac` 容器；
- 使用内置 zlib，或者把解压交给宿主引擎；
- 一次载入完整动画并生成只读轨道数据；
- 查询帧数、时长、关节名和父子关系；
- 按帧或按时间采样，支持截断与循环；
- 播放时重复使用调用方提供的姿态缓冲区；
- Unity 和 Unreal 共用同一个公开解码核心。

C ABI 输出四元数 `xyzw` 和位移 `xyz`。Decoder 打开后只读；不同 Decoder 可以放在不同
线程采样，错误信息按线程保存。

### Beta 的边界

- 原生层当前只做同角色播放；
- 当前是整段载入，不是分块或流式解码；
- 保留源动画的坐标系和单位；
- Unity、Unreal 适配层暂不自动驱动角色；
- Python 的 Profile 2.0 跨骨骼路径还没有编译进原生运行时；
- 公开仓库提供适配源码，不提供预编译引擎二进制。

这些边界需要明确：现有 Python Profile 流程和原生播放 Beta 属于同一项目，但现在还不是
一套已经合并完成的生产运行时。

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
```

比较使用程序生成的公开测试动画，临时元数据和 Payload 会写在发布目录之外。

[Jump to English](#english)
