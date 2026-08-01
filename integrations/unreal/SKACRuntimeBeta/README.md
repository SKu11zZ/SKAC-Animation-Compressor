# SKAC Runtime Beta for Unreal Engine

Copy `SKACRuntimeBeta` into a project's `Plugins` directory and regenerate project
files. The module compiles the public native decoder directly and delegates `.skac`
zlib inflation to Unreal's compression API, so no extra native binary is checked in.

`FSkacClip` keeps one native pose buffer and exposes frame- or time-based sampling as
local `FTransform` values. The values still use the source animation's axes and units;
perform the project-specific conversion in the import or animation layer.

Call `OpenRetargeter` with the frozen Profile JSON and generated runtime target-skeleton
JSON, then use `SampleRetargetedFrame` or `SampleRetargetedTime`. The module reuses a
target native pose buffer for the life of that playback instance.

This Beta performs whole-clip same-character and Profile 2.0 playback. It does not yet
build an Animation Sequence, stream chunks from disk, or apply project coordinate
conversion automatically. See the repository's `RUNTIME_BETA.md` for the exact scope.

---

# Unreal Engine 接入说明

把 `SKACRuntimeBeta` 复制到项目的 `Plugins` 目录并重新生成工程文件。模块会直接编译
仓库中的公开原生解码核心，并把 `.skac` 的 zlib 解压交给 Unreal 自带的压缩接口，因此
仓库里不需要提交额外二进制文件。

`FSkacClip` 会复用一份原生姿态缓冲区，并按帧或时间输出局部 `FTransform`。输出仍然沿用
源动画的坐标轴和单位，项目应当在导入层或动画层做一次统一转换。

把冻结 Profile JSON 和生成好的目标骨骼运行时 JSON 交给 `OpenRetargeter`，随后调用
`SampleRetargetedFrame` 或 `SampleRetargetedTime`。每个播放实例会一直复用自己的目标
原生姿态缓冲区。

当前 Beta 支持整段同角色解码和 Profile 2.0 播放；它暂不生成 Animation Sequence，
没有分块流式读取，也不会自动完成项目坐标转换。完整范围见仓库根目录的
`RUNTIME_BETA.md`。
