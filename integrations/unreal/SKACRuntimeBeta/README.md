# SKAC Runtime Beta for Unreal Engine

Copy `SKACRuntimeBeta` into a project's `Plugins` directory and regenerate project
files. The module compiles the public native decoder directly and delegates `.skac`
zlib inflation to Unreal's compression API, so no extra native binary is checked in.

`FSkacClip` keeps one native pose buffer and exposes frame- or time-based sampling as
local `FTransform` values. The values still use the source animation's axes and units;
perform the project-specific conversion in the import or animation layer.

This Beta performs whole-clip, same-character decoding. It does not yet build an
Animation Sequence, run a cross-skeleton Profile, or stream chunks from disk. See the
repository's `RUNTIME_BETA.md` for the exact scope.

---

# Unreal Engine 接入说明

把 `SKACRuntimeBeta` 复制到项目的 `Plugins` 目录并重新生成工程文件。模块会直接编译
仓库中的公开原生解码核心，并把 `.skac` 的 zlib 解压交给 Unreal 自带的压缩接口，因此
仓库里不需要提交额外二进制文件。

`FSkacClip` 会复用一份原生姿态缓冲区，并按帧或时间输出局部 `FTransform`。输出仍然沿用
源动画的坐标轴和单位，项目应当在导入层或动画层做一次统一转换。

当前 Beta 只支持整段载入和同角色解码；它暂不生成 Animation Sequence，不在原生层运行
跨骨骼 Profile，也没有分块流式读取。完整范围见仓库根目录的 `RUNTIME_BETA.md`。
