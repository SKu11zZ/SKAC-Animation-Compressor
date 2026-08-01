# SKAC Runtime Beta for Unity

Add this folder as a local Unity package, then place a platform build of the
`skac_runtime` shared library in the Unity project's usual native-plugin directory.
Build the library with `SKAC_RUNTIME_WITH_ZLIB=ON` because `SkacClip.Open` accepts a
complete `.skac` container.

Allocate one `SkacTransform[]` per player and reuse it on every sample. The returned
values are local joint transforms in the coordinate convention and units stored by
the source animation. Convert them once in your importer or playback layer before
assigning them to Unity transforms.

This Beta performs whole-clip, same-character decoding. It does not yet run a Profile
or stream chunks from disk. See the repository's `RUNTIME_BETA.md` for the exact scope.

---

# Unity 接入说明

把这个目录作为本地 Unity Package 加入项目，再把当前平台编译出的
`skac_runtime` 动态库放进 Unity 的原生插件目录。`SkacClip.Open` 直接读取完整
`.skac` 文件，因此动态库需要使用 `SKAC_RUNTIME_WITH_ZLIB=ON` 编译。

每个播放器准备一个 `SkacTransform[]`，逐帧重复使用，避免运行时分配。输出是动画
原坐标系、原单位下的局部关节变换；应当在导入器或播放层统一转换以后，再写入 Unity
骨骼。

当前 Beta 只支持整段载入和同角色解码，暂不在原生层运行 Profile，也没有分块流式读取。
完整范围见仓库根目录的 `RUNTIME_BETA.md`。
