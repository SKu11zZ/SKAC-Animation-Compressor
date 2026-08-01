# SKAC Runtime Beta for Unity

Add this folder as a local Unity package, then place a platform build of the
`skac_runtime` shared library in the Unity project's usual native-plugin directory.
Build the library with `SKAC_RUNTIME_WITH_ZLIB=ON` because `SkacClip.Open` accepts a
complete `.skac` container.

Allocate one `SkacTransform[]` per player and reuse it on every sample. The returned
values are local joint transforms in the coordinate convention and units stored by
the source animation. Convert them once in your importer or playback layer before
assigning them to Unity transforms.

For different-character playback, load the frozen Profile JSON and the generated
runtime target-skeleton JSON, then call `SkacClip.CreateRetargeter`. Keep the clip alive,
allocate one target `SkacTransform[]`, and reuse both for the playback instance.

This Beta performs whole-clip same-character and Profile 2.0 playback. It does not yet
stream chunks from disk or apply project coordinate conversion automatically. See the
repository's `RUNTIME_BETA.md` for the exact scope.

---

# Unity 接入说明

把这个目录作为本地 Unity Package 加入项目，再把当前平台编译出的
`skac_runtime` 动态库放进 Unity 的原生插件目录。`SkacClip.Open` 直接读取完整
`.skac` 文件，因此动态库需要使用 `SKAC_RUNTIME_WITH_ZLIB=ON` 编译。

每个播放器准备一个 `SkacTransform[]`，逐帧重复使用，避免运行时分配。输出是动画
原坐标系、原单位下的局部关节变换；应当在导入器或播放层统一转换以后，再写入 Unity
骨骼。

不同角色播放需要读取冻结 Profile JSON 和生成好的目标骨骼运行时 JSON，然后调用
`SkacClip.CreateRetargeter`。播放期间保持 Clip 存活，每个实例只分配一次目标
`SkacTransform[]` 并重复使用。

当前 Beta 支持整段同角色解码和 Profile 2.0 播放，但还没有分块流式读取，也不会自动完成
项目坐标转换。完整范围见仓库根目录的 `RUNTIME_BETA.md`。
