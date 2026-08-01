# Codec runtime quality gates

[English](#english) · [中文](#chinese)

<a id="english"></a>

## English

There are two gates because there are two playback routes. Combining them would make
the numbers ambiguous.

Same character means encode, decode, and play on the source skeleton. It has no target
argument and does not build or execute a Profile:

```text
python -m skac_codec quality-gate-same source.bvh \
  --output reports/same-character.json \
  --visual reports/same-character.svg
```

Different character means encode, decode, then play through one frozen, compiled
source-to-target Profile:

```text
python -m skac_codec quality-gate-different source.bvh target.bvh \
  --output reports/different-character.json \
  --visual reports/different-character.svg
```

The old `quality-gate source.bvh target.bvh` command remains as a compatibility route.
It classifies equal skeleton signatures as same-character. New scripts should use the
explicit commands above.

Both commands require JSON and SVG outputs. JSON is for CI and Agents; SVG is a
self-contained visual report with no external assets.

### Same-character gate

The default high-quality route freezes these limits:

| Check | Limit |
| --- | ---: |
| Codec maximum local rotation error | <= 0.0625 degrees |
| Codec maximum global position error | <= 0.001 skeleton heights |
| Decoded quaternion norm error | <= 0.0000000001 |
| Whole-clip decode speed | >= 10x real time |

The report also records encoded size, compression ratio, and bits per joint per frame.
There are deliberately no mapping, Profile, or cross-skeleton timing fields.

### Different-character gate

This route includes every Codec check above, then adds:

| Check | Limit |
| --- | ---: |
| Shared core-joint mapping coverage | >= 100% |
| Compiled/runtime maximum rotation difference | <= 0.00001 degrees |
| Compiled/runtime maximum global position difference | <= 0.000000001 skeleton heights |
| Runtime quaternion norm error | <= 0.0000000001 |
| Sustained target-frame p95 | <= 4 ms |
| Decode plus target-playback preparation | >= 5x real time |

The slower matrix path is only a validation oracle. It is not used in playback. Frame
timing uses consecutive batches of up to ten frames and reports the p95 of per-frame
batch averages. Reports include Python, NumPy, and platform versions because timing is
machine-dependent.

The different-character gate does not claim perceptual target-motion accuracy without
target ground truth. It proves Codec bounds, shared mapping coverage, equivalence to
the reference transform, and playback cost. Dataset retargeting scores remain separate.

The public Python decoder currently reconstructs a complete clip before playback.
Chunked and random-access decoding are later format milestones.

### Native Runtime Beta gate

The generated Release fixture adds engine-facing regression limits without replacing
the two animation-quality gates above:

| Check | Limit |
| --- | ---: |
| 65-joint same-character time-sample p95 | <= 0.10 ms |
| 65-to-67-joint Profile time-sample p95 | <= 0.25 ms |
| 100 sequential Profile instances, whole tick p95 | <= 16.667 ms |

The benchmark uses 2,000 measured ticks after warmup and excludes file I/O. It is a
local performance regression gate, not a cross-machine comparison. Each retargeter is
created before timing and reuses its scratch and output buffers.

[跳到中文](#chinese)

---

<a id="chinese"></a>

## 中文

现在有两套门槛，因为实际就有两条播放路线。混在一张报告里，数字很容易说不清。

同角色就是压缩、解压，然后回到源骨架播放。这个命令没有目标骨架参数，也不会创建或
执行 Profile：

```text
python -m skac_codec quality-gate-same source.bvh \
  --output reports/same-character.json \
  --visual reports/same-character.svg
```

不同角色是在解压之后，通过一次性冻结并编译好的源到目标 Profile 播放：

```text
python -m skac_codec quality-gate-different source.bvh target.bvh \
  --output reports/different-character.json \
  --visual reports/different-character.svg
```

旧的 `quality-gate source.bvh target.bvh` 还留着兼容；它会按照骨架签名是否相同自动判断。
新脚本直接用上面两个明确入口。

两个命令都必须生成 JSON 和 SVG。JSON 给 CI 和 Agent 读，SVG 是不带外部依赖的可视化
报告，浏览器直接打开就行。

### 同角色门槛

默认 high 档检查：

| 检查项 | 固定门槛 |
| --- | ---: |
| Codec 最大局部旋转误差 | <= 0.0625 度 |
| Codec 最大全局位置误差 | <= 骨架高度的 0.001 |
| 解压后四元数长度误差 | <= 0.0000000001 |
| 整段解压速度 | >= 10 倍实时 |

报告也会记录文件大小、压缩比和每关节每帧位数。这里不会出现映射覆盖、Profile 或跨骨骼
耗时，因为这条路线根本不执行它们。

### 不同角色门槛

除了上面的 Codec 检查，还会增加：

| 检查项 | 固定门槛 |
| --- | ---: |
| 双方共有核心关节的映射覆盖率 | >= 100% |
| 编译运行时和参考实现的最大旋转差 | <= 0.00001 度 |
| 编译运行时和参考实现的最大全局位置差 | <= 骨架高度的 0.000000001 |
| 运行时四元数长度误差 | <= 0.0000000001 |
| 持续目标骨架逐帧耗时 p95 | <= 4 ms |
| 解压加目标骨架播放准备 | >= 5 倍实时 |

较慢的矩阵版本只拿来做数学对照，不进入播放。逐帧计时会连续跑最多十帧，再统计每帧
摊销值的 p95。性能数字跟机器有关，所以报告会记录 Python、NumPy 和平台版本。

没有目标动作真值时，不同角色报告不会假装自己证明了视觉上绝对正确。它能证明的是
Codec 误差、共有映射覆盖、快速执行器与参考实现一致，以及播放速度。数据集上的重定向
分数仍然单独报告。

当前公开 Python 解码器会先还原完整动画；分块和随机帧解码还是后续格式能力。

### 原生 Runtime Beta 门槛

生成型 Release 夹具增加了引擎侧回归门槛，但不会替代上面的两套动画质量检查：

| 检查项 | 固定门槛 |
| --- | ---: |
| 65 关节同角色时间采样 p95 | <= 0.10 ms |
| 65 到 67 关节 Profile 时间采样 p95 | <= 0.25 ms |
| 100 个 Profile 实例串行整帧 p95 | <= 16.667 ms |

基准会预热后测量 2,000 个 Tick，并排除文件读写。它是本机性能回归门槛，不是跨机器比较。
Retargeter 全部在计时前创建，测试期间重复使用临时缓冲区和输出缓冲区。

[Jump to English](#english)
