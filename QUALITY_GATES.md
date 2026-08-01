# Codec runtime quality gates

[English](#english) · [中文](#chinese)

<a id="english"></a>

## English

SKAC is built around a compressed animation that can be prepared and played on more
than one public skeleton. Profile construction is offline; playback is the hot path.
The gate therefore treats visual reconstruction and runtime cost as two hard limits,
not as a trade where either side can silently erase the other.

Run one gate on a public source animation and a public target skeleton:

```text
python -m skac_codec quality-gate source.bvh target.bvh \
  --output reports/quality-gate.json \
  --visual reports/quality-gate.svg
```

Both outputs are required. JSON is for CI and Agents. SVG is a self-contained visual
summary that opens in a browser without external assets.

The default high-quality gate checks:

| Check | Frozen limit |
| --- | ---: |
| Codec maximum local rotation error | <= 0.0625 degrees |
| Codec maximum global position error | <= 0.001 skeleton heights |
| Shared core-joint mapping coverage | >= 100% |
| Compiled/runtime maximum rotation difference | <= 0.00001 degrees |
| Compiled/runtime maximum global position difference | <= 0.000000001 skeleton heights |
| Runtime quaternion norm error | <= 0.0000000001 |
| Sustained retarget frame p95 | <= 4 ms |
| Whole-clip decode speed | >= 10x real time |
| Decode plus target-playback preparation | >= 5x real time |

The compiled/runtime comparison uses the slower matrix implementation as a validation
oracle. That implementation is not the product path. It exists so an optimization
cannot change the frozen transform result unnoticed.

Frame timing uses consecutive batches of up to ten frames and reports the p95 of the
per-frame batch averages. This removes timer and scheduler spikes while still measuring
sustained playback throughput. Timing is machine-dependent, so reports include Python, NumPy, and platform versions.
Do not compare two performance reports as if they came from the same machine unless
their environments match. Threshold overrides are recorded in the JSON; lowering a
threshold does not count as passing the standard gate.

This gate does not claim perceptual target-motion accuracy without target ground truth.
It proves Codec reconstruction bounds, mapping completeness for shared body semantics,
mathematical equivalence to the reference transform, and playback cost. Dataset scoring
remains a separate report.

The current public Python decoder reconstructs a complete clip before playback. Its
full-clip speed is gated, and the compiled target step is measured one frame at a time.
Chunked random-access decoding is still a later format milestone.

[跳到中文](#chinese)

---

<a id="chinese"></a>

## 中文

SKAC 的主线是先把动画压成 `.skac`，播放前解压，然后让同一段动画可以送到不同的
公开骨架。Profile 是离线编译步骤，真正需要盯性能的是播放热路径。所以质量和速度
都是硬门槛，不能为了让其中一个数字好看，偷偷牺牲另一个。

对一段公开源动画和一个公开目标骨架执行：

```text
python -m skac_codec quality-gate source.bvh target.bvh \
  --output reports/quality-gate.json \
  --visual reports/quality-gate.svg
```

JSON 和 SVG 都必须生成。JSON 给 CI 和 Agent 读；SVG 是不带外部依赖的可视化报告，
浏览器直接打开就能看。

默认 high 档会卡住这些指标：

| 检查项 | 固定门槛 |
| --- | ---: |
| Codec 最大局部旋转误差 | <= 0.0625 度 |
| Codec 最大全局位置误差 | <= 骨架高度的 0.001 |
| 双方共有核心关节的映射覆盖率 | >= 100% |
| 编译运行时和矩阵参考的最大旋转差 | <= 0.00001 度 |
| 编译运行时和矩阵参考的最大全局位置差 | <= 骨架高度的 0.000000001 |
| 运行时四元数长度误差 | <= 0.0000000001 |
| 持续逐帧重定向耗时 p95 | <= 4 ms |
| 整段解压速度 | >= 10 倍实时 |
| 解压加目标骨架播放准备 | >= 5 倍实时 |

矩阵版本只用来做数学对照，不是产品播放路径。这样以后继续优化四元数执行器时，
只要结果发生了不该有的变化，门槛就会直接报错。

逐帧计时会连续执行最多十帧，再统计每帧摊销值的 p95，避免把计时器和系统调度尖峰
误当成算法耗时，同时仍然检查持续播放吞吐。性能数字跟机器有关，所以报告会记 Python、NumPy 和平台版本。环境不同的报告不要
直接当成同机对比。命令允许临时覆盖阈值，但新阈值会明确写进 JSON；把门槛调低，
不算通过标准门槛。

没有目标动作真值时，这份报告不会假装自己证明了“视觉上绝对正确”。它证明的是
Codec 还原误差、共有核心关节覆盖、快速执行器与参考算法一致，以及播放性能。
数据集上的重定向分数仍然单独报告。

当前公开 Python 解码器会先还原完整动画，再逐帧送进目标骨架。整段解码速度和逐帧
目标执行速度都已经进门槛；压缩流的分块和随机帧解码仍是后续格式能力。

[Jump to English](#english)
