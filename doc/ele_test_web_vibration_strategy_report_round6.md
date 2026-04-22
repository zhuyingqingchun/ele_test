# 第6轮：振动模态改进策略检索分析报告

## 1. 本轮目标
围绕“让振动模态真正起作用”这一目标，结合当前仓库四模态主线代码与网页检索到的多模态融合、模态dropout、振动故障诊断相关算法与论文，给出：

1. 最小改动实验补丁路线是否值得先做；
2. 振动增强版正式重写路线应该怎样设计；
3. 两条路线的优先级、风险、收益与实施顺序。

---

## 2. 当前仓库的关键现状

当前四模态主线里：

- Stage 1：四分支信号编码器 + 分类头
- Stage 2：在四模态 token 上做 Transformer 融合
- 脚本默认开启 `feature-mode modality_tf`
- 但默认没有打开 `quality_aware_fusion`
- 也没有在主脚本里显式使用模态 dropout 来压制强模态依赖
- vibration 分支当前是较轻量的 `VibrationStateEncoder`

由已有实验现象可推断：

- vibration 单模态不是完全无信息；
- 但在融合场景中，振动没有稳定带来正收益；
- 问题更像是“强模态压制 + 振动分支提特征能力不够 + 任务监督没有强制模型必须利用振动”。

---

## 3. 网页检索后的核心结论

### 3.1 支持“最小改动补丁”路线的证据

检索到的多模态融合文献显示，以下思路是成熟且低风险的：

#### (1) ModDrop / modality dropout
经典 ModDrop 工作提出：在渐进融合过程中随机丢弃某些模态，可以在学习跨模态相关性的同时保留各模态特有表征，并提升对缺失/噪声模态的鲁棒性。

这与当前仓库里已经预留的：

- `quality_aware_fusion`
- `modality_drop_prob`
- `quality_min_gate`

非常契合。

这说明：
**先做“quality-aware fusion + modality dropout + gate 导出”是合理且有文献支撑的低风险路线。**

#### (2) 动态权重 / lazy modality 激活
较新的多模态融合工作强调：低质量、弱势或“lazy modality”不能只靠简单拼接自动发挥作用，需要显式地做动态权重、关系量化或 lazy modality 激活。

这类工作说明：
**如果 vibration 在训练中被 electrical / position 压制，那么引入动态门控、模态级权重统计和按样本的可靠性调节，是更合适的第一步。**

#### (3) 融合瓶颈（fusion bottleneck）
Attention bottleneck 类方法强调：不要让所有模态 token 直接全连接混合，而是通过少量瓶颈 token 交换信息，从而减少强模态对弱模态的淹没。

这对当前 Stage 2 的启示是：
如果后续要继续升级融合结构，可以考虑让 vibration 与其他模态的信息交换经过受限 bottleneck，而不是直接大拼接后统一自注意力。

### 3.2 支持“振动增强版正式重写”路线的证据

振动故障诊断检索结果非常一致地指向以下事实：

#### (1) 包络谱对冲击类故障非常关键
多篇 bearing / rotating machinery 相关工作都强调：局部缺陷和冲击故障往往在原始振动中不够显著，但在包络谱中会形成更稳定的特征峰或周期性结构。

这说明当前 vibration 分支如果只做浅层卷积 + rFFT 线性映射，通常不足以把“冲击解调信息”显式提出来。

#### (2) 时频表示适合非平稳振动
多篇 2024–2026 的工作表明，CWT、STFT、wavelet time-frequency representation 对非平稳振动、变工况和噪声环境更友好；特别是当故障在时间上是突发、在频率上是局部聚集时，时频图比纯时域或纯FFT更稳。

这说明：
如果你们的振动模态对应的是机械冲击、间歇阻塞、轴承类特征，那么只保留 1D 时域卷积和简单 FFT，很可能表达不够。

#### (3) 多尺度分支在振动场景里很常见
检索到的振动诊断网络经常采用：

- multi-branch
- multi-scale convolution
- dilated temporal convolution
- time + frequency / time + envelope / time + time-frequency 并行

其共同点是：
**振动特征往往分布在多个时间尺度与多个频带上，单一尺度提取容易丢失关键机械证据。**

---

## 4. 两条路线怎么选

## 结论：先做最小改动补丁，再决定是否进入正式重写

原因如下：

### 4.1 为什么不能直接先重写 vibration encoder
因为你们当前还没有严格证明“问题主要在振动特征提取”。

目前同样可能成立的解释是：

- vibration 特征其实已经够用；
- 只是被 electrical / position 压制；
- 或者 Stage 2 全局分类目标没有驱动模型去用 vibration。

如果不先排除这些情况，直接重写 encoder，会导致：

- 改动大；
- 训练成本高；
- 结果变好也难解释到底是结构变好还是只是容量变大。

### 4.2 为什么最小改动补丁应该先做
它直接回答最关键问题：

> vibration 没起作用，到底是因为“不会提特征”，还是因为“提出来也被融合层忽略”？

如果最小改动后：

- vibration gate 上升；
- 振动相关故障类别收益上升；
- A0 全模态重新超过 A4 无振动；

那就说明主问题是融合压制，不必立即重写 vibration encoder。

如果最小改动后仍然：

- vibration gate 很低；
- 或者 vibration 只在很少样本上被利用；
- 或者与机械类故障无明显对应；

那就说明必须进入“振动增强版正式重写”。

---

## 5. 建议的最小改动实验补丁

### 5.1 必做项

#### A. 打开 quality-aware fusion
在 Stage 1 / Stage 2 主脚本里显式打开：

- `--quality-aware-fusion`
- `--quality-hidden-dim 128`
- `--quality-min-gate 0.10`

目的：
让模型先学“每个样本、每个模态的可靠性/有效性”，而不是直接默认所有模态同权。

#### B. 加 modality dropout（优先对强模态施压）
先做两组：

- 统一模态 dropout：`0.10`
- 偏置式 dropout：对 position / electrical 更强，对 thermal / vibration 更弱

如果代码暂时只支持统一概率，先用统一版；
后续再扩展成 branch-specific dropout。

#### C. 导出每个样本的 gate / branch importance
至少导出：

- pos gate
- electrical gate
- thermal gate
- vibration gate
- 预测类别
- 是否预测正确

然后按 scenario 聚合，看 vibration 是否只在少数机械类故障里被激活。

### 5.2 推荐新增分析

#### D. 做“振动相关故障子集”单独评估
单独统计以下类别的准确率或 F1：

- bearing defect
- jam / intermittent jam
- backlash growth
- friction wear

不要只看总 accuracy。

如果 vibration 真有价值，它应该优先提升这些机械/冲击相关类别，而不是所有类别平均提升。

#### E. 做证据屏蔽实验
对测试集样本：

- 保留全部模态
- 单独置零 vibration
- 比较正确类 logit / margin / top1 变化

并只在机械类故障上看统计量。

---

## 6. 振动增强版正式重写：推荐结构

如果最小改动后仍然证明 vibration 没被有效利用，我建议正式把 vibration encoder 重写成：

## “三支路振动编码器”

### 支路 1：Raw time branch
目的：保留原始瞬态冲击和局部时序模式。

建议：

- 多尺度 1D 卷积（kernel 3 / 7 / 15）
- 或 dilated TCN block
- 输出 token 序列

### 支路 2：Envelope / demodulation branch
目的：显式提取冲击解调后的周期故障特征。

建议：

- 对振动做包络提取（如绝对值+平滑，或 Hilbert envelope）
- 再做 1D conv / TCN
- 或直接做 envelope spectrum 特征映射

这一支路对 bearing、repetitive impact、间歇冲击类故障最关键。

### 支路 3：Time-frequency branch
目的：增强对非平稳、变工况、局部频带异常的建模。

建议选一条轻量路线：

- STFT-based token map（优先）
- 或 CWT scalogram（更强但更重）

如果想控制工程复杂度，先用 STFT，不必一上来就做 2D CWT 大网络。

### 三支路融合方式
不建议简单相加，建议：

- branch attention / gating
- 再做 token concat
- 最后 projection 到统一 `token_dim`

### 容量建议
当前 vibration 分支偏小，建议至少提升到：

- `hidden_dim: 64`
- `token_count: 16~20`

---

## 7. 不建议直接照搬的方案

### 7.1 不建议立刻做重型 2D 图像化大网络
虽然 CWT/时频图 + CNN 文献很多，但你们当前主系统是四分支时序网络，不是纯振动图像分类系统。

如果一上来把 vibration 改成重型 2D backbone，会出现：

- 分支间参数量严重失衡
- 与现有 Stage 1/2 token 接口不对齐
- 振动变好但系统公平性变差

所以更稳妥的做法是：
**先做轻量 STFT token 分支，而不是直接换成大 2D CNN。**

### 7.2 不建议只盲目增加 vibration hidden_dim
如果只把 hidden_dim 从 32 提到 128，但不改变特征表达方式，通常只是增加容量，并不能保证把包络/冲击/时频这些真正关键特征学出来。

---

## 8. 推荐实施顺序

### 阶段 A：最低风险验证（优先）
目标：判断是不是融合压制。

1. 打开 quality-aware fusion
2. 加 modality dropout
3. 导出 gate
4. 看 A0 与 A4 关系是否逆转
5. 看 vibration gate 是否在机械类故障显著上升

### 阶段 B：中等改动
目标：增加 vibration 的监督信号。

1. 给 vibration pooled embedding 增加辅助头
2. 任务可设为：mechanical vs non-mechanical
3. 或 bearing/jam/backlash-sensitive family 分类

### 阶段 C：正式重写 vibration encoder
目标：增强 vibration 特征表达。

1. raw time branch
2. envelope branch
3. STFT time-frequency branch
4. branch attention / gating

### 阶段 D：如果还要进一步升级融合
目标：减少强模态直接淹没弱模态。

1. 在 Stage 2 里引入 fusion bottleneck token
2. 限制跨模态直接全连接自注意力
3. 观察 vibration 是否通过 bottleneck 获得更稳定存在感

---

## 9. 最终建议

### 当前最优决策
**先做“最小改动实验补丁”，不要直接重写 vibration encoder。**

这是因为：

- 文献上有充分依据支持 ModDrop / dynamic weighting / lazy modality activation；
- 你们代码里已经预留了相关机制；
- 这条路线最能先判明问题根源；
- 结果也最容易写进论文：
  - “为缓解强模态压制，引入质量感知融合与模态dropout”；
  - “进一步通过 gate 统计验证振动模态在机械类故障中的激活程度”。

### 何时进入正式重写
只有当最小改动后依然出现：

- vibration gate 低；
- vibration removal 不降反升；
- 机械类故障没有明显依赖 vibration；

才进入正式重写。

### 正式重写时最推荐的设计
**Raw + Envelope + STFT 三支路振动编码器**。

这是当前最符合网页检索结果、也最贴你们电动舵机机械故障机理的一条路线。

---

## 10. 本轮交付说明

本轮只生成长分析说明报告文件，不生成补丁。

建议下一轮直接进入：

1. 最小改动实验补丁版；
或
2. 正式振动增强版补丁。 
