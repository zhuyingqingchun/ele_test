# 第8轮：振动增强版正式重写方案报告

## 1. 目标

本轮补丁用于在“quality-aware fusion + modality dropout”验证失败后，进一步判断问题是否来自**振动分支本身的特征提取能力不足**。

设计目标不是引入重型 2D 网络，而是在当前四模态主框架下，用一个**轻量三支路振动编码器**替换原始 `VibrationStateEncoder`：

- 原始时域支路（raw time branch）
- 包络支路（envelope branch）
- 轻量时频支路（STFT branch）

并通过分支级 soft gate 做轻量融合。

---

## 2. 当前问题

原始振动编码器本质上只有：

1. 一个浅时域卷积分支
2. 一个 `rfft` 后线性映射的频域分支
3. 最后直接相加

这会带来三个问题：

- 时域冲击特征与缓慢背景变化没有显式分离
- 包络类机械故障证据没有被单独建模
- 时频分支过于粗糙，且与时域分支是“直接相加”，缺少选择机制

因此即使振动模态本身有信息，也可能因为提取方式过弱而在融合时被进一步边缘化。

---

## 3. 本轮补丁修改内容

### 3.1 重写 `VibrationStateEncoder`

文件：

- `ele_servo_paper_repro_bundle/servo_diagnostic/multimodal_method.py`

修改为三支路结构：

#### A. 原始时域支路

使用两条轻量卷积分支：

- `raw_local_branch`：较短卷积核，偏局部冲击
- `raw_context_branch`：较长卷积核，偏较长时间尺度上下文

两者相加后形成 raw tokens。

#### B. 包络支路

先对绝对值振动信号进行平滑，得到近似包络，再进入轻量卷积支路提取 envelope tokens。

这一步主要对应机械冲击、滚动体故障、重复性冲击调制等更敏感的证据。

#### C. 轻量时频支路

用 `torch.stft` 生成轻量时频幅值图，不走重型 2D CNN，只做：

- STFT 幅值
- 对通道求均值
- 线性投影到 `model_dim`
- 自适应池化为固定 token 数

这样既保留局部时频变化，又不把当前系统改成重型图像网络。

#### D. 分支级 soft gate

三支路 token 叠加后，不再直接相加，而是：

- 先计算分支 summary
- 再通过 `branch_gate` 得到 3 个分支权重
- 最终按 softmax 权重融合三支路 tokens

这样可以在不同样本上自适应强调：

- 原始冲击
- 包络调制
- 时频局部模式

并保留 `last_branch_weights` 作为后续分析入口。

### 3.2 提高振动分支容量

文件：

- `ele_servo_paper_repro_bundle/experiments_smoke_20260316/exp1_decoupled_models.py`

将 vibration branch 从：

- `token_count=12`
- `hidden_dim=32`

提升为：

- `token_count=16`
- `hidden_dim=64`

目的：避免振动模态在容量上先天弱于 electrical / position。

---

## 4. 为什么这版结构比原版更合理

### 4.1 更贴机械故障机理

机械类振动故障通常不只是“频域能量变化”，还会表现为：

- 时间域的冲击或脉冲
- 包络域的周期性调制
- 时频域的非平稳局部异常

原模型只做了“time + rfft”两路，非常容易漏掉包络证据。

### 4.2 仍然保持轻量

这版没有引入重型 2D backbone，也没有把主系统改成图像模型。

额外复杂度主要来自：

- 两条轻量 raw 分支
- 一条 envelope 分支
- 一条小型 STFT 投影支路
- 一个分支级 gate

与当前四模态主系统仍然兼容。

### 4.3 适合后续一致性评估

由于保留了 `last_branch_weights`，后续可以扩展导出：

- 哪类故障更依赖 raw / envelope / tf
- 机械类故障是否明显提高 envelope 分支权重
- bearing / backlash / jam 是否表现出不同的分支偏好

这对“模态—故障证据一致性评估”是有帮助的。

---

## 5. 推荐实验顺序

这个补丁不建议在第一优先级之前直接用，而建议在以下条件成立后再使用：

1. 已经完成 `quality_aware_fusion + modality_drop_prob` 验证
2. 结果显示 vibration 仍不稳定
3. 或者 gate 已经允许 vibration 进入，但性能仍无明显提升

如果满足以上条件，再切换到本补丁版本。

---

## 6. 训练建议

为了让这版振动编码器发挥作用，建议先复跑：

### 主线

- Stage 1
- Stage 2

优先观察：

- A0 全模态是否优于 A4 无振动
- mechanical-related classes 上的提升是否更明显

### 消融

重点重新比较：

- `A0_full`
- `A4_no_vibration`

如果此时 A0 重新稳定超过 A4，说明“振动特征提取不足”确实是之前的重要原因之一。

---

## 7. 风险与注意事项

### 7.1 STFT 分支会引入额外训练耗时

虽然是轻量版本，但相比原始 `rfft + linear`，本版仍然会略增计算量。

### 7.2 包络构造是近似实现

当前补丁用的是“绝对值 + 平滑”近似包络，而不是更复杂的 Hilbert 包络；优点是实现简单、可训练、与现有系统兼容。

### 7.3 仍需结合机械类任务验证

若数据集中的机械类故障比例偏低，或标签本身对 vibration 不敏感，则即使结构增强，也未必一定带来很大总体精度提升。

因此更重要的是看：

- 机械类子集表现
- vibration 相关 gate / branch 权重
- A0 vs A4 的变化趋势

---

## 8. 本轮交付文件

- 振动增强版正式重写补丁：
  - `ele_test_vibration_encoder_patch_round8.patch`

---

## 9. 本轮校验

本轮已完成以下静态校验：

1. 补丁格式检查通过
2. `patch -p1 --dry-run` 在构造的目标文件上通过
3. 修改后的振动编码器核心代码通过 Python AST 语法解析

未完成的部分：

- 未在你的完整本地仓库上执行真实训练
- 未对真实数据进行前向性能和显存测试

因此本补丁当前状态为：

- **静态可应用**
- **语法可通过**
- **训练效果需你本地复跑验证**
