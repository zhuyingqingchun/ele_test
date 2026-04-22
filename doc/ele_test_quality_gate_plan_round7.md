# 第7轮：quality-aware fusion + modality dropout + gate 导出补丁方案

## 1. 本轮目标

本轮补丁只做最小改动验证，不重写振动分支编码器。

目标有四个：

1. 打开 `quality_aware_fusion`
2. 加入 `modality_drop_prob`，优先验证是否存在 `electrical/position` 对 `vibration` 的压制
3. 导出每个样本与每类故障的 `gate / branch energy`
4. 支持用同一套配置重新比较 `A0_full` 与 `A4_no_vibration`

---

## 2. 本轮补丁改动文件

补丁覆盖 4 个文件：

- `ele_servo_paper_repro_bundle/experiments_smoke_20260316/exp1_decoupled_models.py`
- `ele_servo_paper_repro_bundle/experiments_smoke_20260316/train_exp1_decoupled_stages.py`
- `ele_servo_paper_repro_bundle/experiments_smoke_20260316/run_exp1_decoupled_modality_tf_control_curriculum.sh`
- `ele_servo_paper_repro_bundle/experiments_smoke_20260316/run_exp1_decoupled_modality_ablation_async.sh`

---

## 3. 代码层改动说明

### 3.1 模型输出增加 gate / branch energy

在 `StageOutputs` 中新增：

- `quality_gates`
- `branch_energies`

并在 `DecoupledSignalTokenBackbone.forward()` 中返回：

- 当前 batch 的四模态 gate
- 当前 batch 的四模态 token energy

这样不改变原有主流程，只是把质量感知融合内部已经存在的信息显式导出。

### 3.2 训练/评估阶段导出 gate 诊断文件

在 `train_exp1_decoupled_stages.py` 中：

- `DecoupledBatch` 新增 `sample_index`
- `evaluate(..., return_predictions=True)` 时收集：
  - `sample_indices`
  - `quality_gates`
  - `branch_energies`
- 训练结束后自动导出：
  - `test_quality_gates.csv`
  - `test_quality_gate_summary.json`

其中：

- CSV 面向样本级排查
- JSON 面向类级/家族级统计

### 3.3 shell 脚本默认打开 quality-aware fusion

两个 runner 都增加以下默认开关：

- `QUALITY_AWARE_FUSION=1`
- `QUALITY_DROP_PROB=0.10`
- `QUALITY_HIDDEN_DIM=128`
- `QUALITY_MIN_GATE=0.10`
- `LAMBDA_QUALITY=0.05`

并自动拼接到 Stage 1~4 命令中。

这样主线训练和 A0/A4 消融会使用同一套质量感知设置，避免结论口径不一致。

---

## 4. 新导出文件的用途

### 4.1 `test_quality_gates.csv`

面向样本级检查，包含：

- `sample_index`
- `target_scenario`
- `target_family`
- `pred_scenario`
- `pred_family`
- `position/electrical/thermal/vibration gate`
- `position/electrical/thermal/vibration energy`
- `mechanical_related`

适合回答：

- 哪些机械类样本 vibration gate 很高
- 哪些电气类样本 vibration gate 持续很低
- A0 与 A4 的差别究竟发生在哪些样本上

### 4.2 `test_quality_gate_summary.json`

面向统计级检查，包含：

- overall summary
- mechanical-related summary
- non-mechanical summary
- by true family
- by true class

适合回答：

- vibration 在机械相关 family 中是否被明显激活
- vibration 是否只在少数类有效
- electrical/position 是否长期占主导

---

## 5. 推荐复现实验顺序

### 实验 1：主线对照

重新运行当前主脚本：

```bash
git apply ele_test_quality_gate_vibration_patch_round7.patch
bash ele_servo_paper_repro_bundle/experiments_smoke_20260316/run_exp1_decoupled_modality_tf_control_curriculum.sh
```

重点检查：

- `stage2_signal_fusion/report.json`
- `stage2_signal_fusion/test_quality_gate_summary.json`

观察点：

- vibration 是否不再长期最低
- mechanical-related summary 中 vibration 的 gate 是否明显抬升

### 实验 2：A0 vs A4 再比较

重新运行异步消融：

```bash
bash ele_servo_paper_repro_bundle/experiments_smoke_20260316/run_exp1_decoupled_modality_ablation_async.sh
```

重点比较：

- `A0_full/stage2_signal_fusion/report.json`
- `A4_no_vibration/stage2_signal_fusion/report.json`
- 两边的 `test_quality_gate_summary.json`

目标不是只看最终 acc，而是看：

- A0 是否重新超过 A4
- vibration 是否在机械相关 family 中成为显著 gate

---

## 6. 如何解释可能结果

### 情况 A：A0 重新超过 A4，且机械类 vibration gate 上升

结论：

- 根因主要是融合层的强模态压制
- 当前振动分支提取的特征并非无效，只是之前未被有效使用

下一步：

- 保留当前 encoder
- 转向做模态—故障证据一致性评估

### 情况 B：A0 仍不如 A4，但 vibration 在机械类确实被激活

结论：

- vibration 学到了局部有效证据
- 但全局分类目标下仍带来额外噪声

下一步：

- 给 vibration 加机械类辅助监督
- 做 family-aware 或 class-aware gate regularization

### 情况 C：A0 仍不如 A4，且 vibration 在机械类也没有被激活

结论：

- 问题不只是融合压制
- vibration encoder 本身对机械故障特征提取不足

下一步：

- 进入“振动增强版正式补丁”
- 直接重写 vibration encoder 为 raw + envelope + STFT 三支路

---

## 7. 本轮补丁的定位

这不是最终结构升级补丁，而是一个“根因定位补丁”。

优点：

- 对现有主线侵入小
- 能快速判断 vibration 问题是“融合压制”还是“编码不足”
- 还能顺手补上你后续论文所需的 gate/branch 解释证据

风险控制：

- 不改变数据集
- 不改变类定义
- 不直接重写大结构
- 只在已有 quality-aware 分支基础上显式启用和导出

---

## 8. 本轮建议默认参数

建议先保持：

- `QUALITY_DROP_PROB=0.10`
- `QUALITY_MIN_GATE=0.10`
- `LAMBDA_QUALITY=0.05`

不要第一轮就把 dropout 拉太高，否则可能误伤本来已经稳定的 electrical/position 表征。

如果第一轮 gate 变化太小，再试：

- `QUALITY_DROP_PROB=0.15`

但不建议一开始超过 `0.20`。

---

## 9. 最终建议

当前阶段最合理的动作不是立刻重写振动分支，而是先做这一步最小改动验证。

只有在以下两条同时成立时，才进入正式振动增强补丁：

1. A0 仍持续不如 A4
2. mechanical-related summary 中 vibration gate 仍长期偏低

如果这两条同时出现，就可以比较有把握地说明：

> 当前问题已经不只是融合压制，而是 vibration encoder 的故障特征建模能力本身不足。
