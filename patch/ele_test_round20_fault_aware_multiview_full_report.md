# 第20轮：第19轮推荐内容综合实现补丁说明

## 一、交付文件

- 综合实现补丁：
  - `ele_test_round20_fault_aware_multiview_full.patch`
- 综合实现报告：
  - `ele_test_round20_fault_aware_multiview_full_report.md`

## 二、补丁目标

本补丁在第14轮综合版基础上，继续把第19轮建议中的核心内容接入当前 Stage 3/4 主线：

1. fault-aware gate
2. multi-view alignment
3. modality-level evidence alignment
4. thermal / position 受控补强
5. 评估体系升级
6. 新实验 runner

## 三、涉及部署位置

补丁中的目标位置为仓库根目录相对路径：

- `ele_servo_paper_repro_bundle/experiments_smoke_20260316/exp1_decoupled_models.py`
- `ele_servo_paper_repro_bundle/experiments_smoke_20260316/train_exp1_decoupled_stages.py`
- `ele_servo_paper_repro_bundle/experiments_smoke_20260316/run_exp1_fault_aware_gate_multiview_align.sh`

如果你之前未应用第14轮综合补丁，这一版补丁应视为“增强版综合补丁”，建议直接单独使用，不要和第14轮补丁交叉重复应用。

## 四、实现内容摘要

### 1. fault-aware gate
在 Stage 3/4 中新增 `FaultAwareGate`：
- 输入：modality embeddings + pooled signal + evidence vec + mechanism vec
- 输出：`fault_gate_logits` 与 `fault_gates`
- 用于替代“只看 corruption target 的质量门控”不足

### 2. multi-view alignment
Stage 3/4 不再只依赖一个 `combined_text`：
- `combined`
- `evidence`
- `mechanism`
- `contrast`

其中：
- evidence / mechanism 参与正向对齐
- contrast 参与 rejection penalty

### 3. modality-level evidence alignment
新增：
- `evidence_scores`
- `mechanism_scores`
- `contrast_scores`

训练中加入：
- evidence 主模态监督
- mechanism 主/辅模态分布监督
- gate prior 监督

### 4. thermal / position 受控补强
延续第14轮的轻量增强：
- position：tracking_error / encoder_gap
- thermal：trend / delta / thermal_gap

### 5. 评估体系升级
新增导出：
- `test_modality_mechanism_summary.json`
- `test_contrast_rejection_summary.json`
- `test_vibration_branch_weight_summary.json`
- `test_counterfactual_modality_sensitivity.json`

### 6. 新实验
新增 runner：
- `run_exp1_fault_aware_gate_multiview_align.sh`

默认实验名：
- `exp1_fault_aware_gate_multiview_align_v1`

## 五、推荐运行方式

把补丁文件放到仓库根目录后执行：

```bash
git apply ele_test_round20_fault_aware_multiview_full.patch
```

然后运行：

```bash
bash ele_servo_paper_repro_bundle/experiments_smoke_20260316/run_exp1_fault_aware_gate_multiview_align.sh
```

## 六、本地校验说明

这次我已经做过的本地静态校验：

1. 补丁文件已成功生成
2. 新 runner 脚本内容已做 `bash -n` 语法检查
3. 新增 Python 结构片段做了 `ast.parse` 语法检查
4. 补丁整体 unified diff 结构已检查

这次没有完成的部分：

- 没有在你的真实仓库上执行 `git apply --check`
- 没有完成训练级运行验证
- 没有与第14轮补丁做二次自动合并测试

## 七、结果解释建议

这个新实验最重要看三类结果：

1. Stage 3 分类是否继续保持最优或接近最优
2. Stage 4 的 evidence / mechanism / contrast 指标是否进一步提升
3. counterfactual modality drop sensitivity 是否更符合故障机理预期
