# 第39轮：新的实验版本

## 实验名

`exp1_stage4_token_evidence_alignment_v1`

## 设计目标

基于第38轮分析，这一版不再重复做 `TopK + dual-level` 的主结构，而是直接把 Stage 4 往 **token-level evidence/mechanism alignment** 推进一步。

注意：由于当前文本缓存接口仍然提供的是 **view-level sentence embedding**，这一版实现的是：

- **signal-side token-level alignment**
- **text-side multi-view token guidance**

也就是让 **signal tokens** 与 `evidence/mechanism` 两个文本视图建立更细粒度的对应关系；它是“往 token-level 推进的 v1”，还不是最终的 full text-token alignment。

## 新增文件（仓库相对路径）

- `ele_servo_paper_repro_bundle/experiments_smoke_20260316/token_evidence_alignment_v1_utils.py`
- `ele_servo_paper_repro_bundle/experiments_smoke_20260316/exp1_stage4_token_evidence_alignment_v1_models.py`
- `ele_servo_paper_repro_bundle/experiments_smoke_20260316/train_exp1_stage4_token_evidence_alignment_v1.py`
- `ele_servo_paper_repro_bundle/experiments_smoke_20260316/run_exp1_stage4_token_evidence_alignment_v1.sh`

## 核心改动

1. 继续保留 Round 32 的：
   - text-guided fault gate
   - TopK/listwise
   - soft prototype dual-level
   - confusion-aware hard negative

2. 新增：
   - `token_transport_plan(signal_tokens, evidence, mechanism)`
   - `token_primary_support_loss(...)`
   - 基于 token plan 的模态质量统计导出
   - `test_token_alignment_scores.csv`
   - `test_token_alignment_summary.json`

## 建议初始化

用已经较强的 Stage 3 作为初始化：

```bash
STAGE3_INIT=ele_servo_paper_repro_bundle/experiments_smoke_20260316/models/exp1_topk_duallevel_align_v2/stage3_topk_duallevel/best.pt
```

## 应用方式

把补丁放到仓库根目录后执行：

```bash
git apply ele_test_round39_stage4_token_alignment_v1.patch
```

## 运行方式

```bash
bash ele_servo_paper_repro_bundle/experiments_smoke_20260316/run_exp1_stage4_token_evidence_alignment_v1.sh
```

## 关键输出

- `report.json`
- `test_token_alignment_scores.csv`
- `test_token_alignment_summary.json`
- `test_topk_consistency_summary.json`

## 重点关注的故障类

- `friction_wear_mild`
- `friction_wear_severe`
- `current_sensor_bias`
- `partial_demagnetization`

## 静态校验结果

- Python 文件已通过 `py_compile`
- Shell runner 已通过 `bash -n`

## 本地工程文件导出位置

以下是可直接查看的单文件版本：

- `token_evidence_alignment_v1_utils.py`
- `exp1_stage4_token_evidence_alignment_v1_models.py`
- `train_exp1_stage4_token_evidence_alignment_v1.py`
- `run_exp1_stage4_token_evidence_alignment_v1.sh`
