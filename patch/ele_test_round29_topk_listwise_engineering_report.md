# Round 29 工程交付说明

## 1. 交付目标

基于第28轮结论，本轮工程文件优先落地三件事：

1. **TopK / Listwise 一致性训练目标**
   - 不再只优化 top1 主模态
   - 改为优化 `primary > support > others`
   - 更贴近当前 `top2/top3` 评估口径

2. **混淆感知 hard negative**
   - 针对 friction / sensor bias / demagnetization 等困难类
   - 使用定向负样本，而不是随机 negative

3. **轻量 transport / OT 风格约束**
   - 在“模态向量 × 视图向量”上构造稀疏 transport plan
   - 让 evidence / mechanism / contrast 三种视图的模态分配更有结构性

## 2. 新增文件

补丁会新增以下文件到仓库中的固定位置：

- `ele_servo_paper_repro_bundle/experiments_smoke_20260316/modality_evidence_topk_v3.py`
- `ele_servo_paper_repro_bundle/experiments_smoke_20260316/train_exp1_topk_listwise_token_alignment_v1.py`
- `ele_servo_paper_repro_bundle/experiments_smoke_20260316/run_exp1_topk_listwise_token_alignment_v1.sh`

## 3. 文件职责

### 3.1 `modality_evidence_topk_v3.py`

职责：

- 修订 `PRIMARY_MODALITY_BY_SCENARIO`
- 修订 `SUPPORT_MODALITIES_BY_SCENARIO`
- 提供 `CONFUSION_AWARE_HARD_NEGATIVES`
- 提供 `top2/top3/weighted` 一致性工具函数
- 提供 `normal` 的 entropy 评估辅助函数

### 3.2 `train_exp1_topk_listwise_token_alignment_v1.py`

职责：

- 在 Stage 3 / Stage 4 上直接复用现有 backbone
- 新增四视图文本缓存：
  - combined
  - evidence
  - mechanism
  - contrast
- 新增损失：
  - combined / evidence / mechanism alignment
  - contrast repulsion
  - listwise alignment loss
  - ranking margin loss
  - prototype positive loss
  - confusion-aware hard negative loss
  - transport alignment loss
- 导出：
  - `test_modality_evidence_scores.csv`
  - `test_modality_evidence_summary.json`
  - `report.json`

### 3.3 `run_exp1_topk_listwise_token_alignment_v1.sh`

职责：

- 从已有 Stage 2 checkpoint 启动
- 先跑 Stage 3
- 再跑 Stage 4
- 输出到：
  - `experiments_smoke_20260316/models/exp1_topk_listwise_token_alignment_v1/`

## 4. 部署方式

在仓库根目录执行：

```bash
git apply ele_test_round29_topk_listwise_engineering.patch
```

然后运行：

```bash
bash ele_servo_paper_repro_bundle/experiments_smoke_20260316/run_exp1_topk_listwise_token_alignment_v1.sh
```

## 5. 部署位置校验

应用补丁后，建议先做三个最小检查。

### 5.1 文件存在性检查

```bash
ls ele_servo_paper_repro_bundle/experiments_smoke_20260316/modality_evidence_topk_v3.py
ls ele_servo_paper_repro_bundle/experiments_smoke_20260316/train_exp1_topk_listwise_token_alignment_v1.py
ls ele_servo_paper_repro_bundle/experiments_smoke_20260316/run_exp1_topk_listwise_token_alignment_v1.sh
```

### 5.2 Python 语法检查

```bash
python -m py_compile   ele_servo_paper_repro_bundle/experiments_smoke_20260316/modality_evidence_topk_v3.py   ele_servo_paper_repro_bundle/experiments_smoke_20260316/train_exp1_topk_listwise_token_alignment_v1.py
```

### 5.3 Shell 语法检查

```bash
bash -n ele_servo_paper_repro_bundle/experiments_smoke_20260316/run_exp1_topk_listwise_token_alignment_v1.sh
```

## 6. 本地已完成校验

本次交付在生成阶段已经完成以下静态校验：

- `modality_evidence_topk_v3.py` 通过 `py_compile`
- `train_exp1_topk_listwise_token_alignment_v1.py` 通过 `py_compile`
- `run_exp1_topk_listwise_token_alignment_v1.sh` 通过 `bash -n`
- 补丁通过 `git apply --check`

## 7. 仍需你本地完成的校验

本次没有完成训练级验证，因此仍建议你本地补两类检查：

1. **运行级检查**
   - Stage 3 能否正常加载 Stage 2 checkpoint
   - Stage 4 能否正常读取 Stage 3 输出

2. **结果级检查**
   - `friction_wear_mild / severe`
   - `current_sensor_bias`
   - `partial_demagnetization`

重点看这些类是否改善：

- `evidence_primary_in_top2`
- `evidence_primary_in_top3`
- `evidence_primary_or_support_at3`
- `mechanism_primary_or_support_at3`
- `weighted_consistency_at3`

## 8. 预期实验目录

```text
ele_servo_paper_repro_bundle/experiments_smoke_20260316/models/exp1_topk_listwise_token_alignment_v1/
├── stage3_topk_listwise/
│   ├── best.pt
│   ├── report.json
│   ├── test_modality_evidence_scores.csv
│   └── test_modality_evidence_summary.json
└── stage4_topk_listwise/
    ├── best.pt
    ├── report.json
    ├── test_modality_evidence_scores.csv
    └── test_modality_evidence_summary.json
```

## 9. 本轮工程定位

这轮不是替换你现有主线，而是：

- 在 **Round 23/27 结果之上**
- 新增一条 **TopK / Listwise / Confusion-Aware / Transport** 的实验分支

它最适合回答的问题是：

- 现有模型是不是已经“会了”，只是排序不够准
- evidence / mechanism 的一致性是否能通过排序目标进一步拉高
- friction / current_sensor_bias / partial_demagnetization 这些类，问题主要在先验、排序还是 hard negative
