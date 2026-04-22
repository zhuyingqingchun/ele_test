# Round 27 工程文件说明：TopK 一致性评估与故障-模态先验修订

## 1. 本轮目标

基于第24轮、第25轮、第26轮的结论，本轮不再优先改动主模型结构，而是先补齐**评估口径与先验口径**，把“单一 top1 一致性”升级为更符合多模态耦合机理的：

- `primary_in_top2`
- `primary_in_top3`
- `primary_or_support_hit_at_2`
- `primary_or_support_hit_at_3`
- `weighted_consistency_at_3`
- `normal` 的 `attention_entropy`

同时，修订故障到模态的主/支撑先验，重点纠正：

- `friction_wear_mild / severe`
- `current_sensor_bias`
- `partial_demagnetization`

## 2. 新增工程文件

补丁会新增 3 个文件：

1. `ele_servo_paper_repro_bundle/experiments_smoke_20260316/modality_evidence_topk_v2.py`
   - 保存 Round 25 修订后的 `primary/support` 先验
   - 提供 topK 判断、加权一致性、normal 熵计算

2. `ele_servo_paper_repro_bundle/experiments_smoke_20260316/analyze_modality_evidence_topk.py`
   - 读取已有的 `test_modality_evidence_scores.csv`
   - 自动推断 `evidence / mechanism` 两种视图分数列
   - 输出新的 `top2 / top3 / weighted` 结果

3. `ele_servo_paper_repro_bundle/experiments_smoke_20260316/run_exp1_evidence_mechanism_topk_v2.sh`
   - 针对 Round 23 产物做一键重分析
   - 默认分析：
     - `stage3_fault_aware_multiview`
     - `stage4_fault_aware_multiview_llm`

## 3. 部署位置

这些文件都应部署到仓库根目录下的：

```text
./ele_servo_paper_repro_bundle/experiments_smoke_20260316/
```

也就是：

```text
<repo_root>/ele_servo_paper_repro_bundle/experiments_smoke_20260316/
```

## 4. 适用前提

这轮工程文件依赖你已经完成 Round 23，并且存在以下输入文件：

```text
ele_servo_paper_repro_bundle/experiments_smoke_20260316/models/exp1_evidence_mechanism_primary_alignment_v1/stage3_fault_aware_multiview/test_modality_evidence_scores.csv

ele_servo_paper_repro_bundle/experiments_smoke_20260316/models/exp1_evidence_mechanism_primary_alignment_v1/stage4_fault_aware_multiview_llm/test_modality_evidence_scores.csv
```

如果这两个 CSV 不存在，runner 会直接退出并报错。

## 5. 生成的输出文件

对每个 stage 目录，会新增：

- `test_modality_evidence_topk_scores_v2.csv`
- `test_modality_evidence_topk_summary_v2.json`

### `test_modality_evidence_topk_scores_v2.csv`

按样本导出：

- scenario
- primary_modalities
- support_modalities
- evidence_top1 / top2 / top3
- mechanism_top1 / top2 / top3
- evidence/mechanism 的 top2/top3 命中结果
- weighted consistency
- attention entropy

### `test_modality_evidence_topk_summary_v2.json`

导出：

- 总体 `evidence / mechanism` 视图 topK 指标
- `normal` 的 attention entropy
- fault-only per-class topK 指标

## 6. 如何应用补丁

在仓库根目录执行：

```bash
git apply ele_test_round27_topk_consistency_engineering.patch
```

## 7. 如何运行

在仓库根目录执行：

```bash
bash ele_servo_paper_repro_bundle/experiments_smoke_20260316/run_exp1_evidence_mechanism_topk_v2.sh
```

如需指定 Round 23 目录，可覆盖：

```bash
ROUND23_ROOT=/your/path/to/exp1_evidence_mechanism_primary_alignment_v1 \
  bash ele_servo_paper_repro_bundle/experiments_smoke_20260316/run_exp1_evidence_mechanism_topk_v2.sh
```

## 8. 本地校验结果

本轮文件生成后，已完成以下静态校验：

1. `modality_evidence_topk_v2.py` 通过 `py_compile`
2. `analyze_modality_evidence_topk.py` 通过 `py_compile`
3. `run_exp1_evidence_mechanism_topk_v2.sh` 通过 `bash -n`
4. 补丁通过 `git apply --check`

这意味着：

- 文件路径正确
- 语法层面正确
- 补丁可在仓库根目录按相对路径应用

## 9. 本轮工程价值

这轮不是重训主模型，而是把你现有 Round 23 结果升级成更符合论文叙事的评估口径：

- 从 `top1` 升级到 `top2/top3`
- 从“唯一主模态”升级到“主模态 + 支撑模态”
- 把 `normal` 从 fault consistency 里剥离，单独用 entropy 分析

因此，这轮最适合用来：

- 修正 Round 23 的一致性表述
- 给论文实验部分生成更合理的新表格
- 判断当前问题究竟在模型，还是在先验与指标设计

## 10. 下一步建议

跑完这轮后，优先看三个数字：

1. `evidence.primary_in_top2`
2. `mechanism.primary_or_support_hit_at_2`
3. `normal.attention_entropy`

如果 top2/top3 指标明显比 top1 更好，而难类如 `friction_wear_mild/severe` 仍然很差，那下一步更可能要继续改：

- class prior
- evidence text
- prototype / hard negative 配置

而不是立刻再改 backbone。
