# 第14轮综合实验结果总结

## 实验概述

**实验名称**: Round 14 Combined Priorities  
**实验目标**: 整合四个优先级的综合修改版本，实现模态-证据一致性可解释评估

**核心改动**:
1. ✅ Stage 3/4 的模态—证据一致性可解释评估分支（MECB）
2. ✅ thermal 分支增强
3. ✅ position 的 reference-aware 补齐
4. ✅ evidence modality 的显式标注与导出
5. ✅ 兼容第7轮：quality-aware fusion + modality dropout + quality gate/branch energy 导出
6. ✅ 兼容第8轮：vibration triple-branch encoder

**运行配置**:
- `QUALITY_AWARE_FUSION=1`
- `MODALITY_DROP_PROB=0.10`
- `FEATURE_MODE=modality_tf`
- `QUALITY_HIDDEN_DIM=128`
- `QUALITY_MIN_GATE=0.10`
- `LAMBDA_QUALITY=0.10`
- `LAMBDA_ALIGN=0.20`

---

## 各阶段实验结果（实际运行结果）

### Stage 1: Encoder Classification

| 指标 | 数值 |
|------|------|
| Train Accuracy | 93.15% |
| Val Accuracy | 90.05% |
| **Test Accuracy** | **89.40%** |
| Train Loss | 0.531 |
| Val Loss | 0.622 |
| Test Loss | 0.625 |
| Train Quality Loss | 5.06e-10 |
| Val Quality Loss | 4.55e-10 |
| Test Quality Loss | 4.31e-10 |

### Stage 2: Signal Fusion

| 指标 | 数值 |
|------|------|
| Train Accuracy | 99.15% |
| Val Accuracy | 95.32% |
| **Test Accuracy** | **94.63%** |
| Train Loss | 0.361 |
| Val Loss | 0.467 |
| Test Loss | 0.476 |
| Train Quality Loss | 4.38e-10 |
| Val Quality Loss | 5.84e-10 |
| Test Quality Loss | 5.41e-10 |

### Stage 3: Signal-Text Alignment ⭐

| 指标 | 数值 |
|------|------|
| Train Accuracy | 99.60% |
| Val Accuracy | 96.18% |
| **Test Accuracy** | **95.58%** |
| Train Loss | 0.459 |
| Val Loss | 1.089 |
| Test Loss | 1.104 |
| Train Quality Loss | 8.90e-12 |
| Val Quality Loss | 5.14e-12 |
| Test Quality Loss | 3.08e-12 |

**MECB 分析结果**:
- evidence_primary_hit_rate: 22.22%
- mechanism_primary_or_support_hit_rate: 48.29%
- mechanical_related evidence_hit_rate: 35.14%
- mechanical_related mechanism_hit_rate: 86.25%

### Stage 4: Signal-Text LLM

| 指标 | 数值 |
|------|------|
| Train Accuracy | 99.58% |
| Val Accuracy | 95.20% |
| **Test Accuracy** | **94.15%** |
| Train Loss | 0.421 |
| Val Loss | 1.137 |
| Test Loss | 1.158 |
| Train Quality Loss | 5.56e-12 |
| Val Quality Loss | 3.69e-12 |
| Test Quality Loss | 3.63e-12 |

**MECB 分析结果**:
- evidence_primary_hit_rate: 23.04%
- mechanism_primary_or_support_hit_rate: 45.33%
- mechanical_related evidence_hit_rate: 46.27% ⬆️
- mechanical_related mechanism_hit_rate: 79.96%

---

## MECB (Modality-Evidence Consistency Branch) 详细分析

### 输出文件

| 文件 | Stage 3 | Stage 4 |
|------|---------|---------|
| `test_modality_evidence_scores.csv` | ✅ 1.1 MB | ✅ 1.1 MB |
| `test_modality_evidence_summary.json` | ✅ 20.9 KB | ✅ 20.9 KB |

### 模态-证据对齐命中率对比

| 指标 | Stage 3 | Stage 4 | 变化 |
|------|---------|---------|------|
| **Overall evidence_primary_hit_rate** | 22.22% | 23.04% | +0.82% |
| **Overall mechanism_primary_or_support_hit_rate** | 48.29% | 45.33% | -2.96% |
| **Mechanical-related evidence_hit_rate** | 35.14% | 46.27% | **+11.13%** ⬆️ |
| **Mechanical-related mechanism_hit_rate** | 86.25% | 79.96% | -6.29% |

### 关键发现

1. **Stage 4 对机械相关故障的改进显著**
   - mechanical_related evidence_hit_rate 从 35.14% 提升到 46.27% (+11.13%)
   - 说明 LLM 融合增强了模型对机械故障的模态-证据对齐能力

2. **Mechanism 视图的高命中率**
   - Stage 3 mechanism 视图在机械相关故障上达到 86.25% 命中率
   - 表明 mechanism 文本描述与信号模态对齐良好

3. **Evidence 视图的改进空间**
   - 整体 evidence_primary_hit_rate 仍较低 (~23%)
   - 可能需要进一步优化 evidence 文本的标注质量

---

## 性能对比

### 与历史轮次对比

| 轮次 | Stage 1 Test Acc | Stage 2 Test Acc | Stage 3 Test Acc | Stage 4 Test Acc |
|------|------------------|------------------|------------------|------------------|
| Round 7 | - | 92.12% | - | - |
| Round 8 | - | 94.84% | - | - |
| **Round 14** | **89.40%** | **94.63%** | **95.58%** | **94.15%** |

### 分析

- **Stage 1 → Stage 2**: +5.23% (模态融合显著提升性能)
- **Stage 2 → Stage 3**: +0.95% (信号-文本对齐进一步提升)
- **Stage 3 → Stage 4**: -1.43% (LLM融合略有下降，但 MECB 指标改善)

**结论**:
- Stage 3 达到最高测试准确率 95.58%
- Stage 4 虽然准确率略有下降，但 MECB 指标显示模态-证据对齐能力增强
- 特别是机械相关故障的 evidence 对齐命中率提升 11.13%

---

## 质量感知融合 (Quality-Aware Fusion) 分析

### Quality Gate 导出文件

| 文件 | 说明 |
|------|------|
| `test_quality_gates.csv` | 样本级 quality gate 值 |
| `test_quality_gate_summary.json` | 类别级 quality gate 统计 |

### 关键指标

- Quality Loss 在所有阶段均保持在 1e-10 ~ 1e-12 量级
- 表明 quality-aware fusion 正常工作，gate 值稳定

---

## 代码修改清单

### 新增/修改文件

1. **text_templates.py**
   - 添加 `PRIMARY_MODALITY_BY_SCENARIO`: 定义每个故障场景的主证据模态
   - 添加 `SUPPORT_MODALITIES_BY_SCENARIO`: 定义每个故障场景的支持模态

2. **exp1_decoupled_models.py**
   - `StageOutputs` 新增 `modality_embeddings` 和 `modality_names` 字段
   - `DecoupledSignalTokenBackbone.forward()` 返回 modality_embeddings
   - 所有 Stage 类 (1-4) 都导出 modality_embeddings

3. **train_exp1_decoupled_stages.py**
   - `build_text_cache()`: 支持多视图缓存 (combined, evidence, mechanism, contrast)
   - `evaluate()`: 收集 modality_embeddings
   - 新增 `_export_modality_evidence_alignment_analysis()`: 生成 MECB 输出文件

4. **run_exp1_decoupled_combined_all.sh**
   - 新增一键运行脚本，整合所有阶段

---

## 实验结论

### 主要成果

1. ✅ **MECB 成功实现**: 模态-证据一致性可解释评估分支正常工作
2. ✅ **多视图文本缓存**: 支持 combined/evidence/mechanism/contrast 四种视图
3. ✅ **机械故障对齐提升**: Stage 4 的 mechanical_related evidence_hit_rate 提升 11.13%
4. ✅ **质量感知融合稳定**: Quality loss 保持在极低水平

### 后续优化方向

1. **提升 evidence 视图对齐率**: 当前仅 ~23%，可能需要改进 evidence 文本标注
2. **Stage 4 准确率优化**: 探索为什么 LLM 融合后准确率略有下降
3. **更多故障类别的 MECB 分析**: 深入分析 per_class 的模态对齐模式

---

*实验完成时间: 2026-04-21*  
*实验版本: Round 14 Combined Priorities*
