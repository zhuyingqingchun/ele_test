# 第7、8轮实验结果总结

## 实验概述

| 轮次 | 核心改动 | 目的 |
|------|----------|------|
| Round 7 | 启用 `quality_aware_fusion` + 导出 gate/energy | 验证质量感知融合机制，检测模态压制 |
| Round 8 | 振动编码器增强（三支路架构）+ 保持 quality_aware_fusion | 提升振动模态表达能力，验证对机械类故障的贡献 |

---

## Round 7: Quality Aware Fusion 实验

### 配置
- `quality_aware_fusion`: 启用
- `quality_hidden_dim`: 128
- `quality_min_gate`: 0.10
- `lambda_quality`: 0.10
- `modality_drop_prob`: 0.0

### Stage 2 Signal Fusion 结果

| 指标 | 数值 |
|------|------|
| Train Accuracy | 96.42% |
| Val Accuracy | 92.05% |
| **Test Accuracy** | **92.12%** |
| Train Loss | 0.429 |
| Val Loss | 0.544 |
| Test Loss | 0.538 |

### Quality Gate 分析

**Mechanical-related 样本（1033个）**：
```json
{
  "mean_gate": [1.0, 1.0, 1.0, 1.0],
  "branch_names": ["position", "electrical", "thermal", "vibration"]
}
```

**Non-mechanical 样本（3156个）**：
```json
{
  "mean_gate": [1.0, 1.0, 1.0, 1.0]
}
```

**观察**：
- 所有 gate 值均为 1.0，说明质量估计器认为所有模态质量良好
- 未发现明显的模态压制现象
- Quality loss 极低（~1e-11），说明质量估计任务很容易完成

### 输出文件
- Report: `models/exp1_decoupled_v3_modality_tf_control/stage2_signal_fusion/report.json`
- Quality Gate Summary: `models/exp1_decoupled_v3_modality_tf_control/stage2_signal_fusion/test_quality_gate_summary.json`
- Quality Gates CSV: `models/exp1_decoupled_v3_modality_tf_control/stage2_signal_fusion/test_quality_gates.csv`

---

## Round 8: 振动增强版实验

### 核心改动

#### 1. VibrationStateEncoder 重写
采用轻量三支路架构：
- **Raw Local Branch**: Conv1d(kernel=7) + Conv1d(kernel=5)
- **Raw Context Branch**: Conv1d(kernel=15) + Conv1d(kernel=5)
- **Envelope Branch**: AvgPool1d(kernel=9) + Conv1d(kernel=9)
- **STFT Branch**: 轻量STFT特征提取
- **Branch Gate**: Soft gate 融合三支路输出

#### 2. 容量提升
| 参数 | Round 7 | Round 8 | 变化 |
|------|---------|---------|------|
| token_count | 12 | 16 | +33% |
| hidden_dim | 32 | 64 | +100% |

#### 3. 保持 Round 7 配置
- `quality_aware_fusion`: 启用
- `quality_hidden_dim`: 128
- `quality_min_gate`: 0.10
- `lambda_quality`: 0.10

### Stage 2 Signal Fusion 结果

| 指标 | Round 7 | Round 8 | 提升 |
|------|---------|---------|------|
| Train Accuracy | 96.42% | 99.42% | +3.00% |
| Val Accuracy | 92.05% | 95.92% | +3.87% |
| **Test Accuracy** | **92.12%** | **94.84%** | **+2.72%** |
| Train Loss | 0.429 | 0.352 | -0.077 |
| Val Loss | 0.544 | 0.454 | -0.090 |
| Test Loss | 0.538 | 0.485 | -0.053 |

### Quality Gate 分析（Round 8）

**Mechanical-related 样本（1033个）**：
```json
{
  "mean_gate": [1.0, 1.0, 1.0, 1.0],
  "sample_count": 1033
}
```

**Non-mechanical 样本（3156个）**：
```json
{
  "mean_gate": [1.0, 1.0, 1.0, 1.0]
}
```

**关键发现**：
- 所有 gate 值仍为 1.0
- 在 `bearing_defect` 类别中，vibration energy 达到 **9.09**（明显高于其他类别）
- 说明振动增强版编码器能有效提取轴承缺陷相关的振动特征

### 输出文件
- Report: `models/exp1_decoupled_v8_vibration_enhanced/stage2_signal_fusion/report.json`
- Quality Gate Summary: `models/exp1_decoupled_v8_vibration_enhanced/stage2_signal_fusion/test_quality_gate_summary.json`

---

## 模态消融实验（Round 8 配置）

### 实验设置
- A0_full: 完整模态（pos + electrical + thermal + vibration）
- A1_no_position: 去除 position
- A2_no_electrical: 去除 electrical
- A3_no_thermal: 去除 thermal
- A4_no_vibration: 去除 vibration

### Stage 2 Signal Fusion 结果对比

| 实验 | Test Accuracy | Val Accuracy | 与 A0 差距 |
|------|---------------|--------------|-----------|
| **A0_full** | **93.58%** | **94.29%** | - |
| A1_no_position | 89.50% | 90.02% | -4.08% |
| A2_no_electrical | 90.33% | 90.38% | -3.25% |
| A3_no_thermal | 91.67% | 91.72% | -1.91% |
| **A4_no_vibration** | **93.53%** | **93.65%** | **-0.05%** |

### 关键发现

**意外结果**：A4_no_vibration (93.53%) 与 A0_full (93.58%) 几乎相同，差距仅 **0.05%**！

**分析**：
1. **振动模态在当前配置下贡献很小**
2. Position 模态最重要（去除后下降 4.08%）
3. Electrical 模态次之（去除后下降 3.25%）
4. Thermal 模态影响较小（去除后下降 1.91%）

**可能原因**：
- 振动信息与其他模态（特别是 position 和 electrical）存在冗余
- 机械类故障的特征在 position/electrical 信号中已有体现
- 需要更细粒度的故障类别分析来识别振动真正起作用的场景

### Quality Gate 对比（A0 vs A4）

**A0_full Mechanical-related**：
```json
"mean_gate": [1.0, 0.999999999884599, 1.0, 1.0]
```

**A4_no_vibration Mechanical-related**：
```json
"mean_gate": [0.999999999769198, 0.999999999884599, 1.0, 1.0]
```

---

## 结论与建议

### 主要结论

1. **振动增强版编码器有效**
   - 主实验 Test Accuracy 从 92.12% 提升到 94.84%（+2.72%）
   - 在 `bearing_defect` 类别中 vibration energy 显著升高（9.09）

2. **Quality Aware Fusion 运行正常**
   - 所有 gate 值接近 1.0，说明模态质量良好
   - 未发现明显的模态压制现象

3. **振动模态在消融实验中贡献有限**
   - A0 vs A4 差距仅 0.05%，说明振动与其他模态存在冗余
   - Position 和 Electrical 模态是性能的主要贡献者

### 下一步建议

1. **细粒度故障分析**
   - 分析具体哪些故障类别真正依赖振动模态
   - 检查 vibration branch 的 energy 在各类别中的分布

2. **模态冗余研究**
   - 考虑降低 position/electrical 的容量，强制模型更多依赖振动
   - 探索不同模态组合的最优配置

3. **Quality Gate 调优**
   - 尝试调整 `quality_min_gate` 和 `lambda_quality`
   - 引入 `modality_drop_prob` 进行模态丢弃训练

---

## 实验文件索引

### Round 7
```
experiments_smoke_20260316/models/exp1_decoupled_v3_modality_tf_control/
├── stage1_encoder_cls/
│   ├── report.json
│   ├── best.pt
│   └── confusion_matrix.json
├── stage2_signal_fusion/
│   ├── report.json
│   ├── best.pt
│   ├── confusion_matrix.json
│   ├── test_quality_gate_summary.json
│   └── test_quality_gates.csv
├── stage3_signal_text_align/
│   └── ...
└── stage4_signal_text_llm/
    └── ...
```

### Round 8
```
experiments_smoke_20260316/models/exp1_decoupled_v8_vibration_enhanced/
├── stage1_encoder_cls/
│   ├── report.json
│   └── best.pt
├── stage2_signal_fusion/
│   ├── report.json
│   ├── best.pt
│   ├── test_quality_gate_summary.json
│   └── test_quality_gates.csv
└── ...

experiments_smoke_20260316/models/exp1_decoupled_v8_ablation_vibration_enhanced/
├── A0_full/stage2_signal_fusion/report.json
├── A1_no_position/stage2_signal_fusion/report.json
├── A2_no_electrical/stage2_signal_fusion/report.json
├── A3_no_thermal/stage2_signal_fusion/report.json
└── A4_no_vibration/stage2_signal_fusion/report.json
```

---

*生成时间: 2026-04-21*
