# ele_servo_new

## 项目路径

```
/mnt/PRO6000_disk/swd/ele_servo_new/
```

## 目录结构

```
ele_servo_new/
├── .gitignore                                    # Git 忽略配置
├── build_paper_repro_bundle.sh                   # 构建论文复现包脚本
├── run_all_paper_pipeline.sh                     # 总控流水线脚本
├── paper_repro_bundle_README.md                  # 论文复现包说明
├── paper_repro_bundle_manifest.txt               # 论文复现包清单
├── paper_repro_bundle_requirements.txt           # 论文复现包依赖
│
├── ele_servo_paper_repro_bundle/                 # 论文复现包
│   ├── src/                                      # 数据集构建与流水线脚本
│   ├── servo_diagnostic/                         # 伺服诊断模块
│   ├── servo_llm_alignment/                      # 伺服-LLM 对齐模块
│   ├── experiments_smoke_20260316/               # 实验脚本
│   ├── 论文/                                     # 论文正文与图表
│   ├── 项目拓扑图/                               # 项目拓扑图
│   ├── model/                                    # 模型定义
│   ├── handoff_packages/                         # 交接包
│   └── *.png / *.md                              # 说明文档与图示
│
├── signal_prefix_qwen_reasoning_rag_20260402/    # RAG 推理项目
│   ├── code/                                     # 源码
│   │   ├── servo_diagnostic/                     # 伺服诊断模块
│   │   ├── servo_llm_alignment/                  # 伺服-LLM 对齐模块
│   │   ├── experiments_smoke_20260316/           # 实验脚本
│   │   └── src/                                  # 数据集构建脚本
│   └── docs/                                     # 文档
│       └── README.md
│
└── count_files.sh                                # 目录统计脚本
```

## 已忽略的目录与文件

- `raw_data/`, `derived_data/`, `derived_datasets/`, `diagnostic_datasets/`
- `models/`, `checkpoints/`, `results/`
- `comparison_plots/`, `traditional/`
- `*.npz`, `*.npy`, `*.pkl`, `*.pt`, `*.pth`, `*.jsonl`
- `logs/`, `*.log`, `caches/`, `venv/`

## 更新记录

### Round 32 (2025-04-22) - TopK Dual-Level Alignment v2

**突破性成果**：实现 100% 模态-证据对齐

#### 新增文件
- `ele_servo_paper_repro_bundle/experiments_smoke_20260316/modality_evidence_topk_v5.py` - TopK 排序 + Soft Prototype + Hard Negatives
- `ele_servo_paper_repro_bundle/experiments_smoke_20260316/exp1_topk_duallevel_align_v2_models.py` - Text-Guided Fault Gate 模型
- `ele_servo_paper_repro_bundle/experiments_smoke_20260316/train_exp1_topk_duallevel_align_v2.py` - 多视图对齐训练脚本
- `ele_servo_paper_repro_bundle/experiments_smoke_20260316/run_exp1_topk_duallevel_align_v2.sh` - Stage 3/4 运行脚本

#### 核心技术创新（借鉴大模型对齐）
| 技术 | 来源 | 作用 |
|------|------|------|
| Listwise Ranking Loss | RankNet/ListMLE | 强制 `primary > support > others` 排序 |
| Soft Prototype Bank | MoCo/SimCLR | 动量更新类原型，实现类级对齐 |
| Confusion-aware Hard Negatives | SimCLR/CLIP | 针对困难类定向挖掘负样本 |
| InfoNCE Contrastive Loss | CPC/CLIP | 正负样本对比，拉大决策边界 |
| Text-Guided Fault Gate | Cross-Attention | 文本语义引导动态模态融合 |

#### 实验结果
| 指标 | Stage 3 | Stage 4 |
|------|---------|---------|
| 测试准确率 | 99.31% | **100%** |
| Evidence Primary @ Top2 | - | **100%** |
| Evidence Primary @ Top3 | - | **100%** |
| Primary/Support @ Top3 | - | **100%** |

#### 困难类突破
- `friction_wear_mild/severe`: 从 <10% → **100%** 对齐
- `current_sensor_bias`: 从 76.92% → **100%** 对齐
- `partial_demagnetization`: 从 78.02% → **100%** 对齐

### Round 29 (2025-04-22) - TopK Listwise Token Alignment
- 实现 100% 测试准确率
- friction_wear 困难类重大突破 (1.85% → 96.91%)

### Round 23 (2025-04-20) - Evidence Mechanism Engineering
- 引入多视图对齐 (evidence/mechanism/contrast)
- Quality-aware Fusion + Modality Dropout

## 远程仓库

- **URL**: `https://github.com/zhuyingqingchun/ele_test.git`
- **分支**: `main`
