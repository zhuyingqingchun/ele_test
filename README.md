# ele_servo_new

## 项目路径

```
/mnt/PRO6000_disk/swd/ele_servo_new/
```

## 目录结构

```
ele_servo_new/
├── .gitignore                                    # Git 忽略配置
├── README.md                                     # 项目说明（本文件）
├── build_paper_repro_bundle.sh                   # 构建论文复现包脚本
├── run_all_paper_pipeline.sh                     # 总控流水线脚本
├── paper_repro_bundle_README.md                  # 论文复现包说明
├── paper_repro_bundle_manifest.txt               # 论文复现包清单
├── paper_repro_bundle_requirements.txt           # 论文复现包依赖
├── count_files.sh                                # 目录统计脚本
│
├── ele_servo_paper_repro_bundle/                 # 论文复现包
│   ├── src/                                      # 数据集构建与流水线脚本
│   │   ├── run_servo_pipeline_expanded.py        # 伺服仿真生成 CSV
│   │   ├── build_servo_handoff_dataset.py        # 构建多模态数据集
│   │   ├── build_stage1_alignment_corpus.py      # 构建对齐语料
│   │   └── generate_paper_fault_mechanism_figure.py
│   │
│   ├── servo_diagnostic/                         # 伺服诊断模块
│   │   ├── config.py                             # 配置定义
│   │   ├── plant.py                              # 被控对象模型
│   │   ├── controllers.py                        # 控制器实现
│   │   ├── controller_tuning.py                  # 控制器调参
│   │   ├── simulator.py                          # 仿真引擎
│   │   ├── feature_engineering.py                # 特征工程
│   │   ├── scenarios.py                          # 故障场景定义
│   │   ├── operating_conditions.py               # 工况定义
│   │   ├── hierarchical_diagnosis.py             # 分层诊断
│   │   ├── pipeline.py                           # 诊断流水线
│   │   ├── multimodal_method.py                  # 多模态方法
│   │   ├── multimodal_method_variants.py         # 多模态变体
│   │   ├── baseline_diagnosis.py                 # 基线诊断
│   │   ├── cnn_baseline.py                       # CNN 基线
│   │   ├── tree_baseline.py                      # 树模型基线
│   │   ├── benchmark_runner.py                   # 基准测试运行器
│   │   ├── benchmark_utils.py                    # 基准测试工具
│   │   ├── validation.py                         # 验证工具
│   │   ├── plots.py                              # 绘图工具
│   │   └── io_utils.py                           # I/O 工具
│   │
│   ├── servo_llm_alignment/                      # 伺服-LLM 对齐模块
│   │   ├── config.py                             # 配置定义
│   │   ├── corpus.py                             # 语料构建
│   │   ├── dataset.py                            # 数据集定义
│   │   ├── diagnostic.py                         # 诊断逻辑
│   │   ├── events.py                             # 事件定义
│   │   ├── losses.py                             # 损失函数
│   │   ├── prototypes.py                         # 原型网络
│   │   ├── qwen_reasoner.py                      # Qwen 推理器
│   │   ├── retriever.py                          # 检索器
│   │   ├── runtime.py                            # 运行时
│   │   ├── signal_encoder.py                     # 信号编码器
│   │   ├── text_encoder.py                       # 文本编码器
│   │   ├── text_templates.py                     # 文本模板
│   │   ├── template_config.py                    # 模板配置
│   │   ├── context_builder.py                    # 上下文构建器
│   │   └── trainer.py                            # 训练器
│   │
│   ├── experiments_smoke_20260316/               # 实验脚本
│   │   ├── exp1_decoupled_models.py              # 四阶段解耦模型定义
│   │   ├── train_exp1_decoupled_stages.py        # 四阶段训练入口
│   │   ├── prepare_traditional_window_dataset.py # 传统模型数据集准备
│   │   │
│   │   ├── run_exp1_decoupled_modality_tf_control_raw_thermal_full.sh  # 主四模态实验
│   │   ├── run_exp1_decoupled_single_modality.sh                       # 单模态对比
│   │   ├── run_exp1_decoupled_single_modality_async.sh                 # 单模态异步版
│   │   ├── run_exp1_decoupled_modality_ablation.sh                     # 模态消融实验
│   │   ├── run_exp1_decoupled_modality_ablation_async.sh               # 模态消融异步版
│   │   │
│   │   ├── run_compare_all_current_modalities.sh                       # 传统模型对比总入口
│   │   ├── run_compare_cnn_tcn_current_modalities.sh                   # CNN-TCN 基线
│   │   ├── run_compare_bilstm_current_modalities.sh                    # BiLSTM 基线
│   │   ├── run_compare_resnet_fcn_current_modalities.sh                # ResNet-FCN 基线
│   │   ├── run_compare_transformer_encoder_current_modalities.sh       # Transformer Encoder 基线
│   │   ├── run_compare_dual_branch_xattn_current_modalities.sh         # Dual-Branch XAttn 基线
│   │   │
│   │   ├── run_ablation_single.sh                  # 消融实验单配置脚本（并行用）
│   │   ├── run_ablation_stage12.sh                 # 消融实验 Stage1+2 顺序脚本
│   │   ├── eval_representative_modality_sensitivity.py  # 代表性故障模态敏感性分析
│   │   └── run_eval_representative_modality_sensitivity.sh # 敏感性分析启动脚本
│   │
│   ├── derived_datasets/                         # 派生数据集（已忽略）
│   ├── diagnostic_datasets/                      # 诊断数据集（已忽略）
│   ├── experiments_smoke_20260316/models/        # 实验模型产物（已忽略）
│   │
│   ├── model/                                    # 模型定义
│   ├── handoff_packages/                         # 交接包
│   ├── 论文/                                     # 论文正文与图表
│   │   ├── paper.md                              # 论文正文
│   │   └── figures/                              # 论文图表
│   ├── 项目拓扑图/                               # 项目拓扑图
│   ├── servo_control_topology.png                # 伺服控制拓扑图
│   ├── servo_diagnostic_architecture.png         # 诊断架构图
│   ├── servo_diagnostic_overview.png             # 诊断概览图
│   ├── servo_jam_signals.png                     # 舵机卡滞信号图
│   ├── servo_signal_topology.png                 # 信号拓扑图
│   ├── servo_model_math_and_topology_zh.md       # 伺服数学模型与拓扑说明
│   └── paper_repro_bundle_manifest.txt           # 复现包清单
│
├── signal_prefix_qwen_reasoning_rag_20260402/    # RAG 推理项目
│   ├── code/                                     # 源码
│   │   ├── servo_diagnostic/                     # 伺服诊断模块（同 bundle）
│   │   ├── servo_llm_alignment/                  # 伺服-LLM 对齐模块（同 bundle）
│   │   ├── experiments_smoke_20260316/           # 实验脚本
│   │   │   ├── train_exp1_decoupled_stages.py    # 四阶段训练入口
│   │   │   ├── train_signal_prefix_qwen_reasoning.py  # Qwen 推理训练
│   │   │   ├── train_signal_prefix_qwen_lora.py  # Qwen LoRA 训练
│   │   │   └── run_signal_prefix_qwen_reasoning.sh
│   │   └── src/                                  # 数据集构建脚本
│   │       ├── build_servo_handoff_dataset.py
│   │       ├── build_servo_multimodal_dataset.py
│   │       ├── build_stage1_alignment_corpus.py
│   │       └── run_servo_pipeline_expanded.py
│   └── docs/                                     # 文档
│       └── README.md
│
├── patch/                                        # 补丁目录
│   └── representative_modality_sensitivity_bundle_fix.patch
│
└── ele_tree.txt                                  # 目录统计输出
```

## 实验体系

### 主实验（四模态）
- **入口**: `ele_servo_paper_repro_bundle/experiments_smoke_20260316/run_exp1_decoupled_modality_tf_control_raw_thermal_full.sh`
- **输入模态**: 位置、electrical、thermal、vibration
- **模型阶段**: Stage 1（信号分类）→ Stage 2（信号融合）→ Stage 3（文本对齐）→ Stage 4（信号-文本深度融合）

### 单模态对比实验
- **入口**: `run_exp1_decoupled_single_modality.sh`
- **目的**: 单独使用某一种模态时的诊断效果

### 模态消融实验
- **入口**: `run_exp1_decoupled_modality_ablation.sh`
- **目的**: 去掉某些模态后性能变化
- **配置**: A0_full（全模态）、A1_no_position、A2_no_electrical、A3_no_thermal、A4_no_vibration

### 传统模型对比实验
- **入口**: `run_compare_all_current_modalities.sh`
- **基线**: CNN-TCN、BiLSTM、ResNet-FCN、Transformer Encoder、Dual-Branch XAttn

### 代表性故障模态敏感性分析
- **入口**: `run_eval_representative_modality_sensitivity.sh`
- **故障**: bearing_defect、inverter_voltage_loss、thermal_saturation
- **输出**: summary.csv、per_sample.csv、report.json、report.md

### Qwen 结构化诊断扩展
- **入口**: `run_signal_prefix_qwen_reasoning_qwen25_1p5b_v6_raw_thermal_tuned.sh`
- **输出**: fault_result、diagnostic_basis、explanation

## 已忽略的目录与文件

- `raw_data/`, `derived_data/`, `derived_datasets/`, `diagnostic_datasets/`
- `models/`, `checkpoints/`, `results/`
- `comparison_plots/`, `traditional/`
- `*.npz`, `*.npy`, `*.pkl`, `*.pt`, `*.pth`, `*.jsonl`
- `logs/`, `*.log`, `caches/`, `venv/`

## 远程仓库

- **URL**: `https://github.com/zhuyingqingchun/ele_test.git`
- **分支**: `main`
