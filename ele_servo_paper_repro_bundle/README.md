# 电动舵机论文复现实验打包说明

本包只保留论文主线真正用到的代码、配置、关键报告和正文图文，不包含历史无关实验目录，不包含大模型本体权重。

## 内容范围
- 电动舵机建模与故障注入仿真代码
- 从仿真 CSV 到多模态数据集与语义语料的生成脚本
- 论文主实验脚本
- 单模态对比脚本
- 模态消融脚本
- 传统深度时序对比脚本
- Qwen2.5-1.5B 结构化诊断扩展脚本
- 论文正文、关键图和关键 report.json

## 外部依赖
- Python 3.10+
- PyTorch
- scikit-learn
- matplotlib
- pandas
- transformers
- peft
- 本包不带 Qwen 模型权重；如需运行结构化诊断扩展，请自行准备 Qwen2.5-1.5B-Instruct，并通过 `QWEN_PATH` 指向本地模型目录

## 推荐运行顺序
### 1. 运行电动舵机仿真，生成原始 CSV
```bash
python3 src/run_servo_pipeline_expanded.py --repeat-count 1 --final-time 4.0 --dt 0.001 --random-seed 7
```

### 2. 构建论文实验使用的数据集
```bash
python3 src/build_servo_handoff_dataset.py
python3 src/build_stage1_alignment_corpus.py
python3 experiments_smoke_20260316/prepare_traditional_window_dataset.py
```

### 3. 运行论文主实验
```bash
bash experiments_smoke_20260316/run_exp1_decoupled_modality_tf_control_raw_thermal_full.sh
```

### 4. 运行单模态对比实验
```bash
bash experiments_smoke_20260316/run_exp1_decoupled_single_modality.sh
```
或异步版本：
```bash
bash experiments_smoke_20260316/run_exp1_decoupled_single_modality_async.sh
bash experiments_smoke_20260316/check_exp1_decoupled_single_modality_async.sh
```

### 5. 运行模态消融实验
```bash
bash experiments_smoke_20260316/run_exp1_decoupled_modality_ablation.sh
```
或异步版本：
```bash
bash experiments_smoke_20260316/run_exp1_decoupled_modality_ablation_async.sh
bash experiments_smoke_20260316/check_exp1_decoupled_modality_ablation_async.sh
```

### 6. 运行传统模型对比实验
```bash
bash experiments_smoke_20260316/run_compare_all_current_modalities.sh
```
也可单独运行：
```bash
bash experiments_smoke_20260316/run_compare_cnn_tcn_current_modalities.sh
bash experiments_smoke_20260316/run_compare_bilstm_current_modalities.sh
bash experiments_smoke_20260316/run_compare_resnet_fcn_current_modalities.sh
bash experiments_smoke_20260316/run_compare_transformer_encoder_current_modalities.sh
bash experiments_smoke_20260316/run_compare_dual_branch_xattn_current_modalities.sh
```

### 7. 运行结构化诊断扩展实验
```bash
export QWEN_PATH=/path/to/Qwen2.5-1.5B-Instruct
bash experiments_smoke_20260316/run_signal_prefix_qwen_reasoning_qwen25_1p5b_v6_raw_thermal_tuned.sh
```

## 迁移注意事项
- 所有脚本都基于相对 `PROJECT_ROOT` 运行，拷到新目录后仍可直接执行
- 如果使用自己的 Python 环境，请在运行前设置 `PYTHON_PATH`，否则默认使用 `python3`
- 如果需要 GPU，请设置 `CUDA_VISIBLE_DEVICES`
