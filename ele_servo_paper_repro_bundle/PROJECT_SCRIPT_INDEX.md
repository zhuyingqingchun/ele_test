# 项目脚本与核心实现文件索引

更新时间：2026-03-17  
项目根目录：`/mnt/PRO6000_disk/swd/ele_servo_gpt5`

## 1. 说明
本文件用于集中记录当前项目涉及到的主要脚本文件与核心实现文件，方便交接、复现实验与代码定位。

建议阅读顺序：
1. 先看 `src/` 下可直接执行的入口脚本
2. 再看 `servo_diagnostic/` 与 `servo_llm_alignment/` 下的核心实现文件
3. 最后看根目录辅助脚本与文档

## 2. 当前主线最常用入口脚本
### 数据生成与数据集构建
- `src/run_servo_pipeline_expanded.py`
- `src/build_servo_multimodal_dataset.py`
- `src/build_servo_handoff_dataset.py`
- `src/build_servo_window_dataset.py`
- `src/build_stage1_alignment_corpus.py`

### 训练与评估
- `src/train_stage1_alignment.py`
- `src/train_stage1_diagnostic.py`
- `src/evaluate_stage1_diagnostic.py`
- `src/evaluate_stage1_retrieval.py`
- `src/run_stage1_ablation.py`
- `src/run_stage2_reasoning.py`
- `src/stage2_evaluate.py`

### 传统神经网络/对比实验
- `src/train_convtransformer_classifier.py`
- `src/evaluate_classifier.py`
- `src/evaluate_cnn1d_loco.py`
- `src/evaluate_tree_loco.py`
- `src/run_loco_benchmarks.py`
- `src/train_models.py`

### 分析与可视化
- `src/analyze_built_dataset.py`
- `src/analyze_dataset.py`
- `src/analyze_fault_injection_mechanisms.py`
- `src/analyze_servo_failure_modes.py`
- `src/analyze_servo_multimodal_predictions.py`
- `src/plot_confusion_matrix.py`
- `src/summarize_servo_multimodal_reports.py`
- `src/visualize_embeddings.py`
- `src/visualize_small_models.py`
- `src/infer_servo_multimodal_method.py`
- `src/extract_servo_features.py`
- `src/run_hierarchical_diagnosis.py`
- `src/simple_fault_diagnosis.py`
- `src/build_multiclass_dataset.py`

## 3. `src/` 目录完整脚本列表
- `src/analyze_built_dataset.py`
- `src/analyze_dataset.py`
- `src/analyze_fault_injection_mechanisms.py`
- `src/analyze_servo_failure_modes.py`
- `src/analyze_servo_multimodal_predictions.py`
- `src/build_multiclass_dataset.py`
- `src/build_servo_handoff_dataset.py`
- `src/build_servo_multimodal_dataset.py`
- `src/build_servo_window_dataset.py`
- `src/build_stage1_alignment_corpus.py`
- `src/evaluate_classifier.py`
- `src/evaluate_cnn1d_loco.py`
- `src/evaluate_stage1_diagnostic.py`
- `src/evaluate_stage1_retrieval.py`
- `src/evaluate_tree_loco.py`
- `src/extract_servo_features.py`
- `src/infer_servo_multimodal_method.py`
- `src/plot_confusion_matrix.py`
- `src/run_hierarchical_diagnosis.py`
- `src/run_loco_benchmarks.py`
- `src/run_servo_pipeline_expanded.py`
- `src/run_stage1_ablation.py`
- `src/run_stage2_reasoning.py`
- `src/simple_fault_diagnosis.py`
- `src/stage2_evaluate.py`
- `src/summarize_servo_multimodal_reports.py`
- `src/train_convtransformer_classifier.py`
- `src/train_models.py`
- `src/train_servo_multimodal_method.py`
- `src/train_stage1_alignment.py`
- `src/train_stage1_diagnostic.py`
- `src/visualize_embeddings.py`
- `src/visualize_small_models.py`

## 4. `servo_diagnostic/` 核心实现文件
这些文件不是直接运行入口，但属于数据生成、故障机理、传统方法和多模态诊断主链的核心实现。

### 仿真与工况/故障机理
- `servo_diagnostic/pipeline.py`
- `servo_diagnostic/simulator.py`
- `servo_diagnostic/plant.py`
- `servo_diagnostic/controllers.py`
- `servo_diagnostic/controller_tuning.py`
- `servo_diagnostic/operating_conditions.py`
- `servo_diagnostic/scenarios.py`
- `servo_diagnostic/validation.py`
- `servo_diagnostic/io_utils.py`

### 特征与诊断
- `servo_diagnostic/feature_engineering.py`
- `servo_diagnostic/multimodal_method.py`
- `servo_diagnostic/hierarchical_diagnosis.py`
- `servo_diagnostic/baseline_diagnosis.py`
- `servo_diagnostic/cnn_baseline.py`
- `servo_diagnostic/tree_baseline.py`
- `servo_diagnostic/benchmark_runner.py`
- `servo_diagnostic/benchmark_utils.py`

### 其他支持文件
- `servo_diagnostic/config.py`
- `servo_diagnostic/plots.py`
- `servo_diagnostic/servo_knowledge_base.json`
- `servo_diagnostic/__init__.py`

## 5. `servo_llm_alignment/` 核心实现文件
这些文件构成当前 `stage1 alignment / stage1 diagnostic / semantic token fusion` 主链。

### 核心模型与训练支撑
- `servo_llm_alignment/signal_encoder.py`
- `servo_llm_alignment/diagnostic.py`
- `servo_llm_alignment/trainer.py`
- `servo_llm_alignment/losses.py`
- `servo_llm_alignment/runtime.py`
- `servo_llm_alignment/prototypes.py`
- `servo_llm_alignment/retriever.py`

### 文本语料与模板
- `servo_llm_alignment/text_templates.py`
- `servo_llm_alignment/text_encoder.py`
- `servo_llm_alignment/corpus.py`
- `servo_llm_alignment/context_builder.py`
- `servo_llm_alignment/events.py`
- `servo_llm_alignment/template_config.py`

### 数据与推理接口
- `servo_llm_alignment/dataset.py`
- `servo_llm_alignment/qwen_reasoner.py`
- `servo_llm_alignment/config.py`
- `servo_llm_alignment/__init__.py`

## 6. 根目录当前可见辅助脚本/文档
### 根目录脚本
- `run_all.sh`
- `sim.py`
- `plot_csv_timeseries.py`
- `v5_test.sh`
- `v6_test.sh`

### 重要文档
- `CURRENT_HANDOFF.md`
- `TEXT_SEMANTIC_CLASSIFICATION_EXPERIMENT.md`
- `fault_diagnosis_assessment.md`
- `servo_model_math_and_topology.md`
- `数学公式.md`
- `总结.md`

## 7. 当前最值得优先关注的文件
如果只想快速理解当前主线，先读下面这些：

1. `CURRENT_HANDOFF.md`
2. `src/run_servo_pipeline_expanded.py`
3. `src/build_servo_multimodal_dataset.py`
4. `src/build_stage1_alignment_corpus.py`
5. `src/train_stage1_diagnostic.py`
6. `src/evaluate_stage1_diagnostic.py`
7. `servo_diagnostic/scenarios.py`
8. `servo_diagnostic/feature_engineering.py`
9. `servo_llm_alignment/signal_encoder.py`
10. `servo_llm_alignment/diagnostic.py`
11. `servo_llm_alignment/runtime.py`
12. `TEXT_SEMANTIC_CLASSIFICATION_EXPERIMENT.md`

## 8. 当前语义实验新增关注文件
如果要继续 `semantic_token_hardcore / semantic_aux_hardcore` 这条线，重点再加读：
- `servo_llm_alignment/diagnostic.py`
- `servo_llm_alignment/prototypes.py`
- `servo_llm_alignment/runtime.py`
- `src/train_stage1_diagnostic.py`
- `src/evaluate_stage1_diagnostic.py`
- `TEXT_SEMANTIC_CLASSIFICATION_EXPERIMENT.md`

## 9. 维护建议
后续新增入口脚本后，建议同步更新本文件，至少补：
- 文件路径
- 用途
- 是否属于默认主线
