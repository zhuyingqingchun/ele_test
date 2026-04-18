# Signal Prefix Qwen Reasoning Handoff

## 1. Scope
This bundle is prepared for external handoff around the `run_signal_prefix_qwen_reasoning.sh` experiment and downstream RAG-style data use.

It includes:
- original generated servo CSV data under `raw_data/diagnostic_datasets/`
- experiment-ready derived datasets under `derived_data/`
- experiment scripts and dependent code under `code/`
- final 1.5B experiment outputs under `results/`

## 2. Main Files
- `raw_data/diagnostic_datasets/servo_fault_diagnosis_dataset_full.csv`
  - full merged source CSV used as the input to downstream dataset builders
- `derived_data/servo_multimodal_handoff_dataset.npz`
  - multimodal signal dataset used by the signal-prefix reasoning experiment
- `derived_data/servo_multimodal_handoff_metadata.csv`
  - metadata aligned with the multimodal handoff dataset
- `derived_data/stage1_alignment_corpus.jsonl`
  - text corpus aligned with the handoff dataset by row index
- `results/report.json`
  - final merged test report for the Qwen2.5-1.5B signal-prefix reasoning experiment
- `results/test_reasoning_predictions.csv`
  - per-sample test predictions for the final experiment

## 3. Data Alignment
- `stage1_alignment_corpus.jsonl` line `i` corresponds to sample `i` in `servo_multimodal_handoff_dataset.npz`
- `index` is the stable alignment key used across metadata, text, and signal views

## 4. Data Generation Flow
1. Generate raw servo simulation CSV data:
   - `code/src/run_servo_pipeline_expanded.py`
2. Build multimodal dataset from the full CSV:
   - `code/src/build_servo_multimodal_dataset.py`
3. Build external handoff dataset and metadata:
   - `code/src/build_servo_handoff_dataset.py`
4. Build stage-1 alignment text corpus from metadata:
   - `code/src/build_stage1_alignment_corpus.py`
5. Run the 1.5B signal-prefix reasoning experiment:
   - `code/experiments_smoke_20260316/run_signal_prefix_qwen_reasoning.sh`

## 5. Minimal Reproduction Commands
Assuming the current repository root layout is preserved inside this bundle:

```bash
PYTHONPATH=. python code/src/run_servo_pipeline_expanded.py --repeat-count 3 --seed-stride 10000
python code/src/build_servo_multimodal_dataset.py
python code/src/build_servo_handoff_dataset.py
PYTHONPATH=. python code/src/build_stage1_alignment_corpus.py
bash code/experiments_smoke_20260316/run_signal_prefix_qwen_reasoning.sh
```

## 6. Notes
- The raw and derived datasets are large. Prefer working on local storage rather than network mounts.
- The build scripts import modules from `code/servo_diagnostic` and `code/servo_llm_alignment`.
- The experiment script expects a local Qwen checkpoint path and a trained stage-2 signal checkpoint; update those paths before rerunning in a new environment.
- The text fields in the alignment corpus are summary-style evidence and mechanism descriptions, not fine-grained pointwise annotations over the raw signal.
