# Four-Modality Experiment Report

## Scope

This report only covers the current strict four-modality line:

- `position_feedback`
- `electrical_feedback`
- `thermal_feedback`
- `vibration_feedback`

The report excludes the older mixed-modality experiments based on `ctx/res/freq` style inputs.

## Input Definition

Current four-modality input is organized as:

- `position_feedback`
  - `theta_meas_deg`
  - `theta_motor_meas_deg`
  - `omega_motor_meas_deg_s`
- `electrical_feedback`
  - `current_meas_a`
  - `current_d_meas_a`
  - `current_q_meas_a`
  - `bus_current_meas_a`
  - `phase_voltage_a_meas_v`
  - `phase_voltage_b_meas_v`
  - `phase_voltage_c_meas_v`
  - `voltage_meas_v`
  - `available_bus_voltage_v`
- `thermal_feedback`
  - `winding_temp_c`
  - `housing_temp_c`
- `vibration_feedback`
  - `vibration_accel_mps2`

## Dataset and Split Protocol

### Dataset Source

The current main experiments use:

- signal dataset:
  - `derived_datasets/servo_multimodal_handoff_dataset.npz`
- text corpus:
  - `derived_datasets/stage1_alignment_corpus.jsonl`

### Classification Task

- task type: multi-class servo fault diagnosis
- number of classes: `17`
- sample organization: fixed-length time windows
- window length used by the traditional baseline dataset: `256`
- traditional window dataset split sizes:
  - train: `17361`
  - val: `3716`
  - test: `3736`

The four-modality staged framework uses the same underlying diagnosis labels and aligned text records.

### Train/Validation/Test Split

The current four-modality staged framework uses three-way stratified splitting:

- training ratio: `70%`
- validation ratio: `15%`
- test ratio: `15%`

The split is class-stratified and uses a fixed random seed.

### Four-Modality Input Principle

The current report only considers the strict four-modality setting:

- `position_feedback`
- `electrical_feedback`
- `thermal_feedback`
- `vibration_feedback`

Design principles:

- each raw input column belongs to only one primary modality
- no `ctx/res/freq` mixed external modality is used
- current and voltage are merged into a single electrical modality
- derived features are not exposed as parallel primary modalities in this main line

### Data Sampling Procedure

The current four-modality experiments operate on pre-windowed samples. Each sample corresponds to one fixed-length temporal segment and one diagnosis label.

Sampling procedure:

1. continuous servo signals are segmented into fixed windows
2. each window is assigned one scenario label
3. modality-specific columns are selected from the multimodal dataset
4. the selected columns are reorganized into the current four-modality tensor structure

For the traditional baseline dataset:

- source file:
  - `derived_datasets/servo_window_handoff_dataset.npz`
- each sample shape:
  - `256 x 34`
- after column selection for the current four-modality comparison, the effective input becomes:
  - `256 x 14`

For the staged four-modality framework:

- source file:
  - `derived_datasets/servo_multimodal_handoff_dataset.npz`
- each sample is already organized as modality-specific temporal arrays
- the four-modality experiment only keeps:
  - position channels
  - electrical channels
  - thermal channels
  - vibration channels
- residual, context, and frequency-style external modality groups are excluded from the current main line

For the staged framework, each sample is represented as:

- `X_pos`: position window
- `X_electrical`: electrical window
- `X_thermal`: thermal window
- `X_vibration`: vibration window
- `y_cls`: diagnosis label

This means the staged framework does not consume one flat mixed feature tensor. It consumes a structured four-branch signal batch.

### Modality Reconstruction from the Original Dataset

The current four-modality tensors are reconstructed from the underlying multimodal source arrays as follows:

- `position_feedback`
  - selected from the original position channel set
- `electrical_feedback`
  - selected from the original electrical channel set
  - current and voltage variables are merged into one branch
- `thermal_feedback`
  - selected from the original thermal channel set
- `vibration_feedback`
  - selected from the original vibration channel set

This reconstruction step is important because the report does not use the earlier mixed modality buckets such as residual, frequency, or context as standalone primary inputs.

### Data Preprocessing

The current preprocessing pipeline is intentionally simple and uniform.

#### Numeric Standardization

For each modality branch:

- mean is computed on the training split only
- standard deviation is computed on the training split only
- each split is normalized using the training statistics
- very small standard deviations are clipped to `1.0` to avoid numerical instability

This avoids validation/test leakage during normalization.

More explicitly, for each modality tensor:

- `mean = mean(train_values, axis=(sample, time))`
- `std = std(train_values, axis=(sample, time))`
- normalized value:
  - `(x - mean) / std`

The normalization is therefore channel-wise and branch-wise, but shared over all time steps inside the same channel.

#### Branch-Wise Normalization

Normalization is applied separately for:

- `X_pos`
- `X_electrical`
- `X_thermal`
- `X_vibration`

This preserves the independent statistical scale of each modality instead of forcing all channels into one global mixed normalization step.

#### Text Preprocessing in Stage 3 and Stage 4

For Stage 3 and Stage 4:

- aligned text descriptions are loaded from the text corpus
- text embeddings are encoded in batches
- the resulting text embedding cache is stored on disk
- later training and evaluation reuse the cached embeddings instead of re-encoding text every iteration

This reduces repeated inference cost for the frozen text encoder.

### Modality-Wise Tensor Shapes

Under the current four-modality design, one sample is decomposed into:

- `position_feedback`
  - `T x 3`
- `electrical_feedback`
  - `T x 9`
- `thermal_feedback`
  - `T x 2`
- `vibration_feedback`
  - `T x 1`

where `T` is the temporal window length.

For the traditional baseline comparison, `T = 256`.

## Evaluation Metrics

### Classification Metrics

For the four-stage diagnosis framework and the traditional baselines, the main metric is:

- `scenario accuracy`

The report uses:

- `train_scenario_accuracy`
- `val_scenario_accuracy`
- `test_scenario_accuracy`

or for traditional models:

- `best_val_acc`
- `test_acc`

### Loss Terms

For the staged framework:

- classification loss: cross-entropy with label smoothing
- alignment loss: contrastive signal-text alignment loss in Stage 3 and Stage 4
- total loss in Stage 3 and Stage 4:
  - `classification loss + lambda_align * alignment loss`

### Additional Diagnostic Outputs

The staged framework also exports:

- train confusion matrix
- validation confusion matrix
- test confusion matrix

The traditional baseline framework exports:

- test confusion matrix in compact `csv` format

### QA Metric

For the dialog QA LoRA experiment, the reported metric is:

- `test_scenario_accuracy`

Its definition is:

- only scenario-identification questions are counted
- the generated answer is normalized
- accuracy is computed by exact-match style comparison against the target scenario label

## Training Configuration

### Shared Optimization Setup

For the main four-stage classification line:

- optimizer: `AdamW`
- random seed: `7`
- label smoothing:
  - Stage 1: `0.05`
  - Stage 2: `0.05`
  - Stage 3: `0.03`
  - Stage 4: `0.03`

### Stage-Wise Hyperparameters

Default configuration of the current four-modality line:

- Stage 1
  - epochs: `36` in the current curriculum script
  - learning rate: `1e-3`
  - batch size: `16`
- Stage 2
  - epochs: `20`
  - learning rate: `3e-4`
  - batch size: `16`
- Stage 3
  - epochs: `16`
  - learning rate: `2e-4`
  - batch size: `16`
  - `lambda_align = 0.15`
- Stage 4
  - epochs: `16`
  - learning rate: `2e-4`
  - batch size: `16`
  - `lambda_align = 0.08`

Transformer-related settings in the main line:

- token dimension: `256`
- model dimension: `128`
- Stage 2 fusion encoder:
  - layers: `2`
  - heads: `8`
  - feedforward dimension: `768`
- Stage 4 joint encoder:
  - layers: `4`
  - heads: `8`
  - feedforward dimension: `768`

### Text Encoder Settings

Two text settings are involved in the current report:

- default semantic line:
  - `mock` frozen text encoder
- enhanced semantic line:
  - `QwenFrozenTextEncoder`
  - default path:
    - `/mnt/PRO6000_disk/models/Qwen/Qwen2.5-7B-Instruct`

The text encoder is used as a frozen feature encoder inside Stage 3 and Stage 4 classification experiments.

### QA LoRA Settings

For the dialog QA experiment:

- base model: `Qwen2.5-7B-Instruct`
- adaptation method: `LoRA`
- optimizer: `AdamW`
- default batch size: `2`
- default epochs: `1`
- gradient accumulation: `8`
- max sequence length: `256`

## Stage-Wise Training Protocol

The current four-modality framework is trained sequentially.

### Initialization Chain

- Stage 1 is trained from scratch
- Stage 2 is initialized from Stage 1
- Stage 3 is initialized from Stage 2
- Stage 4 is initialized from Stage 3

This protocol enforces progressive transfer of representation ability across stages.

### Role of Each Stage

- Stage 1 learns modality-specific signal representations and a basic classification boundary
- Stage 2 learns cross-modality token fusion on top of the signal backbone
- Stage 3 introduces semantic supervision through signal-text alignment
- Stage 4 performs joint signal-text reasoning with the text representation injected as a token

### Parameter Training Behavior

In the standard classification line:

- the signal backbone is trainable
- the text encoder used for semantic embedding is frozen
- Stage 3 and Stage 4 optimize both classification and alignment objectives

## Fairness of Baseline Comparison

The traditional baselines under the current four-modality report follow the same experimental protocol as closely as possible.

### Shared Conditions

- same dataset family
- same train/validation/test split directory
- same signal standardization rule:
  - mean and standard deviation computed on the training split only
- same current four-modality input subset
- same diagnosis target labels

### Compared Models

The current baseline comparison includes:

- `CNN-TCN`
- `BiLSTM`
- `ResNet-FCN`
- `Transformer Encoder`
- `Dual-branch CNN-BiLSTM-Transformer + Cross-Attention`

### Comparison Scope

The baseline comparison is intended to show relative diagnostic performance under the same four-modality input.

It should not be interpreted as a computational-efficiency benchmark because:

- the architectures have different capacities
- the dual-branch model is structurally more complex
- no parameter-budget matching has been enforced yet

## Ablation Design

### Leave-One-Out Modality Ablation

The main modality ablation uses the following protocol:

- train a full four-modality model
- remove one modality at a time
- retrain and evaluate under the same training schedule

Purpose:

- measure how much each modality contributes to the fused system
- identify which modality is most critical for performance retention

### Why Electrical is a Single Modality

Current and voltage are merged into `electrical_feedback` because:

- both are tightly coupled physical measurements
- both describe the electrical state of the servo drive
- treating them as one primary branch yields a cleaner modality definition than exposing them as two separate top-level modalities

### Single-Modality Experiment Design

In addition to leave-one-out ablation, single-modality scripts have been prepared for:

- `only_position`
- `only_electrical`
- `only_thermal`
- `only_vibration`

Purpose:

- measure the independent upper bound of each modality
- complement leave-one-out ablation with a direct single-branch comparison

These results are not yet included in the current report because the runs are not part of the currently collected result set.

## Dialog QA LoRA Experimental Protocol

### QA Data Construction

The QA experiment uses the aligned text corpus:

- each record provides a semantic fault description
- from each record, two QA-style examples are constructed:
  - scenario identification question
  - short diagnosis generation question

### QA Split

The current QA pipeline uses:

- train split
- validation split
- test split

The split is stratified by scenario label and controlled by the same random seed.

### QA Evaluation Scope

The current accuracy report only counts:

- scenario-identification questions

The diagnosis-generation questions are used for supervision, but are not the direct accuracy target in the current summary.

### Caution on QA Result Interpretation

The current dialog QA result is promising, but should still be interpreted carefully because:

- QA examples are derived from aligned semantic descriptions rather than raw signal input
- the benchmark is easier than end-to-end signal-conditioned generation
- stricter record-level leakage checks are still recommended before using it as a final headline result

## Model Architecture

### Main Four-Modality Backbone

The current main model uses a strict four-branch signal backbone:

- `position branch`
- `electrical branch`
- `thermal branch`
- `vibration branch`

Each branch encodes one modality independently before cross-modality fusion. In the current implementation:

- `position branch` uses a temporal encoder for motion-related feedback
- `electrical branch` uses a temporal encoder over the merged current-voltage feedback
- `thermal branch` uses a compact thermal-state encoder
- `vibration branch` uses a vibration-state encoder

The encoded modality tokens are concatenated into a unified signal token sequence and projected into a shared token space.

At the signal backbone level, the full processing path is:

1. modality-specific time series input
2. branch encoder produces modality token sequence
3. branch tokens are concatenated
4. concatenated tokens are projected into a shared token embedding space
5. later stages operate on this shared signal token sequence

### Branch Encoder Details

The current backbone is not a shallow concatenation model. Each modality first passes through its own dedicated encoder.

#### Position Branch

- encoder type: temporal encoder
- input channels: position-related feedback channels
- input tensor shape per sample: `T x 3`
- token count: `20`
- hidden dimension inside branch encoder: `48`
- temporal dilation pattern: `(1, 2, 4, 8)`

#### Electrical Branch

- encoder type: temporal encoder
- input channels: merged current-voltage feedback channels
- input tensor shape per sample: `T x 9`
- token count: `24`
- hidden dimension inside branch encoder: `56`
- temporal dilation pattern: `(1, 2, 4, 8)`

#### Thermal Branch

- encoder type: thermal state encoder
- input tensor shape per sample: `T x 2`
- token count: `8`
- hidden dimension inside branch encoder: `32`

#### Vibration Branch

- encoder type: vibration state encoder
- input tensor shape per sample: `T x 1`
- token count: `12`
- hidden dimension inside branch encoder: `32`

### Shared Token Projection

After branch-specific encoding:

- branch tokens are concatenated
- total signal token count becomes:
  - `20 + 24 + 8 + 12 = 64`
- tokens are projected into a shared latent token space

Shared dimensions:

- branch encoder output dimension before shared projection: `model_dim = 128`
- shared token dimension after projection: `token_dim = 256`

This shared token space is then used by Stage 2, Stage 3, and Stage 4.

In other words, the Stage 1 backbone outputs a signal token tensor with shape approximately:

- `B x 64 x 256`

where:

- `B` is batch size
- `64` is the total token count
- `256` is the shared token dimension

### Four-Stage Framework

#### Stage 1: Encoder Classification

- four modality-specific encoders
- token projection into shared latent space
- global pooled signal representation
- multilayer classification head

Purpose:

- learn stable signal-side discriminative features without text involvement

Detailed flow:

1. four modality encoders generate signal tokens
2. all tokens are concatenated into a `64-token` signal sequence
3. the sequence is mean-pooled into one global signal embedding
4. the pooled embedding is normalized
5. a multilayer classifier outputs the final diagnosis logits

Classifier structure:

- LayerNorm
- Linear
- GELU
- Dropout
- Linear to `17` diagnosis classes

#### Stage 2: Signal Fusion

- Stage 1 signal backbone
- Transformer encoder over concatenated signal tokens
- sequence pooling
- classification head

Purpose:

- explicitly model inter-modality relationships and signal-level token interactions

Detailed flow:

1. Stage 1 backbone outputs the `64-token` signal sequence
2. a learnable `CLS` token is prepended
3. sequence length becomes `65`
4. a Transformer encoder performs signal-level token mixing
5. the encoded sequence is pooled
6. the pooled representation is sent to the classifier

Current default Stage 2 fusion settings:

- Transformer layers: `2`
- attention heads: `8`
- feedforward dimension: `768`
- pooling mode in the current main line: attention pooling

This stage can be viewed as the main inter-modality interaction module in the current framework.

#### Stage 3: Signal-Text Alignment

- Stage 1 signal backbone
- signal classification head
- text encoder projection branch
- contrastive alignment loss between signal embedding and text embedding

Purpose:

- align signal representation with semantic fault descriptions
- preserve classification while adding semantic supervision

Detailed flow:

1. Stage 1 backbone produces the signal token sequence
2. signal tokens are pooled into one global signal embedding
3. the same signal embedding is used for:
   - diagnosis classification
   - semantic alignment
4. the frozen text encoder generates one text embedding per sample
5. the text embedding is projected into the same latent space
6. a contrastive alignment loss encourages signal and text embeddings from the same sample to be close

Optimization objective:

- `classification loss + lambda_align * contrastive alignment loss`

Text pathway details:

- text encoder output is frozen
- one text embedding is produced per sample
- the text embedding is projected to the same latent dimension as the signal embedding before contrastive alignment

#### Stage 4: Signal-Text Joint Reasoning

- Stage 1 signal backbone
- text embedding projected as a text token
- signal tokens and text token jointly processed by a Transformer encoder
- pooled joint representation for classification
- auxiliary alignment loss retained

Purpose:

- perform joint signal-text reasoning for final diagnosis

Detailed flow:

1. Stage 1 backbone outputs the `64-token` signal sequence
2. frozen text embedding is projected into one `text token`
3. a learnable `CLS` token is prepended
4. the final input sequence becomes:
   - `CLS + 64 signal tokens + 1 text token`
5. the Stage 4 Transformer encoder jointly processes signal and text tokens
6. the pooled output is classified into the final diagnosis label

Current default Stage 4 settings:

- Transformer layers: `4`
- attention heads: `8`
- feedforward dimension: `768`
- text token is appended to the signal token sequence
- auxiliary alignment loss is still retained during training

The resulting Stage 4 sequence length is:

- `1 CLS + 64 signal tokens + 1 text token = 66`

### Qwen-Enhanced Variant

For the enhanced semantic experiment:

- Stage 3 replaces the default text encoder with `QwenFrozenTextEncoder`
- Stage 4 also uses `QwenFrozenTextEncoder` for the joint text token

This keeps the signal architecture unchanged while strengthening the semantic representation on the text side.

### Dialog QA LoRA Variant

The fault dialog experiment is a separate language-model branch built on Qwen:

- base model: `Qwen2.5-7B-Instruct`
- adaptation: `LoRA`
- task style: diagnosis-oriented dialogue question answering
- training target: generate scenario labels or short diagnosis responses from textual fault descriptions

This branch is not used as the main signal classifier. It is an auxiliary experiment that evaluates whether the learned fault descriptions can support diagnosis-oriented dialogue generation.

### Signal-Prefix Qwen LoRA Variant

In addition to the text-only dialog QA branch, a signal-conditioned language-modeling experiment has also been prepared.

Its design is:

1. load the trained Stage 2 four-modality signal encoder
2. freeze the signal encoder
3. map the signal token sequence into a small number of learnable soft prefix tokens
4. prepend these signal-prefix tokens to the Qwen dialogue prompt
5. fine-tune Qwen with LoRA for diagnosis generation

This variant is intended to make the large language model consume signal-side information directly rather than relying only on textual fault descriptions.

### Traditional Baseline Architectures

The following baselines are evaluated under the same current four-modality input:

- `CNN-TCN`
  - convolutional frontend
  - temporal convolution blocks
  - pooled classification head
- `BiLSTM`
  - bidirectional recurrent sequence encoder
  - final classification layer
- `ResNet-FCN`
  - residual convolution branch
  - fully convolutional branch
  - fused classification head
- `Transformer Encoder`
  - pure sequence Transformer encoder
  - classification head
- `Dual-branch CNN-BiLSTM-Transformer + Cross-Attention`
  - time-domain branch: CNN -> BiLSTM -> Transformer
  - frequency-domain branch: FFT magnitude -> CNN -> BiLSTM -> Transformer
  - cross-attention between branches
  - CNN refinement
  - fully connected classifier

For the traditional baseline experiments under the current four-modality setting:

- the input is a flat `256 x 14` tensor
- no explicit modality branch separation is used inside the traditional baselines
- modality information is preserved only through channel selection, not through branch-specific encoders

## Main Multi-Stage Results

Source:

- `models/exp1_decoupled_v3/stage1_encoder_cls/report.json`
- `models/exp1_decoupled_v3/stage2_signal_fusion/report.json`
- `models/exp1_decoupled_v3/stage3_signal_text_align/report.json`
- `models/exp1_decoupled_v3/stage4_signal_text_llm/report.json`

| Model | Val Acc | Test Acc | Observation |
| --- | ---: | ---: | --- |
| Stage 1 encoder classification | 0.9215 | 0.9237 | Baseline four-modality encoder |
| Stage 2 signal fusion | 0.9580 | 0.9579 | Best stable result in the vanilla four-stage line |
| Stage 3 signal-text align | 0.9562 | 0.9561 | Very close to Stage 2, but no clear gain |
| Stage 4 signal-text joint reasoning | 0.9531 | 0.9528 | Slight drop relative to Stage 2/3 |

### Interpretation

- Moving from Stage 1 to Stage 2 gives the major performance gain.
- Under the current four-modality setting, Stage 3 does not provide a reliable improvement over Stage 2 when using the default text backbone.
- Stage 4 does not improve the vanilla four-modality line and is slightly worse than Stage 2.

## Qwen-Enhanced Results

Source:

- `models/exp1_decoupled_v3_qwen_dialog_qa_lora/stage3_signal_text_align_qwen/report.json`
- `models/exp1_decoupled_v3_qwen_dialog_qa_lora/stage4_signal_text_llm_qwen_cls/report.json`

| Model | Val Acc | Test Acc | Observation |
| --- | ---: | ---: | --- |
| Stage 3 with Qwen text encoder | 0.9681 | 0.9700 | Clear improvement over vanilla Stage 3 |
| Stage 4 with Qwen text encoder | 1.0000 | 1.0000 | Too ideal, needs strict verification |

### Interpretation

- Replacing the default text encoder with Qwen improves Stage 3 substantially:
  - vanilla Stage 3 test accuracy: `0.9561`
  - Qwen Stage 3 test accuracy: `0.9700`
- This is currently the most convincing text-related improvement in the four-modality line.
- The Stage 4 Qwen result reaches `1.0` on both validation and test. This should be treated as a high-risk result until it is revalidated under stricter split and leakage checks.

## Traditional Model Comparison Under Current Four-Modality Input

Source:

- `models/traditional_compare_current_modalities/cnn_tcn/cnn_tcn_current_modalities_report.json`
- `models/traditional_compare_current_modalities/bilstm/bilstm_current_modalities_report.json`
- `models/traditional_compare_current_modalities/resnet_fcn/resnet_fcn_current_modalities_report.json`
- `models/traditional_compare_current_modalities/transformer_encoder/transformer_encoder_current_modalities_report.json`
- `models/traditional_compare_current_modalities/dual_branch_xattn/dual_branch_xattn_current_modalities_report.json`

| Model | Val Acc | Test Acc | Rank |
| --- | ---: | ---: | ---: |
| Transformer Encoder | 0.8644 | 0.8573 | 1 |
| ResNet-FCN | 0.8584 | 0.8536 | 2 |
| CNN-TCN | 0.8555 | 0.8482 | 3 |
| Dual-branch CNN-BiLSTM-Transformer + Cross-Attention | 0.8146 | 0.8051 | 4 |
| BiLSTM | 0.6588 | 0.6469 | 5 |

### Interpretation

- Among traditional deep sequence baselines, `Transformer Encoder` is the strongest under the current four-modality input.
- `ResNet-FCN` and `CNN-TCN` are competitive and form a stable second tier.
- `BiLSTM` is clearly weaker.
- The newly added dual-branch cross-attention model is currently underperforming compared with simpler baselines, suggesting that its current training setup or capacity allocation is not yet optimal.

## Four-Modality Ablation Results

### Main Model Leave-One-Out Ablation

Source:

- `models/exp1_decoupled_v3_ablation_async/A0_full/stage4_signal_text_llm/report.json`
- `models/exp1_decoupled_v3_ablation_async/A1_no_position/stage4_signal_text_llm/report.json`
- `models/exp1_decoupled_v3_ablation_async/A2_no_electrical/stage4_signal_text_llm/report.json`
- `models/exp1_decoupled_v3_ablation_async/A3_no_thermal/stage4_signal_text_llm/report.json`
- `models/exp1_decoupled_v3_ablation_async/A4_no_vibration/stage4_signal_text_llm/report.json`

| Setting | Test Acc | Drop vs Full |
| --- | ---: | ---: |
| A0 full | 0.9616 | 0.0000 |
| A1 no position | 0.9486 | -0.0130 |
| A2 no electrical | 0.9050 | -0.0566 |
| A3 no thermal | 0.9441 | -0.0175 |
| A4 no vibration | 0.9547 | -0.0069 |

### Interpretation

- `electrical_feedback` is the most important modality in the current main model.
- `position_feedback` is the second most important modality.
- `thermal_feedback` has a measurable but smaller contribution.
- `vibration_feedback` contributes the least among the four, but removing it still causes degradation.

### CNN-TCN Leave-One-Out Ablation

Source:

- `models/cnn_tcn_modality_ablation_current_modalities_async/cnn_tcn_full/cnn_tcn_full_report.json`
- `models/cnn_tcn_modality_ablation_current_modalities_async/cnn_tcn_no_position/cnn_tcn_no_position_report.json`
- `models/cnn_tcn_modality_ablation_current_modalities_async/cnn_tcn_no_electrical/cnn_tcn_no_electrical_report.json`
- `models/cnn_tcn_modality_ablation_current_modalities_async/cnn_tcn_no_thermal/cnn_tcn_no_thermal_report.json`
- `models/cnn_tcn_modality_ablation_current_modalities_async/cnn_tcn_no_vibration/cnn_tcn_no_vibration_report.json`

| Setting | Test Acc | Drop vs Full |
| --- | ---: | ---: |
| full | 0.8482 | 0.0000 |
| no position | 0.7725 | -0.0757 |
| no electrical | 0.7465 | -0.1017 |
| no thermal | 0.8426 | -0.0056 |
| no vibration | 0.8346 | -0.0137 |

### Interpretation

- The same ranking is reproduced in the CNN-TCN baseline:
  - electrical is most important
  - position is second
  - vibration is limited but useful
  - thermal is weakest
- This consistency increases confidence that the observed modality ranking is not unique to one architecture.

## Single-Modality Results

Source:

- `models/exp1_decoupled_v3_single_modality_async/only_position/.../report.json`
- `models/exp1_decoupled_v3_single_modality_async/only_electrical/.../report.json`
- `models/exp1_decoupled_v3_single_modality_async/only_thermal/.../report.json`
- `models/exp1_decoupled_v3_single_modality_async/only_vibration/.../report.json`

### Best Result Across the Four Stages for Each Single Modality

| Single Modality | Best Stage | Best Test Acc |
| --- | --- | ---: |
| only_electrical | Stage 2 | 0.9237 |
| only_position | Stage 2 | 0.8558 |
| only_vibration | Stage 2 | 0.6495 |
| only_thermal | Stage 3 | 0.5670 |

### Stage 2 Single-Modality Comparison

| Single Modality | Val Acc | Test Acc |
| --- | ---: | ---: |
| only_electrical | 0.9230 | 0.9237 |
| only_position | 0.8546 | 0.8558 |
| only_vibration | 0.6542 | 0.6495 |
| only_thermal | 0.5639 | 0.5604 |

### Interpretation

- `electrical_feedback` alone already reaches `0.9237`, showing that it contains the strongest independent diagnostic signal in the current dataset.
- `position_feedback` alone is also strong and reaches `0.8558`, but remains clearly below electrical.
- `vibration_feedback` and `thermal_feedback` are much weaker as stand-alone modalities.
- For three out of four single-modality settings, the best stage is `Stage 2`, which is consistent with the main conclusion that signal fusion is the most stable improvement stage in the staged framework.
- The combination of leave-one-out ablation and single-modality experiments leads to the same ranking:
  - electrical is dominant
  - position is second
  - vibration is limited but useful
  - thermal is weakest

## Qwen Dialog QA LoRA Result

Source:

- `models/exp1_decoupled_v3_qwen_dialog_qa_lora/stage4_fault_dialog_qa_lora/report.json`
- `models/exp1_decoupled_v3_qwen_dialog_qa_lora/stage4_fault_dialog_qa_lora/test_predictions.csv`

Observed result:

- `best_val_loss = 3.65e-05`
- `test_scenario_accuracy = 1.0000`
- `test_scenario_questions = 198`

### Interpretation

- The current dialog QA LoRA experiment shows that Qwen can fit the constructed diagnosis QA task extremely well.
- However, this result should be treated as exploratory rather than final.
- The current QA data pipeline constructs multiple question styles from the same underlying record text. Therefore, the held-out QA result may still be optimistic if record-level grouping is not enforced strongly enough for all derived QA instances.

## Overall Conclusions

1. The strict four-modality reorganization is effective and produces a clean, defensible experimental main line.
2. In the vanilla four-stage framework, `Stage 2 signal fusion` is the strongest stable operating point.
3. Qwen text encoding improves `Stage 3` meaningfully and is the strongest confirmed semantic enhancement so far.
4. Under the current four-modality setting, the most important modality is `electrical`, followed by `position`.
5. Single-modality experiments confirm that `electrical` alone already carries strong discriminative power, while `thermal` and `vibration` are much weaker on their own.
6. Traditional baselines remain significantly below the main staged framework.

## Recommended Paper-Facing Claims

Safe claims:

- The proposed four-modality decoupled input design improves clarity and supports robust modality analysis.
- Signal fusion is the key contributor in the staged framework.
- Qwen-enhanced semantic alignment improves the four-modality diagnosis model.
- Electrical signals are the dominant modality for diagnosis under the current dataset and protocol.
- The proposed staged framework outperforms conventional deep time-series baselines.

Claims that should remain cautious:

- `Stage 4 Qwen classification = 1.0`
- `Dialog QA LoRA test accuracy = 1.0`

These two results are promising, but should be labeled as exploratory until they are verified under stricter split control and leakage checks.

## Remaining Gaps

1. The asynchronous ablation directory still contains some old experiment-name residues from earlier modality definitions, so only the current-name entries should be cited.
2. The dialog QA benchmark should be rechecked with stricter record-level separation before being used as a final paper result.

## Table Summary

### Table 1. Main Results of the Four-Modality Staged Framework

| Setting | Val Acc | Test Acc | Note |
| --- | ---: | ---: | --- |
| Stage 1 encoder classification | 0.9215 | 0.9237 | Four-modality signal baseline |
| Stage 2 signal fusion | 0.9580 | 0.9579 | Best stable vanilla result |
| Stage 3 signal-text align | 0.9562 | 0.9561 | Similar to Stage 2 |
| Stage 4 signal-text joint reasoning | 0.9531 | 0.9528 | Slight drop |
| Stage 3 + Qwen text encoder | 0.9681 | 0.9700 | Best confirmed semantic enhancement |
| Stage 4 + Qwen text encoder | 1.0000 | 1.0000 | Exploratory, needs strict verification |

### Table 2. Comparison with Traditional Deep Time-Series Baselines Under Current Four-Modality Input

| Model | Val Acc | Test Acc | Rank |
| --- | ---: | ---: | ---: |
| Transformer Encoder | 0.8644 | 0.8573 | 1 |
| ResNet-FCN | 0.8584 | 0.8536 | 2 |
| CNN-TCN | 0.8555 | 0.8482 | 3 |
| Dual-branch CNN-BiLSTM-Transformer + Cross-Attention | 0.8146 | 0.8051 | 4 |
| BiLSTM | 0.6588 | 0.6469 | 5 |

### Table 3. Leave-One-Out Ablation of the Main Four-Modality Model

| Setting | Test Acc | Drop vs Full |
| --- | ---: | ---: |
| Full model | 0.9616 | 0.0000 |
| No position | 0.9486 | -0.0130 |
| No electrical | 0.9050 | -0.0566 |
| No thermal | 0.9441 | -0.0175 |
| No vibration | 0.9547 | -0.0069 |

### Table 4. Leave-One-Out Ablation of CNN-TCN Under Current Four-Modality Input

| Setting | Test Acc | Drop vs Full |
| --- | ---: | ---: |
| Full model | 0.8482 | 0.0000 |
| No position | 0.7725 | -0.0757 |
| No electrical | 0.7465 | -0.1017 |
| No thermal | 0.8426 | -0.0056 |
| No vibration | 0.8346 | -0.0137 |

### Table 5. Dialog QA LoRA Result

| Setting | Best Val Loss | Test Scenario Accuracy | Number of Test Scenario Questions | Note |
| --- | ---: | ---: | ---: | --- |
| Qwen dialog QA LoRA | 3.65e-05 | 1.0000 | 198 | Exploratory, needs stricter split validation |

### Table 6. Single-Modality Results of the Main Four-Modality Framework

| Single Modality | Best Stage | Best Test Acc | Stage 2 Test Acc |
| --- | --- | ---: | ---: |
| only_electrical | Stage 2 | 0.9237 | 0.9237 |
| only_position | Stage 2 | 0.8558 | 0.8558 |
| only_vibration | Stage 2 | 0.6495 | 0.6495 |
| only_thermal | Stage 3 | 0.5670 | 0.5604 |

## Result Figures

The following figures have been generated under:

- `figures/four_modality/`

### Figure 1. Test Accuracy of the Staged Framework

![Staged Framework Test Accuracy](figures/four_modality/staged_framework_test_accuracy.png)

This figure is suitable for showing the relative positions of:

- vanilla Stage 1 to Stage 4
- Stage 3 with Qwen
- Stage 4 with Qwen

### Figure 2. Validation Curves of the Vanilla Four Stages

![Staged Framework Validation Curves](figures/four_modality/staged_framework_val_curves.png)

This figure is useful for showing:

- convergence behavior
- the stability of Stage 2
- the weaker benefit of vanilla Stage 4

### Figure 3. Traditional Baseline Comparison

![Traditional Baselines](figures/four_modality/traditional_baselines_test_accuracy.png)

This figure supports the baseline comparison section under the current four-modality input.

### Figure 4. Main Model Leave-One-Out Ablation

![Main Model Ablation](figures/four_modality/main_model_ablation.png)

This figure directly visualizes the modality contribution ranking in the main framework.

### Figure 5. CNN-TCN Leave-One-Out Ablation

![CNN-TCN Ablation](figures/four_modality/cnn_tcn_ablation.png)

This figure is useful for showing that the same modality ranking also appears in a simpler baseline model.

### Figure 6. Single-Modality Results

![Single Modality Results](figures/four_modality/single_modality_results.png)

This figure is important because it complements the leave-one-out ablation with direct independent modality performance.

### Figure 7. Confusion Matrix Comparison

![Confusion Matrix Comparison](figures/four_modality/confusion_matrices_stage2_vs_stage3qwen.png)

This figure compares:

- `Stage 2` test confusion matrix
- `Stage 3 + Qwen` test confusion matrix

It is useful for identifying whether Qwen primarily reduces confusion in a few hard classes or improves the error pattern more broadly.

### Figure 8. Qwen Dialog QA Accuracy

![Qwen Dialog QA Accuracy](figures/four_modality/qwen_dialog_qa_accuracy.png)

This figure should be treated as an exploratory result figure rather than a final headline figure, due to the caution already stated for the current QA evaluation protocol.
