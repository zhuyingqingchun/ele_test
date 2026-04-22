# 第14轮综合修改方案报告

## 目标

这一版将前面四个优先级尽量整合到一个可一次性落地、一次性完成主要训练与评估导出的版本：

1. **Stage 3/4 的模态—证据一致性可解释评估分支（MECB）**
2. **thermal 分支增强**
3. **position 的 reference-aware 补齐**
4. **在不重写 electrical 主干的前提下，补足 electrical 在可解释评估中的证据角色**
5. **保留并兼容第7轮的 quality-aware fusion / modality dropout**
6. **兼容第8轮增强版 vibration encoder**
7. **新增一键组合运行入口**

---

## 本次补丁涉及文件

- `ele_servo_paper_repro_bundle/servo_llm_alignment/text_templates.py`
- `ele_servo_paper_repro_bundle/src/build_stage1_alignment_corpus.py`
- `ele_servo_paper_repro_bundle/experiments_smoke_20260316/exp1_decoupled_models.py`
- `ele_servo_paper_repro_bundle/experiments_smoke_20260316/train_exp1_decoupled_stages.py`
- `ele_servo_paper_repro_bundle/experiments_smoke_20260316/run_exp1_decoupled_combined_all.sh`（新增）

---

## 这版具体做了什么

### 1. Stage 3/4 的 MECB 可解释评估分支

补丁不把 MECB 直接塞进主损失，而是先作为 **评估支路** 接到现有 Stage 3/4 上，避免扰动主实验稳定性。

核心改动：

- `StageOutputs` 新增：
  - `modality_embeddings`
  - `modality_names`
- four-branch backbone 在前向中导出四个模态级 pooled embedding：
  - position
  - electrical
  - thermal
  - vibration
- `build_text_cache()` 从只缓存 `combined_text` 扩展为同时缓存：
  - `combined`
  - `evidence`
  - `mechanism`
  - `contrast`
- 测试阶段新增导出：
  - `test_modality_evidence_scores.csv`
  - `test_modality_evidence_summary.json`

评估逻辑：

- 对每个测试样本，计算四个模态 embedding 与四个文本视图的 cosine score
- 得到 `4 modalities × 4 text views` 的解释矩阵
- 统计：
  - 各类别的 mean score
  - 各视图 dominant modality
  - `evidence_primary_hit_rate`
  - `mechanism_primary_or_support_hit_rate`
  - mechanical-related 子集的对应 hit rate

这一步会直接服务论文叙事中的：

> 模态—故障证据一致性  
> signal branch 是否真的对齐到正确的 evidence / mechanism 语义

---

### 2. thermal 分支增强

当前代码里 thermal 还是 `raw_only`，这对热相关故障的论文叙事不够强。

补丁把 `thermal` 的 `modality_tf` 特征从 `raw_only` 改成：

- `low_order_trend`
- `first_difference`
- `thermal_gap`

具体实现：

- `trend = _low_order_dct_trend(x, keep_bins=4)`
- `grad = _first_difference(x)`
- `thermal_gap = winding_temp - housing_temp`

这样 thermal 不再只是瞬时温度输入，而是能表达：

- 热累积趋势
- 升温速度
- 绕组与壳体的温差传播关系

---

### 3. position 的 reference-aware 补齐

当前 strict four-modality 主线的 position 分支缺少 reference-aware 信息。

补丁做了两层补齐：

#### 3.1 strict position columns 增加 `theta_ref_deg`

`STRICT_DECOUPLED_COLUMNS["pos"]` 改为包含：

- `theta_ref_deg`
- `theta_meas_deg`
- `theta_motor_meas_deg`
- `omega_motor_meas_deg_s`
- `encoder_count`
- `motor_encoder_count`

#### 3.2 position 特征增强增加显式 tracking cues

`_augment_position_features()` 新增：

- `tracking_error = theta_meas - theta_ref`
- `encoder_gap = encoder_count - motor_encoder_count`

这样有助于解释：

- tracking-related 故障
- position bias
- backlash / transmission mismatch

---

### 4. evidence modality 显式标注

为了让 Stage 3/4 的一致性评估不只是“后处理拍脑袋”，补丁把 evidence modality 先验显式写回 alignment corpus。

在：

- `text_templates.py`
- `build_stage1_alignment_corpus.py`

中新增：

- `PRIMARY_MODALITY_BY_SCENARIO`
- `SUPPORT_MODALITIES_BY_SCENARIO`
- 输出字段：
  - `evidence_modalities.primary`
  - `evidence_modalities.supporting`
  - `evidence_primary_modality`

这样后续一致性评估可以直接对照“预期主证据模态”。

---

### 5. 与第7轮 / 第8轮保持兼容

这版不会覆盖掉第7轮和第8轮，而是以组合方式继续保留：

- 第7轮：
  - quality-aware fusion
  - modality dropout
  - quality gate / branch energy export
- 第8轮：
  - vibration triple-branch encoder
  - vibration capacity increase

因此，这个综合版本质上是：

> 第7轮机制  
> + 第8轮振动结构  
> + Stage 3/4 一致性评估支路  
> + thermal / position 的补齐

---

## 新增一键运行脚本

新增：

- `run_exp1_decoupled_combined_all.sh`

默认行为：

- 开启 `QUALITY_AWARE_FUSION=1`
- 默认 `MODALITY_DROP_PROB=0.10`
- 默认 `FEATURE_MODE=modality_tf`
- 先跑主 curriculum
- 再可选跑 ablation（`RUN_ABLATION=1`）

---

## 推荐运行方式

在仓库根目录应用补丁后：

```bash
git apply ele_test_round14_combined_priorities.patch
```

然后执行：

```bash
bash ele_servo_paper_repro_bundle/experiments_smoke_20260316/run_exp1_decoupled_combined_all.sh
```

如果只想跑主线，不跑异步消融：

```bash
RUN_ABLATION=0 bash ele_servo_paper_repro_bundle/experiments_smoke_20260316/run_exp1_decoupled_combined_all.sh
```

---

## 预期新增输出

### 原有输出继续保留

- `report.json`
- `confusion_matrix.json`
- `test_quality_gates.csv`
- `test_quality_gate_summary.json`

### 新增输出

- `test_modality_evidence_scores.csv`
- `test_modality_evidence_summary.json`

---

## 这版最适合支撑的论文叙事

这版代码最适合支持如下叙事：

1. 通过 quality-aware fusion 和 modality dropout 缓解四模态中的模态不平衡问题；
2. 通过增强版 vibration encoder 提升机械类故障的振动表征能力；
3. 通过对 thermal 和 position 的补齐，让热故障与 tracking 类故障的物理证据更完整；
4. 通过 Stage 3/4 的 MECB 支路，直接评估信号模态与 evidence / mechanism 语义的一致性；
5. 最终不只看准确率，还能看：
   - 哪个模态和哪类证据对齐
   - 机械类故障中 vibration 是否真的被激活
   - thermal / position 是否在对应故障上更贴近 evidence 语义

---

## 校验情况

本轮已完成的校验：

- 补丁文件已生成
- 补丁 unified diff 语法已用 `git apply --stat` 成功解析
- 报告文件已生成
- 文件路径均为仓库根目录相对路径形式

尚未完成的校验：

- 未在你的本地完整仓库上实际 `git apply --check`
- 未在真实训练环境执行完整训练
- 未对训练后 JSON/CSV 输出做运行级验证

---

## 建议

如果你下一步要最稳地推进，建议先跑：

- `RUN_ABLATION=0` 的主线综合版

先看：

- Stage 2 / Stage 3 / Stage 4 是否稳定
- `test_quality_gate_summary.json`
- `test_modality_evidence_summary.json`

之后再补跑 ablation，避免一次性起太多异步任务导致排查困难。
