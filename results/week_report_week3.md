# BMED 712 — Track A | Week 3 Report
**Date:** 2026-04-06
**Team:** Fatima Habib Farweh, Liang Li, Yasmine Khattab, Zehara Ali
**Period:** Week 3 (post-submission — revision, debugging, retraining)

---

## 1. Summary of Completed Tasks

| # | Task | Status |
|---|------|--------|
| 1 | Address all 6 professor feedback items on Progress Report | ✅ |
| 2 | Identify and document feature extraction bug | ✅ |
| 3 | Validate teammate's corrected frequency-sheet features (300k windows) | ✅ |
| 4 | Retrain 3-class models (SVM / XGBoost / RF) with corrected features | ✅ |
| 5 | First-ever 8-class (subtype-level) classification experiment | ✅ |
| 6 | Sensor ablation with new features | ✅ |
| 7 | Phase-specific asymmetry analysis (pre / during / post U-turn) | ✅ |
| 8 | VGA–IMU ordinal correlation analysis | ✅ |
| 9 | Expanded feature experiment (derived proxy features) | ✅ |
| 10 | Revised Progress Report PDF | ✅ |

---

## 2. Feature Extraction Bug — Root Cause Analysis

During the 8-class analysis attempt, we discovered the original `master_features.csv` was generated with two compounding errors:

### Bug 1 — Missing Acc channel
The extraction pipeline iterated over `FreeAcc` and `Gyr` signals only, **silently skipping the raw `Acc` channel**. Each of the 4 IMUs provides 3 signal types (Acc, FreeAcc, Gyr) × 3 axes, so 1/3 of available signals were never processed. The resulting file had 168 features instead of the expected 216+.

### Bug 2 — Trial-level aggregation
The old code averaged all time windows within each trial into a single row before saving, producing exactly 1,356 rows (one per trial). This collapsed within-trial temporal variability — a critical source of discriminative information for gait classification.

### Bug 3 — No subtype label
The original CSV contained only a 3-class `label` column (Healthy / Neuro / Ortho). The 8-class `cohort` column (HS / PD / CVA / RIL / CIPN / KOA / HOA / ACL) was never written, making subtype analysis impossible.

### Resolution
Teammate Fatemah re-extracted features from scratch with corrected code, producing the `frequency sheets/` dataset:
- **All 3 channels** included (Acc + FreeAcc + Gyr)
- **Window-level rows** (300,991 windows across 4 phases × 12 window/overlap configs)
- **Both 3-class and 8-class labels** present
- **216 features** per window (4 sensors × 3 channels × 3 axes × 6 feature types)
- Missing-value rate: < 0.12% (LF channel only, minor)

---

## 3. ML Results — Corrected Features

### 3.1 Three-class classification (healthy / neuro / ortho)

| Phase | Window | Overlap | Best Model | BAcc | F1 | Δ vs Old |
|-------|--------|---------|------------|------|-----|----------|
| post_uturn | 5 s | 50% | XGBoost | **79.2%** | 80.4% | **+7.6%** |
| pre_uturn | 5 s | 50% | XGBoost | 76.7% | 76.2% | +5.1% |
| full_gait | 3 s | 50% | XGBoost | 76.1% | 75.9% | +4.5% |
| full_gait | 5 s | 50% | SVM | 75.8% | 75.0% | +4.2% |
| uturn | 1 s | 50% | SVM | 76.5% | 75.3% | +4.9% |

Old best (with buggy features): BAcc = 71.6%.

### 3.2 Eight-class classification (subtype-level) — NEW

| Phase | Model | BAcc | F1 | Note |
|-------|-------|------|-----|------|
| full_gait 5s/50% | SVM | **41.5%** | 35.9% | Chance = 12.5% |
| pre_uturn 5s/50% | SVM | 41.0% | 36.9% | |
| post_uturn 6s/50% | SVM | 39.9% | 35.2% | |

8-class is a challenging task (8 classes, imbalanced: RIL has 5,066 windows vs ACL only 478). BAcc of 41.5% is **3.3× chance level**, confirming that IMU features carry subtype-discriminative signal.

### 3.3 Sensor ablation (full_gait 5s/50%, 3-class)

| Sensor Set | SVM | XGBoost | RF |
|------------|-----|---------|-----|
| All (HE+LB+LF+RF) | 75.8% | 75.7% | 75.3% |
| HE+LB | 72.9% | **75.1%** | 73.5% |
| **HE only** | **72.7%** | **74.0%** | **73.2%** |
| Feet (LF+RF) | 70.4% | 69.5% | 67.1% |
| RF only | 67.5% | 67.9% | 63.7% |
| LB only | 71.0% | 69.4% | 64.8% |

**Key surprise:** HE (head) sensor alone achieves 74.0% BAcc with XGBoost — retaining **97.8%** of full-sensor performance. This contrasts with prior results where RF (right foot) was dominant. The head IMU captures whole-body gait rhythm and vertical oscillation, apparently sufficient for 3-class separation.

### 3.4 Expanded feature experiment

Added 5 derived features per sensor-channel-axis (energy, DC ratio, relative variability, spectral complexity, normalized spectral power) → 216 → 396 features. **Result: no improvement** (SVM 73.9% vs 75.8% original). The proxy features are mathematically correlated with existing stats (e.g., energy ≈ rms²). This confirms the original 216-feature set is already well-designed; raw signal access would be needed for genuinely new features (kurtosis, ZCR, etc.).

---

## 4. Professor's Feedback — All 6 Items Addressed

| # | Feedback | Fix Applied |
|---|----------|-------------|
| 1 | Tone down "robust biomarkers" / "clinical deployment" | → "potential indicators", "suggests possible clinical application" |
| 2 | Streamline narrative, foreground contributions | Abstract rewritten with 2 explicit contributions; Discussion trimmed to 3 paragraphs |
| 3 | ML improvement is small — emphasize clinical insight | AUC 0.716 labeled "modest"; added: "primary contribution is clinical characterization, not model performance" |
| 4 | Sensor ablation needs sharper key-takeaway | Added highlighted key-finding box: "Foot-only retains 93% BAcc" |
| 5 | Fig 7: r=−0.206 is weak; remove regression line (VGA is ordinal) | Regression line removed; Spearman ρ annotated; added per-VGA boxplots (Panel B) |
| 6 | Table II: RIL should be negative d=−0.87 | All Cohen's d now signed (pathological − healthy); all negative: RIL −0.87, PD −0.77, … ACL −0.09 |

---

## 5. Key Insights This Week

1. **Feature extraction matters more than model tuning.** Fixing the missing Acc channel and using window-level data improved BAcc by +7.6% — more than any hyperparameter search could achieve.

2. **Head sensor is underrated.** HE alone (74.0%) nearly matches all 4 sensors (75.7%). A single head-mounted IMU could be a practical clinical screening device — simpler than bilateral foot sensors.

3. **Subtype discrimination is feasible.** 8-class BAcc of 41.5% (vs 12.5% chance) shows that IMU features carry subtype-specific signatures, even with the current feature set. With raw-signal features (kurtosis, ZCR, wavelet coefficients), this could improve further.

4. **Straight-line walking carries the signal.** Pre-U-turn phase shows significant group differences (p = 0.026); U-turn and post-U-turn do not. Yet the ML model performs best on post_uturn (79.2%) — suggesting the return corridor has cleaner signal after the turn "normalizes" initial gait variability.

5. **VGA is a coarse proxy.** Spearman ρ = −0.206 means VGA explains only ~4% of IMU asymmetry variance. IMU captures information invisible to clinical visual assessment.

---

## 6. Proposed Next Steps (Week 4)

- [ ] Access raw IMU signals to compute genuinely new features (kurtosis, skewness, zero-crossing rate, wavelet energy)
- [ ] Re-run 8-class with SMOTE or class-weighted loss to address RIL/ACL imbalance
- [ ] Phase-feature fusion: concatenate pre_uturn + post_uturn features per trial
- [ ] Discuss with professor: is head-sensor-only IMU a publishable finding?
- [ ] Begin final report structure planning

---

## 7. Files Delivered This Week

| File | Description |
|------|-------------|
| `analysis/validate_new_features.py` | Feature validation (300k windows, distributions, missing values) |
| `analysis/train_new_features.py` | 3-class, 8-class, sensor ablation with new features |
| `analysis/expand_features.py` | Expanded feature experiment (proxy derived features) |
| `analysis/fix_fig7_table2.py` | Corrected Fig 7 (no regression line) + signed Table II |
| `results/ml_new_features/` | All ML result CSVs and summary plots |
| `results/validation/` | Feature validation outputs (heatmaps, KDE plots, coverage table) |
| `results/Progress_Report_Revised.pdf` | Full revised report addressing all professor feedback |

---

---

# 第三周进展报告（中文版）

## 一、核心发现：特征提取 Bug 分析

旧版 `master_features.csv` 存在两个复合错误：

**错误1 — Acc 通道缺失：** 旧代码只处理了 FreeAcc（去重力加速度）和 Gyr（陀螺仪），**完全跳过了原始 Acc（含重力加速度）通道**。每个 IMU 有 3 种信号 × 3 轴，旧版丢失了 1/3 的信号源。最终只有 168 个特征，而非应有的 216 个。

**错误2 — 聚合粒度错误：** 旧代码在保存前将每个 trial 内的所有窗口取均值，生成 1,356 行（每 trial 一行）。这丢失了 trial 内的时间变异性信息——而这恰恰是步态分类的重要判据。

**错误3 — 无亚型标签：** 旧文件只有 3 类标签（Healthy/Neuro/Ortho），缺少 8 类 cohort 标签，无法进行亚型分析。

**修复：** 同学 Fatemah 使用修正后的代码重新提取了全部特征（`frequency sheets/`），包含全部 3 通道、窗口级数据、3 类 + 8 类标签，共 300,991 个窗口。

## 二、修正后 ML 结果

| 实验 | 最佳模型 | BAcc | 对比旧版 |
|------|---------|------|---------|
| 3类（post_uturn 5s/50%） | XGBoost | **79.2%** | +7.6% |
| 3类（full_gait 5s/50%） | SVM | 75.8% | +4.2% |
| 8类（full_gait 5s/50%） | SVM | **41.5%** | 新实验（随机12.5%） |

**传感器消融新发现：** 头部（HE）单传感器 XGBoost 达到 74.0% BAcc，接近全部 4 传感器的 75.7%。这出乎意料——头部 IMU 可能是最实用的单传感器临床方案。

**扩展特征实验：** 添加 energy/dc_ratio 等 5 种派生特征后无提升（396 个特征 vs 原 216 个），因为这些是已有统计量的数学变换，高度相关。需要访问原始信号才能提取真正新的特征（峰度、过零率、小波能量等）。

## 三、导师反馈——全部 6 条已回应

1. 语气软化 ✅（"robust biomarkers" → "potential indicators"）
2. 叙事精简 ✅（摘要重写，贡献前置）
3. ML 定位调整 ✅（AUC 0.716 标注"modest"，强调临床洞见而非模型性能）
4. 传感器消融关键结论 ✅（突出 foot-only 保留 93% 精度）
5. 图7修正 ✅（删除回归线，VGA 为序数变量，改用 Spearman + 箱线图）
6. 表II修正 ✅（Cohen's d 全部改为负值，d = 病理组 − 健康组）

## 四、本周关键洞见

1. **特征工程 > 模型调参**：修复 Acc 通道缺失 + 使用窗口级数据带来 +7.6% BAcc 提升，远超任何超参搜索效果。
2. **头部传感器被低估**：HE 单传感器（74.0%）几乎匹配全部 4 传感器（75.7%），这是此前文献中未充分强调的发现。
3. **亚型分类可行**：8 类 BAcc 41.5%（随机 12.5%）证明 IMU 特征携带亚型特异性信号。
4. **直线行走 ≈ 主要信号源**：pre-U-turn 阶段 p=0.026 显著；但 ML 在 post-uturn 表现最佳（79.2%），可能因返程步态更稳定、信噪比更高。
5. **VGA 是粗糙代理**：Spearman ρ = −0.206，VGA 仅解释 ~4% 的 IMU 不对称性方差。

## 五、下周计划

- 获取原始 IMU 信号，提取峰度/过零率/小波能量等新特征
- 8 类分类使用 SMOTE 或类别加权损失缓解不平衡
- 尝试 pre_uturn + post_uturn 阶段特征融合
- 与导师讨论：头部单传感器发现是否可作为可发表亮点
- 开始规划 final report 结构
