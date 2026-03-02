# BMED 712 — Week Report: Temporal Gait Asymmetry Analysis

**Date:** 2026-03-02 | **Author:** Track A Team | **Dataset:** 974 valid trials, 216 subjects

---

# Part I: English Report

## 1. Objective

Investigate whether gait pathology manifests as left/right **temporal imbalance** in step and stride timing, measured from heel-strike events in IMU gait data. The goal: turn the professor's observation that "there is something there" in L-R IMU differences into statistically validated evidence.

## 2. Method

| Step | Description |
|------|-------------|
| 1 | Extract heel-strike (HS) timestamps from `leftGaitEvents` / `rightGaitEvents` metadata |
| 2 | Exclude U-turn segments (biomechanically different gait pattern) |
| 3 | Compute **step time** (contralateral HS intervals) and **stride time** (ipsilateral HS intervals) |
| 4 | Calculate asymmetry metrics: signed AI, |AI|, |L-R|, ratio, within-trial CV |
| 5 | Aggregate per subject (mean across trials) to avoid pseudoreplication |
| 6 | Welch's t-test + Cohen's d for healthy vs pathological comparisons |

## 3. Key Finding: Healthy Gait is MORE Asymmetric

The central and counterintuitive finding: **healthy subjects show larger, more consistent temporal asymmetry** than pathological groups. This reflects loss of natural motor lateralization in disease, not increased symmetry.

### 3.1 Subject-Level Group Comparison

| Metric | Healthy (n=70) | Ortho (n=35) | Neuro (n=111) | H vs Path. |
|--------|---------------|-------------|---------------|------------|
| **Stride |AI|** | **0.052 +/- 0.036** | **0.039 +/- 0.027** | **0.029 +/- 0.018** | **p<0.001\*\*\*, d=0.77** |
| **Stride |L-R| (s)** | **0.069 +/- 0.054** | **0.054 +/- 0.037** | **0.038 +/- 0.023** | **p<0.001\*\*\*, d=0.70** |
| Step |AI| | 0.148 +/- 0.091 | 0.088 +/- 0.075 | 0.118 +/- 0.093 | p=0.005\*\*, d=0.42 |
| Step CV (L) | 0.284 +/- 0.236 | 0.176 +/- 0.153 | 0.187 +/- 0.152 | p=0.002\*\*, d=0.55 |
| Mean step time (s) | 0.605 +/- 0.035 | 0.628 +/- 0.053 | 0.608 +/- 0.057 | p=0.19 ns |

**Stride |AI|** yields Cohen's d = 0.77 (large effect) — the strongest temporal gait discriminator identified. Walking speed (mean step time) does NOT differ between groups (p=0.19), ruling out speed as a confound.

![Stride |AI| by Group — Subject-Level Boxplot](figures/step06_asymmetry_boxplot_absAI.png)

*Figure 1. Unsigned stride asymmetry index (|AI|) across three groups and binary (healthy vs pathological). Healthy subjects exhibit significantly higher asymmetry magnitude (d=0.77).*

![Stride |L-R| Absolute Difference](figures/step06_asymmetry_boxplot_stride_abs_diff.png)

*Figure 2. Absolute stride time difference |L-R| in seconds. Confirms the same pattern: healthy gait has a larger, more consistent left-right timing difference (d=0.70).*

### 3.2 Interpretation

| | Healthy Gait | Pathological Gait |
|---|---|---|
| **Asymmetry pattern** | Consistent directional bias (L > R) | Reduced or disorganized |
| **Signed AI** | Positive (left stride slightly longer) | Near zero (direction unpredictable) |
| **Mechanism** | Natural dominant-leg lateralization | Loss of motor laterality |

This is a stronger finding than "more asymmetry = more pathology." It aligns with neuroscience literature on loss of motor lateralization in neurological conditions.

![Step Time Distribution by Side and Group](figures/step06_asymmetry_histogram.png)

*Figure 3. Left/right step time distributions by group. Healthy subjects show a clear directional L-R separation; pathological groups show overlapping distributions.*

![Example Trial: Gait Events on Gyroscope Signal](figures/step06_asymmetry_timeseries_example.png)

*Figure 4. Example trial showing left foot (blue) and right foot (orange) gyroscope Y-axis signals with heel-strike markers. The temporal structure of gait events is clearly visible.*

## 4. Pathology Subtype Analysis

Not all pathologies lose lateralization equally. Neurological conditions show the strongest asymmetry reduction, while ACL injury is indistinguishable from healthy.

| Subtype | Category | Stride |AI| mean | vs Healthy d | p-value |
|---------|----------|-------------------|--------------|---------|
| **RIL** (n=14) | Neuro | 0.024 | **0.87** | <0.001\*\*\* |
| **PD** (n=17) | Neuro | 0.025 | **0.77** | <0.001\*\*\* |
| **CVA** (n=44) | Neuro | 0.027 | **0.73** | <0.001\*\*\* |
| CIPN (n=36) | Neuro | 0.033 | 0.53 | 0.003\*\* |
| KOA (n=14) | Ortho | 0.036 | 0.45 | 0.034\* |
| HOA (n=12) | Ortho | 0.042 | 0.27 | 0.226 ns |
| ACL (n=9) | Ortho | 0.049 | 0.09 | 0.779 ns |

![Subtype Breakdown: Stride |AI|](figures/step07_subtype_stride_absAI.png)

*Figure 5. Stride |AI| broken down by pathology subtype. Green = healthy, blue = orthopedic, red = neurological. Neurological subtypes (RIL, CVA, PD) cluster at the lowest values.*

## 5. ROC Classifier & Clinical Correlation

### 5.1 Stride |AI| as a Screening Tool

| Metric | AUC | Threshold | Sensitivity | Specificity |
|--------|-----|-----------|-------------|-------------|
| Stride |AI| | **0.716** | 0.049 | 0.59 | **0.83** |
| Stride |L-R| | 0.703 | 0.065 | 0.56 | 0.84 |
| Step |AI| | 0.634 | 0.080 | 0.76 | 0.47 |

High specificity (83%) means few false positives — suitable for screening.

![ROC Curve: Stride |AI|](figures/step07_roc_stride_absAI.png)

*Figure 6. ROC curve for stride |AI| as a binary classifier (healthy vs pathological). AUC = 0.716 with an optimal Youden's J threshold at 0.049.*

### 5.2 Clinical Score Correlation

- **VGA vs stride |AI|**: Spearman rho = -0.206, p < 0.001\*\*\* (n=779)
- Higher gait severity (VGA score) is weakly but significantly associated with lower asymmetry magnitude.

![VGA vs Stride |AI| Correlation](figures/step07_corr_vga_stride_absAI.png)

*Figure 7. Scatter plot of Visual Gait Assessment score vs stride |AI|, colored by group. The negative correlation confirms that more severe gait impairment is associated with reduced temporal asymmetry.*

## 6. ML Feature Integration

### 6.1 Experiment Design

Merged 11 temporal asymmetry features with 217 existing sensor features (time-domain + FFT from 4 IMUs). Compared 5-fold StratifiedGroupKFold CV (grouped by subject) across four configurations.

### 6.2 Results (3-class: healthy / ortho / neuro)

| Configuration | Features | LR F1 | RF F1 | SVM F1 | SVM BAcc |
|---------------|----------|-------|-------|--------|----------|
| Sensor only | 217 | 0.748 | 0.810 | 0.816 | 0.809 |
| **Sensor + Asymmetry** | **228** | 0.741 | 0.810 | **0.822** | **0.815** |
| Feet + Asymmetry | 120 | 0.735 | 0.770 | 0.791 | 0.792 |
| Asymmetry only | 11 | 0.447 | 0.466 | 0.491 | 0.505 |

On the matched 974-trial subset (fair comparison): SVM improved from F1=0.806 to **0.812** (+0.007) and BAcc from 0.814 to **0.822** (+0.009).

![ML Comparison: Macro-F1](figures/step08_comparison_f1.png)

*Figure 8. Macro-F1 comparison across feature configurations. Asymmetry features alone achieve ~50% (above 33% chance), confirming discriminative signal. Combined sensor+asymmetry gives marginal SVM improvement.*

![RF Feature Importance: Sensor + Asymmetry](figures/step08_importance_sensor_plus_asym.png)

*Figure 9. Top-25 RF feature importances from the combined sensor+asymmetry model. Blue = sensor features, red = asymmetry features. Spectral features dominate; no asymmetry feature appears in top-25.*

### 6.3 Key Insight

Asymmetry features are **clinically interpretable biomarkers** (d=0.77, AUC=0.716) rather than ML performance boosters. The sensor waveform features already implicitly encode the temporal asymmetry information through their spectral characteristics.

## 7. Conclusions

1. **Stride |AI| is the strongest temporal discriminator** (d=0.77, p<0.001) between healthy and pathological gait
2. **Healthy gait is more asymmetric, not less** — reflecting consistent motor lateralization that is lost in disease
3. **Neurological conditions** (RIL d=0.87, PD d=0.77, CVA d=0.73) show the largest asymmetry reduction
4. **Stride |AI| achieves AUC=0.716** as a single-feature screening tool with 83% specificity
5. **ML integration provides marginal improvement** (+0.9% BAcc for SVM), confirming asymmetry's primary role as an explainability tool

---

# Part II: Chinese Report / 中文报告

## 1. 研究目标

调查步态病理是否表现为左右**时间不对称性**的变化。利用 IMU 步态数据中的足跟着地（Heel-Strike）事件，量化步行时间的左右差异，将教授的观察"左右 IMU 信号确实存在差异"转化为统计学验证的结论。

## 2. 方法概述

- 从 1356 次试验中提取足跟着地时间戳，排除 U 形转弯段，最终获得 **974 次有效试验**（216 名受试者）
- 计算**步长时间**（对侧足跟间隔）和**步幅时间**（同侧足跟间隔）
- 采用 5 种不对称性指标：有符号 AI、|AI|、|L-R| 绝对差、比值、变异系数 CV
- 按受试者聚合（每人取试验均值）避免伪重复
- Welch t 检验 + Cohen's d 效应量

## 3. 核心发现：健康步态反而更加不对称

这是一个反直觉但非常有力的发现：

### 3.1 受试者层面统计结果

| 指标 | 健康组 (n=70) | 骨科组 (n=35) | 神经组 (n=111) | 健康 vs 病理 |
|------|:---:|:---:|:---:|:---:|
| **步幅 |AI|** | **0.052 +/- 0.036** | **0.039 +/- 0.027** | **0.029 +/- 0.018** | **p<0.001, d=0.77** |
| **步幅 |L-R|** | **0.069 +/- 0.054** | **0.054 +/- 0.037** | **0.038 +/- 0.023** | **p<0.001, d=0.70** |
| 步长 |AI| | 0.148 +/- 0.091 | 0.088 +/- 0.075 | 0.118 +/- 0.093 | p=0.005, d=0.42 |
| 步长 CV (左) | 0.284 +/- 0.236 | 0.176 +/- 0.153 | 0.187 +/- 0.152 | p=0.002, d=0.55 |
| 平均步长时间 | 0.605 +/- 0.035 | 0.628 +/- 0.053 | 0.608 +/- 0.057 | p=0.19 ns |

**步幅 |AI|** 的 Cohen's d = 0.77（大效应量），是目前发现的最强时间步态区分指标。步行速度在组间无显著差异（p=0.19），排除了速度混杂因素。

![步幅 |AI| 组间对比](figures/step06_asymmetry_boxplot_absAI.png)

*图 1. 步幅不对称指数 |AI| 的三组对比及二分类（健康 vs 病理）。健康受试者的不对称性显著更高。*

### 3.2 机制解读

| | 健康步态 | 病理步态 |
|---|---|---|
| **不对称模式** | 一致的方向性偏差（左侧略长） | 方向不确定或消失 |
| **有符号 AI** | 正值（左步幅 > 右步幅） | 接近零（左右无规律） |
| **生理机制** | 自然的优势腿侧化 | 运动侧化丧失 |

> **健康步态 = 一致的、有方向性的时间不对称（自然侧化）**
>
> **病理步态 = 减弱或紊乱的时间不对称（侧化丧失）**

这一发现比"病理越重不对称越大"的传统假设更强、更具发表价值，与神经科学领域关于运动侧化丧失的文献一致。

![步长时间直方图](figures/step06_asymmetry_histogram.png)

*图 2. 按组别和左右侧分布的步长时间直方图。健康组呈现清晰的左右分离，病理组的分布重叠。*

## 4. 病理亚型分析

神经系统疾病的不对称性丧失最为显著，而 ACL 损伤与健康组无法区分：

| 亚型 | 类别 | 步幅 |AI| | vs 健康 d | p 值 |
|------|------|---------|---------|-------|
| **RIL** (n=14) | 神经 | 0.024 | **0.87** | <0.001 |
| **PD** (n=17) | 神经 | 0.025 | **0.77** | <0.001 |
| **CVA** (n=44) | 神经 | 0.027 | **0.73** | <0.001 |
| CIPN (n=36) | 神经 | 0.033 | 0.53 | 0.003 |
| KOA (n=14) | 骨科 | 0.036 | 0.45 | 0.034 |
| HOA (n=12) | 骨科 | 0.042 | 0.27 | 0.226 |
| ACL (n=9) | 骨科 | 0.049 | 0.09 | 0.779 |

![亚型分析：步幅 |AI|](figures/step07_subtype_stride_absAI.png)

*图 3. 按病理亚型分组的步幅 |AI|。绿色=健康，蓝色=骨科，红色=神经。神经亚型（RIL、CVA、PD）聚集在最低值区域。*

## 5. ROC 分类器与临床评分相关性

### 5.1 步幅 |AI| 作为筛查工具

- **AUC = 0.716**，最优阈值 0.049，灵敏度 59%，**特异度 83%**
- 高特异度意味着假阳性率低，适合作为初筛工具

![ROC 曲线](figures/step07_roc_stride_absAI.png)

*图 4. 步幅 |AI| 的 ROC 曲线。AUC = 0.716，Youden's J 最优阈值为 0.049。*

### 5.2 视觉步态评估（VGA）相关性

- Spearman rho = -0.206, p < 0.001 (n=779)
- 步态严重程度越高（VGA 分数越大），时间不对称性越低

![VGA 相关性](figures/step07_corr_vga_stride_absAI.png)

*图 5. VGA 评分与步幅 |AI| 的散点图。负相关证实了步态损伤越重，时间不对称性越弱。*

## 6. 机器学习特征集成

将 11 个时间不对称特征加入已有的 217 个传感器特征（4 个 IMU 的时域+频域统计量），5 折交叉验证对比：

| 特征配置 | 特征数 | SVM F1 | SVM BAcc |
|---------|--------|--------|----------|
| 仅传感器 | 217 | 0.816 | 0.809 |
| **传感器+不对称** | **228** | **0.822** | **0.815** |
| 仅不对称 | 11 | 0.491 | 0.505 |

匹配子集公平比较（974 trials）：SVM 从 BAcc 0.814 提升至 **0.822**（+0.9%）。

![ML 对比：Macro-F1](figures/step08_comparison_f1.png)

*图 6. 不同特征配置的 ML 分类性能对比。不对称特征单独可达 ~50%（高于 33% 随机），但组合后对 SVM 仅有微小提升。*

![RF 特征重要性](figures/step08_importance_sensor_plus_asym.png)

*图 7. 组合模型的 RF Top-25 特征重要性。蓝色=传感器特征，红色=不对称特征。频谱特征占主导地位。*

### 关键洞见

不对称特征的核心价值在于**临床可解释性**（d=0.77, AUC=0.716），而非 ML 预测性能提升。传感器波形的频谱特征已经隐式编码了时间不对称信息。**两者互补**：ML 流水线提供预测精度，不对称分析提供临床解释力。

## 7. 总结与结论

| # | 发现 | 意义 |
|---|------|------|
| 1 | 步幅 |AI| 是最强时间区分指标 (d=0.77) | 大效应量，统计显著 |
| 2 | 健康步态更不对称 | 反映自然运动侧化，病理组侧化丧失 |
| 3 | 神经亚型影响最大 (RIL d=0.87) | 可用于亚型鉴别 |
| 4 | 单特征 AUC=0.716, 特异度 83% | 可作为临床筛查工具 |
| 5 | ML 提升 +0.9% (SVM BAcc) | 不对称为可解释性工具，非 ML 主力 |
| 6 | 步行速度非混杂因素 (p=0.19) | 结果稳健 |

---

## Appendix: Files Generated

| Category | File | Description |
|----------|------|-------------|
| Code | `analysis/asymmetry.py` | Core asymmetry pipeline (~534 lines) |
| Code | `analysis/asymmetry_extended.py` | Subtype, ROC, correlation analysis |
| Code | `analysis/train_with_asymmetry.py` | ML feature integration |
| Data | `results/artifacts/asymmetry_per_trial.csv` | 974 trials, 25 columns |
| Data | `results/artifacts/asymmetry_per_subject.csv` | 216 subjects, aggregated |
| Data | `results/artifacts/asymmetry_ml_integration.json` | ML comparison results |
| Figures | `results/figures/step06_*.png` | 8 asymmetry visualizations |
| Figures | `results/figures/step07_*.png` | 12 extended analysis figures |
| Figures | `results/figures/step08_*.png` | 11 ML integration figures |
