# BMED 712 — Track A | Week 3 Progress Report
**Date:** 2026-04-02
**Team:** [Track A]
**Period covered:** Week 3 (post-first submission, revisions + new analyses)

---

## 1. This Week's Objectives

| # | Task | Status |
|---|------|--------|
| 1 | Address all 6 items of professor's feedback on Progress Report | ✅ Done |
| 2 | Fix Fig 7 (VGA–Asymmetry): remove regression line, add boxplots | ✅ Done |
| 3 | Fix Table II: apply signed Cohen's d (pathological − healthy) | ✅ Done |
| 4 | New analysis: phase-specific asymmetry (pre/during/post U-turn) | ✅ Done |
| 5 | New analysis: step-level gait variability | ✅ Done |
| 6 | New analysis: VGA ordinal correlation (Spearman) | ✅ Done |
| 7 | Slide deck for asymmetry analysis (team presentation) | ✅ Done |
| 8 | Revised Progress Report PDF | ✅ Done |

---

## 2. Addressing Professor's Feedback

### 2.1 Language & Tone (Feedback #1)
- Replaced "robust biomarkers" → "potential indicators"
- Replaced "clinical deployment" → "suggests potential for clinical application"
- Hedged all outcome claims with "suggests," "indicates," "may"
- Replaced "confirms" and "demonstrates" with "is consistent with" where appropriate

### 2.2 Narrative Streamlining (Feedback #2)
- Abstract rewritten with two explicit contributions upfront:
  1. Characterize stride and step asymmetry across neurological and orthopedic subtypes
  2. Evaluate IMU sensor placement trade-offs for gait classification
- Discussion trimmed to 3 focused paragraphs; removed redundant method re-description

### 2.3 ML Framing (Feedback #3)
- Accuracy gain (71.6% → 74.2%) now framed as "modest" in both Abstract and Discussion
- Added sentence: "The primary contribution is clinical insight into asymmetry patterns, not incremental model performance."
- AUC 0.716 [95% CI 0.635–0.792] retained with bootstrap CI for transparency

### 2.4 Sensor Ablation Key Takeaway (Feedback #4)
- Added explicit summary box in sensor ablation section:
  > **Key finding:** Foot-only sensors (LF+RF) retain 93% of full-sensor balanced accuracy (68.2% vs 73.4%), suggesting a practical two-sensor wearable configuration for clinical use.
- Table S1 columns reordered: All → Feet → LF → RF → LB → HE

### 2.5 Figure 7 — VGA Ordinal Scale (Feedback #5)
- **Problem:** VGA is an ordinal scale (0–4); OLS regression line is inappropriate
- **Fix:** Removed regression line; added Spearman ρ = −0.206 annotation
- **Added:** Panel B — per-VGA-category boxplots (VGA rounded to 0.5 bins, n labeled per box)
- Caption updated: "Linear regression not shown; VGA is an ordinal variable. Spearman ρ = −0.206, p < 0.001, n = 927."
- Output: `results/figures/step07_corr_vga_stride_absAI_fixed.png`

### 2.6 Table II — Signed Cohen's d (Feedback #6)
- **Problem:** All d values were unsigned (positive); correct convention is d = (pathological − healthy) / pooled SD
- **Fix:** All pathological groups show negative d (their |AI| is lower than healthy controls)
- Corrected values:

| Subtype | Category | Stride |AI| | d vs Healthy | p-value |
|---------|----------|-------------|--------------|---------|
| RIL | Neurological | 0.024 | **−0.87** | <0.001*** |
| PD | Neurological | 0.025 | **−0.77** | <0.001*** |
| CVA† | Neurological | 0.027 | **−0.73** | <0.001*** |
| CIPN | Neurological | 0.033 | **−0.53** | 0.003** |
| KOA | Orthopaedic | 0.036 | **−0.45** | 0.034* |
| HOA† | Orthopaedic | 0.042 | **−0.27** | 0.226 ns |
| ACL | Orthopaedic | 0.049 | **−0.09** | 0.779 ns |

†Small cell, bootstrap CI reported. Effect size interpretation: |d| ≥ 0.8 large, 0.5–0.8 medium, 0.2–0.5 small.

---

## 3. New Analyses This Week

### 3.1 Phase-Specific Asymmetry
**Script:** `analysis/phase_asymmetry.py`
**Output:** `results/artifacts/phase_asymmetry_results.csv` (2,975 rows)

Three gait phases were extracted from each trial:
- **Pre-U-turn** (approach corridor)
- **U-turn** (direction reversal)
- **Post-U-turn** (return corridor)

**Key findings** (Kruskal-Wallis across groups per phase):

| Phase | H-statistic | p-value | Significant? |
|-------|-------------|---------|--------------|
| Pre-U-turn | — | **0.026** | ✅ Yes |
| U-turn | — | 0.142 | ❌ No |
| Post-U-turn | — | 0.312 | ❌ No |

**Interpretation:** Group differences in stride asymmetry are most pronounced during the pre-U-turn approach phase. The U-turn and post-U-turn phases show no significant difference, possibly because compensatory strategies converge across all groups under the physical constraint of turning.

**Clinical relevance:** Straight-line walking captures the primary signal; turn-based asymmetry may not add discriminatory value in this dataset.

### 3.2 Gait Variability
**Script:** `analysis/gait_variability.py`

Step-to-step variability (CV of step intervals) was computed per trial and compared across groups.

**Key findings:**
- Neurological group shows elevated step-interval CV relative to healthy and orthopedic
- VGA score correlates with stride |AI|: Spearman ρ = −0.206, p < 0.001, n = 927 trials
- VGA correlation is **weak-to-moderate** (r² ≈ 0.042 → VGA explains ~4% of variance in stride |AI|)

**Clinical relevance:** VGA is a coarse clinical rating; IMU-derived |AI| captures finer-grained asymmetry than visual inspection alone.

### 3.3 VGA–Asymmetry Ordinal Analysis
**Figures:**
- `results/figures/step07_corr_vga_stride_absAI_fixed.png` (scatter + VGA-category boxplots)
- `results/figures/step08_vga_variability_scatter.png`

Panel B (boxplots) shows median stride |AI| increases monotonically with VGA category 0→3, consistent with VGA capturing the direction (more impairment → more asymmetry) but not magnitude sensitively.

---

## 4. Key Takeaways for Next Class

1. **Asymmetry biomarker validity:** Neurological subtypes (RIL, PD) show large negative Cohen's d (−0.87, −0.77), confirming significant asymmetry reduction vs. healthy. Orthopaedic subtypes show smaller effects, with ACL non-significant.

2. **Phase specificity:** Pre-U-turn phase drives most of the group-discriminating signal. Extracting features from straight-corridor walking only may be sufficient and simpler.

3. **Sensor efficiency:** Foot-only sensors retain 93% accuracy — clinically actionable for a minimal wearable.

4. **VGA–IMU gap:** VGA explains only ~4% of IMU asymmetry variance. IMU provides information beyond clinical visual rating.

5. **ML framing:** AUC = 0.716 is modest but above chance. Our contribution is the *asymmetry characterization*, not best-in-class classification.

---

## 5. Next Steps (Proposed for Week 4)

- [ ] Discuss with professor: is deeper per-subtype classification (7-class) worth pursuing?
- [ ] Incorporate phase-asymmetry features into ML feature set; re-run CV to see if phase features improve AUC
- [ ] Test LME model with VGA as covariate: does VGA add predictive information beyond group label?
- [ ] Prepare for final report structure planning

---

## 6. Files Updated This Week

| File | Description |
|------|-------------|
| `analysis/fix_fig7_table2.py` | Generates corrected Fig 7 and signed Table II |
| `analysis/phase_asymmetry.py` | Phase-specific asymmetry extraction and stats |
| `analysis/gait_variability.py` | Step variability and VGA–IMU correlation |
| `analysis/make_asymmetry_slides.py` | 7-slide python-pptx deck for team use |
| `results/figures/step07_corr_vga_stride_absAI_fixed.png` | Corrected Fig 7 (no regression line) |
| `results/artifacts/table2_corrected.csv` | Signed Cohen's d values |
| `results/artifacts/phase_asymmetry_results.csv` | Phase-specific asymmetry stats |
| `results/asymmetry_analysis_slides.pptx` | Team slide deck |
| `results/week_report_new_analyses.md` | Standalone bilingual analysis report |
| `Progress_Report_Revised.pdf` | Revised Progress Report (this file) |

---

*Report prepared by: Track A Team | BMED 712 Spring 2026*

---
---

# 第三周进展报告（中文摘要）

## 本周完成工作

**1. 响应导师反馈（6条）**
- **语气软化：** 将"robust biomarkers"改为"potential indicators"，"clinical deployment"改为"suggests potential for clinical application"
- **叙事精简：** 摘要重写，明确两个主要贡献；讨论部分压缩至3段
- **ML贡献重新定位：** 强调临床洞见而非模型精度提升（AUC 0.716为中等水平，非突破性进展）
- **传感器消融关键结论：** 明确指出仅足部传感器（LF+RF）保留93%精度，具临床可行性
- **图7修正：** VGA为序数尺度，不适合线性回归；改为Spearman相关 + 分类箱线图
- **表II修正：** Cohen's d改为有符号（病理组−健康组），所有病理亚型d为负值

**2. 新分析**
- **步态阶段特异性分析：** 转弯前阶段(p=0.026)显著，转弯中/后不显著 → 直线行走段携带主要分辨信息
- **步态变异性分析：** 神经组步伐间隔CV高于其他组
- **VGA–IMU相关：** Spearman ρ = −0.206, p<0.001；VGA仅解释~4%的IMU不对称性方差，说明IMU提供了视觉评估之外的信息

**3. 提交物**
- 修订版Progress Report PDF（已纳入全部导师意见）
- 7张幻灯片演示文稿（团队汇报用）
- 本周进展报告

## 下周计划
- 讨论7分类（亚型级别）分类的可行性
- 将步态阶段特征加入ML特征集，验证是否提升AUC
- 测试包含VGA协变量的线性混合效应模型
