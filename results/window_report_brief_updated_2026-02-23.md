# Windowing 简要更新报告（2026-02-23）

更新范围：在既有 3–6 s 窗口评测基础上，补充小窗口（1.0/1.28/2.56 s，当前以 50% overlap 为主）对比，仍采用 RF 单 IMU、受试者分组 5 折 CV，与既有汇总保持一致的建模与度量。

## 结论（哪种窗口更好）
- pre_uturn：总体最优仍为 4.0 s @ 25%（BAcc≈0.900，LR）。在 50% overlap 条件下，小窗 2.56 s（BAcc≈0.873，SVM）优于 3.0 s（BAcc≈0.847），但仍低于 4.0 s @ 25%。
- post_uturn：总体最优仍为 4.0 s @ 25%（BAcc≈0.890，LR）。在 50% overlap 下，2.56 s（BAcc≈0.831，LR）优于 3.0 s（0.784）与 4.0 s（0.820），但仍低于 4.0 s @ 25%。
- gait_full：3.0 s @ 50% 依然在 BAcc 上最优（≈0.891，LR）；2.56 s @ 50% 次之（BAcc≈0.883），但 Macro-F1 略高（≈0.816 vs 0.809）。
- uturn：时间域小窗无优势（2.56 s @ 50% BAcc≈0.587；3.0 s @ 50% ≈0.600）。频域增强下 6.0 s @ 50% 仍显著最优（BAcc≈0.932，LR）。

> 推荐：
> - 部署优先 RF 单 IMU：pre/post 仍用 4 s（25%）；gait_full 用 3 s（50%）；uturn 用 6 s（50%）并开启频带增强。
> - 若必须统一一个窗口：3 s @ 50%（RF）在 gait_full 最优，pre/post 也接近最优；配合阶段判定/汇聚可兼顾表现与效率。

## 关键验证数值（RF，5 折 CV）
- pre_uturn（50%）：2.56 s → BAcc≈0.873（SVM）；3.0 s → ≈0.847（LR）；对照总体最优：4.0 s@25% → ≈0.900（LR）。
- post_uturn（50%）：2.56 s → BAcc≈0.831（LR）；3.0 s → ≈0.784（LR）；4.0 s → ≈0.820（LR）；对照总体最优：4.0 s@25% → ≈0.890（LR）。
- gait_full（50%）：2.56 s → BAcc≈0.883、Macro-F1≈0.816（SVM）；3.0 s → BAcc≈0.891、Macro-F1≈0.809（LR）。
- uturn（50%，时间域）：2.56 s → BAcc≈0.587（LR）；3.0 s → ≈0.600（SVM）。
- uturn（50%，时频）：6.0 s → BAcc≈0.932（LR）。

## 解读与取舍
- 小窗口（≤2.56 s）提升了时间分辨率，能更敏感地捕捉短暂不对称，但在跨受试者分类时 BAcc 多数低于 3–4 s，因为频率分辨率与稳态步态上下文不足。
- 2.56 s 在 50% overlap 下对 post_uturn 与 pre_uturn 有一定优势（相对 3.0 s），但仍不及 4.0 s @ 25% 的总体最佳；gait_full 则 3.0 s 更稳健。
- uturn 需要更长上下文与频带信息（转向相关角速度/曲率）：6.0 s + 频带特征显著优于任意时间域小窗。

## 可复现性
- 新增评测基于现有特征表：`results/windows/*/features_win{win}ms_ov50.csv` 与 `results/windows_features_*_{2560}ms.csv`。
- 既有汇总：`results/window_experiments_summary.csv`、`results/window_report.md`、`results/window_report_freq.md`。


---

# Windowing Brief Update (English, 2026-02-23)

Scope: Extend prior 3–6 s window evaluations with smaller windows (1.0/1.28/2.56 s, mainly 50% overlap). Keep the same 5-fold subject-wise CV, RF single-IMU focus, models, and metrics (Balanced Accuracy as primary, Macro-F1 secondary).

## Which window is better?
- pre_uturn: Overall best remains 4.0 s @ 25% (RF, BAcc≈0.900, LR). With 50% overlap, 2.56 s (BAcc≈0.873, SVM) > 3.0 s (≈0.847), but still below 4.0 s @ 25%.
- post_uturn: Overall best remains 4.0 s @ 25% (RF, BAcc≈0.890, LR). With 50%, 2.56 s (≈0.831, LR) > 3.0 s (≈0.784) and 4.0 s (≈0.820), yet still below 4.0 s @ 25%.
- gait_full: 3.0 s @ 50% still best in BAcc (≈0.891, LR). 2.56 s @ 50% is second (BAcc≈0.883) but with slightly higher Macro‑F1 (≈0.816 vs 0.809).
- uturn: Small time-only windows do not help (2.56 s≈0.587; 3.0 s≈0.600). With frequency-band features, 6.0 s @ 50% remains clearly best (BAcc≈0.932, LR).

Recommendations:
- Preferred RF single-IMU setup: pre/post = 4 s @ 25%; gait_full = 3 s @ 50%; uturn = 6 s @ 50% + frequency bands.
- If one window must be used across phases: 3 s @ 50% (RF) is a robust compromise; add phase gating/aggregation if possible.

## Key numbers (RF, 5-fold CV)
- pre_uturn (50%): 2.56 s → BAcc≈0.873 (SVM); 3.0 s → ≈0.847 (LR). Best overall: 4.0 s @ 25% → ≈0.900 (LR).
- post_uturn (50%): 2.56 s → BAcc≈0.831 (LR); 3.0 s → ≈0.784 (LR); 4.0 s → ≈0.820 (LR). Best overall: 4.0 s @ 25% → ≈0.890 (LR).
- gait_full (50%): 2.56 s → BAcc≈0.883, Macro‑F1≈0.816 (SVM); 3.0 s → BAcc≈0.891, Macro‑F1≈0.809 (LR).
- uturn (50%, time-only): 2.56 s → BAcc≈0.587 (LR); 3.0 s → ≈0.600 (SVM).
- uturn (50%, time+freq): 6.0 s → BAcc≈0.932 (LR).

## Interpretation
- Short windows (≤2.56 s) increase temporal resolution but often underperform 3–4 s for cross-subject BAcc due to coarser frequency resolution and less steady gait context.
- 2.56 s @ 50% can beat 3.0 s @ 50% in pre/post_uturn but still falls short of 4.0 s @ 25% overall; 3.0 s @ 50% is most stable for gait_full.
- uturn benefits from longer context + frequency bands; 6.0 s + bands clearly superior to any time-only small window.

## Reproducibility
- Additional evaluations used existing tables under `results/windows/*/features_win{win}ms_ov50.csv` and `results/windows_features_*_{2560}ms.csv`.
- Prior summaries: `results/window_experiments_summary.csv`, `results/window_report.md`, `results/window_report_freq.md`.

