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

## Leakage 处理说明（我们如何避免泄漏）
- 定义：训练阶段“看到了”验证/测试折信息的任何路径（含数据切分、预处理、特征、模型选择等）都会导致过高评估。
- 我们的做法：
  - 受试者分组切分：`StratifiedGroupKFold(n_splits=5, groups=subject_id)`，同一受试者的所有窗口（含多次试次、不同阶段）严格在同一折，杜绝“同人不同折”的泄漏。
  - 预处理在管线内：`Imputer/StandardScaler` 放在 `sklearn` Pipeline 里，按折拟合（只用训练折统计量），避免“全数据标准化”。
  - 特征提取逐窗进行：时/频特征由该窗口自身数据计算，不使用全数据统计或标签；相位分割依赖 trial 元数据而非标签分布。
  - 重叠窗口相关性：虽然 50% 重叠会带来相关样本，但因按受试者分组，不会出现同一受试者的相关窗口跨折分布，不构成泄漏。
- 残余风险与改进：跨模型对比（LR/SVM/RF）使用同一 CV 选择“最优模型”会有轻微选择偏倚（非泄漏）。如需更严谨，可用“嵌套CV”或预留独立留出受试者作为最终测试。

## 较小窗口的具体含义（面向患者人群）
- 研究对象为神经/骨科等患者，步速更慢、步态变异更大，步行周期（stride）往往长于健康人：约 1.33–2.0 s（60–90 步/分）。
- 在 fs≈100 Hz 下：
  - 1.00 s ≈ 0.5–0.75 个 stride；Δf≈1.00 Hz。优点：高时间分辨率，敏感于瞬时事件；风险：对慢步态不足一个完整周期，频域分辨率偏粗。
  - 1.28 s ≈ 0.64–0.96 个 stride；Δf≈0.78 Hz。更接近一个完整 stride（当步频较快时）。
  - 2.56 s ≈ 1.28–1.92 个 stride；Δf≈0.39 Hz。提供多步上下文且仍较“平稳”，是小窗中的更稳健选择。
  - 3.00 s ≈ 1.5–2.25 个 stride；在本数据上对 gait_full 最优（BAcc≈0.891）。
  - 4.00 s ≈ 2–3 个 stride；在 pre/post‑u‑turn 上最优（BAcc≈0.900/0.890）。
  - 6.00 s ≈ 3–4.5 个 stride；结合频带特征最利于 u‑turn（BAcc≈0.932）。
- 实务建议（患者）：窗口尽量覆盖≥1 个完整 stride；慢步态或步态波动大时，可取 2–3 个 stride 的窗口并配合 50% 重叠以增强鲁棒性；若需降低样本相关性，pre/post‑u‑turn 可选 25% 重叠。

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

## Leakage handling (how we avoid it)
- Definition: any path where training indirectly uses validation/test information (splits, preprocessing, features, model selection) inflates metrics.
- Our safeguards:
  - Subject-wise grouping: `StratifiedGroupKFold(..., groups=subject_id)` keeps all windows of a subject in a single fold; no cross-fold subject leakage.
  - In-fold preprocessing: `Imputer/StandardScaler` inside Pipeline fit on training folds only; no global normalization.
  - Per-window features only: time/spectral descriptors computed from each window itself; phase segmentation uses trial metadata, not labels.
  - Overlap: 50% overlap creates correlation, but since grouping is by subject, correlated windows never cross folds—so no leakage.
- Residual risk and remedy: choosing the best among LR/SVM/RF on the same CV can induce slight selection bias (not leakage). Use nested CV or a subject hold-out test set for final verification if needed.

## What small windows mean for patient cohorts
- Our participants are neurological/orthopedic patients with slower speed and higher variability; stride time is often 1.33–2.0 s (60–90 steps/min), longer than healthy adults.
- With fs≈100 Hz:
  - 1.00 s ≈ 0.5–0.75 strides; Δf≈1.00 Hz. High temporal resolution but may miss a full stride in slow gait; coarse spectral resolution.
  - 1.28 s ≈ 0.64–0.96 strides; Δf≈0.78 Hz. Closer to one full stride at moderate/fast cadence.
  - 2.56 s ≈ 1.28–1.92 strides; Δf≈0.39 Hz. Adds multi‑step context while preserving stationarity—more robust among small windows.
  - 3.00 s ≈ 1.5–2.25 strides; best for gait_full (BAcc≈0.891) in our data.
  - 4.00 s ≈ 2–3 strides; best for pre/post‑u‑turn (BAcc≈0.900/0.890).
  - 6.00 s ≈ 3–4.5 strides; with band features it is best for u‑turn (BAcc≈0.932).
- Practical rule (patients): ensure ≥1 full stride per window; for slow/variable gait, use 2–3‑stride windows with 50% overlap for robustness; use 25% in stationary pre/post‑u‑turn to reduce correlation.

## Reproducibility
- Additional evaluations used existing tables under `results/windows/*/features_win{win}ms_ov50.csv` and `results/windows_features_*_{2560}ms.csv`.
- Prior summaries: `results/window_experiments_summary.csv`, `results/window_report.md`, `results/window_report_freq.md`.

---

## 新增更新（2026-02-23 版，补充）
- 小窗口（1.0/1.28/2.56 s）在 50% 重叠下：
  - pre_uturn：2.56 s 在 BAcc 上优于 3.0 s，但总体仍不及 4.0 s @ 25% 最优；
  - post_uturn：2.56 s 同样优于 3.0/4.0 s @ 50%，但低于 4.0 s @ 25%；
  - gait_full：3.0 s @ 50% 仍最优，2.56 s 次之且 Macro‑F1 略高；
  - uturn：时间域小窗整体无优势；6.0 s @ 50% + 频带特征显著最优（BAcc≈0.932）。
- 研究对象为患者（非健康人）：窗口选择以“至少覆盖 1 个完整 stride，慢步态可取 2–3 个 stride + 50% 重叠”为原则；pre/post‑u‑turn 可降至 25% 重叠以降低相关性。
- Leakage 处理（要点）：按受试者分组 5 折；Imputer/StandardScaler 在 Pipeline 内按折拟合；特征逐窗计算；50% 重叠带来相关样本但不会跨折到同一受试者，不构成泄漏。模型选择偏倚≠泄漏，如需更严谨可用嵌套 CV 或留出受试者测试集。

### English Addendum
- Small windows (1.0/1.28/2.56 s, 50% overlap): 2.56 s beats 3.0 s in pre/post‑u‑turn but still trails 4.0 s @ 25%; 3.0 s @ 50% remains best for gait_full; uturn prefers 6.0 s @ 50% + frequency bands (BAcc≈0.932).
- Patient cohorts: ensure ≥1 full stride per window; for slow/variable gait use 2–3 strides + 50% overlap; for pre/post‑u‑turn consider 25% overlap to reduce correlation.
- Leakage handling: subject‑wise 5‑fold; in‑fold preprocessing; per‑window features; overlap does not leak across folds. Remaining risk is model‑selection bias; use nested CV or held‑out subjects if needed.


### Leakage handling — English (verbatim)
We use subject-wise StratifiedGroupKFold (5 folds) so that all windows from the same subject stay in a single fold; no cross-fold subject leakage. Preprocessing steps (imputation/standardization) live inside sklearn Pipelines and are fit on training folds only—no global normalization. Time/frequency features are computed per window from its own data; phase segmentation relies on trial metadata rather than labels. Although 50% overlap creates correlated samples, subject-wise grouping ensures correlated windows never cross folds, so this is not leakage. The only residual risk is model selection bias from comparing LR/SVM/RF/XGB on the same CV; if needed, use nested CV or a held-out set of subjects for the final estimate.
