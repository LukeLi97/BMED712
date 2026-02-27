# Weekly Update — Windowing/Overlap and Tiny Mamba (English)

Audience: course instructor. This report consolidates what we did this week on time/frequency windowing and the non‑overlapping (0%) check, plus a brief status of the Tiny Mamba runs on Mac (MPS). Links and figures point to reproducible artifacts in the repo.

## Executive Summary
- Non‑overlapping windows (0%) confirm robustness of our earlier conclusions. For matched 3.0 s setups, Balanced Accuracy remains essentially the same (pre_uturn RF: 0.849 vs 0.847@50%), small drops for gait_full (3.0 s: 0.864 vs 0.891@50%), and larger drops at 4.0 s in pre/post (−0.09/−0.08) due to fewer non‑overlapping samples; however the phase‑wise window preferences are unchanged.
- Frequency windowing (bands + filterbank) on top of time windows brings a clear gain only for uturn: 6.0 s @ 50% with bands achieves BAcc ≈ 0.932 (RF/ALL), well above time‑only small windows; other phases do not surpass their time‑only best (pre/post 4 s @ 25%; gait_full 3 s @ 50%).
- Tiny‑Mamba (fallback GRU on Mac MPS) — early sanity: gait_full 3 s @ 50% yields BAcc ≈ 0.663 in a 1‑epoch run. We will proceed to 12‑epoch subject‑wise CV after the window/overlap report is finalized.

## What we did
1) Time windowing review and 0% overlap re‑run across phases (pre_uturn, uturn, post_uturn, gait_full) and sensors (RF/ALL). Subject‑wise StratifiedGroupKFold; preprocessing inside Pipelines; per‑window features computed from its own data.
2) Frequency windowing on the same time windows (bands and filterbank features computed inside each time window; no separate segmentation). Key focus on uturn with 6.0 s @ 50%.
3) Implemented/validated scripts for reproducibility and updated figures/reports in `results/`.

## Key Findings
- Phase‑wise best windows (time‑only) remain:
  - pre_uturn: 4.0 s @ 25%
  - post_uturn: 4.0 s @ 25%
  - gait_full: 3.0 s @ 50%
  - uturn: 6.0 s @ 50% + frequency features
- 0% overlap matched comparisons (BAcc):
  - pre_uturn 3.0 s (RF): 0.849 vs 0.847@50% → essentially the same.
  - gait_full 3.0 s (RF): 0.864 vs 0.891@50% → small drop, conclusion holds.
  - post_uturn 4.0 s (RF): 0.807 vs 0.890@25% → larger drop but trend unchanged (4 s still better than shorter windows; fewer non‑overlapping windows increase variance).
  - uturn small windows at 0% are unstable; validated best is still 6.0 s @ 50% with bands.
- Time vs Time+Frequency (best per phase, RF): only uturn benefits strongly (≈0.932).

## Visuals
- Non‑overlapping vs overlapped (examples):
  - pre_uturn RF:  
    ![pre_uturn RF](/Users/test/Desktop/BMED712 Rehab/BMED712 Project 1_Track A/results/figures/overlap_bacc_quick_pre_uturn_RF.png)
  - pre_uturn ALL:  
    ![pre_uturn ALL](/Users/test/Desktop/BMED712 Rehab/BMED712 Project 1_Track A/results/figures/overlap_bacc_quick_pre_uturn_ALL.png)
  - uturn RF:  
    ![uturn RF](/Users/test/Desktop/BMED712 Rehab/BMED712 Project 1_Track A/results/figures/overlap_bacc_quick_uturn_RF.png)
  - uturn ALL:  
    ![uturn ALL](/Users/test/Desktop/BMED712 Rehab/BMED712 Project 1_Track A/results/figures/overlap_bacc_quick_uturn_ALL.png)
- Time‑only vs Time+Frequency (RF, best per phase):  
  ![Time vs Time+Frequency](/Users/test/Desktop/BMED712 Rehab/BMED712 Project 1_Track A/results/figures/phase_time_vs_timefreq_rf.png)

## Leakage handling (recap)
We use subject‑wise StratifiedGroupKFold (5 folds) so that all windows from the same subject stay in a single fold; no cross‑fold subject leakage. Preprocessing steps (imputation/standardization) live inside sklearn Pipelines and are fit on training folds only—no global normalization. Time/frequency features are computed per window from its own data; phase segmentation relies on trial metadata rather than labels. Although 50% overlap creates correlated samples, subject‑wise grouping ensures correlated windows never cross folds; this is not leakage. The only residual risk is model‑selection bias (same CV used to compare LR/SVM/RF/XGB); a nested CV or held‑out subjects set can be used for the final estimate if needed.

## Tiny Mamba (status)
- Code: `analysis/train_mamba_windows.py` (dataset/model under `analysis/datasets` and `analysis/models`). `mamba-ssm` is not available on this Mac; the model falls back to a tiny GRU backbone with the same interface.
- Current numbers: gait_full 3 s @ 50% (1 epoch) — BAcc≈0.663, Macro‑F1≈0.649; file: `results/artifacts/metrics_mamba_gait_full_3000ms_ov50.json`.
- Next: 12‑epoch subject‑wise CV for gait_full → pre_uturn → post_uturn → uturn (MPS), with logs/artifacts under `results/artifacts/*` and a comparative table added to the main report.

## Reproducibility and Artifacts
- 0% overlap comparison report: `results/window_overlap_0_report.md`
- Frequency windowing report: `results/window_report_freq.md`
- 0% matched table: `results_ov0/overlap_matched_compare.csv`
- Frequency sweep summary: `results/window_experiments_freq_summary.csv`

---

# 每周更新——窗口/重叠与 Tiny Mamba（中文）

面向老师的阶段性汇报。本周完成了时间/频率窗口的系统复核与 0% 重叠验证，并给出 Tiny Mamba 在 Mac（MPS）上的初步结果。

## 核心结论
- 0% overlap 结果总体与既有结论一致：与 3.0 s 匹配时，pre_uturn 基本不变，gait_full 小幅下降；4.0 s 的 pre/post 在 0% 下下降更明显，但“4 s 更优”的趋势未改变（非重叠样本更少、方差更大）。
- 频率特征只在 uturn 明显提升：6.0 s @ 50% + 频带/滤波银行 BAcc≈0.932；其他阶段不超过各自最佳时间窗（pre/post 4 s@25%，gait_full 3 s@50%）。
- Tiny‑Mamba（GRU 回退）1‑epoch 验证 BAcc≈0.663（gait_full 3 s@50%）。

## 本周具体做了什么
1) 统一以时间窗切分（3/4/5/6 s；25%/50%/0%），在同一时间窗内同时计算时间域与频域特征；0% 作为稳健性验证。  
2) 频率窗口化（bandpower + filterbank）重点覆盖 uturn 6 s@50%，并与 Time‑only 做对照。  
3) 生成/更新可复现实验表与图片，写入 `results/` 下对应报告。

## 关键发现（数值摘录）
- pre_uturn 3.0 s（RF）：0%≈0.849 vs 50%≈0.847（几乎相同）。
- gait_full 3.0 s（RF）：0%≈0.864 vs 50%≈0.891（小幅下降）。
- post_uturn 4.0 s（RF）：0%≈0.807 vs 25%≈0.890（下降较多，但“4 s 更优”趋势不变）。
- uturn：0% 小窗不稳定；最佳为 6.0 s@50% + 频域（BAcc≈0.932）。

## 泄漏处理（复述）
受试者分组 5 折；预处理置于 Pipeline 并在折内拟合；特征逐窗计算；重叠不跨受试者，不构成泄漏。剩余风险为模型选择偏倚，可用嵌套 CV 或留出受试者测试集进一步收敛估计。

## Tiny Mamba（现状与计划）
- 现状：Mac 无 `mamba-ssm`，自动回退 GRU 小模型；gait_full 3 s@50% 的 1‑epoch 基线 BAcc≈0.663。  
- 计划：在 Window/Overlap 报告定稿后，按 gait_full → pre → post → uturn 启动 12‑epoch 训练，并在总报告中加入端到端模型对照表与混淆矩阵。

## 可视化
- 非重叠 vs 重叠（示例）：
  - pre_uturn RF：  
    ![pre_uturn RF](/Users/test/Desktop/BMED712 Rehab/BMED712 Project 1_Track A/results/figures/overlap_bacc_quick_pre_uturn_RF.png)
  - pre_uturn ALL：  
    ![pre_uturn ALL](/Users/test/Desktop/BMED712 Rehab/BMED712 Project 1_Track A/results/figures/overlap_bacc_quick_pre_uturn_ALL.png)
  - uturn RF：  
    ![uturn RF](/Users/test/Desktop/BMED712 Rehab/BMED712 Project 1_Track A/results/figures/overlap_bacc_quick_uturn_RF.png)
  - uturn ALL：  
    ![uturn ALL](/Users/test/Desktop/BMED712 Rehab/BMED712 Project 1_Track A/results/figures/overlap_bacc_quick_uturn_ALL.png)
- 时间 vs 时频（RF，按相位最优）：  
  ![Time vs Time+Frequency](/Users/test/Desktop/BMED712 Rehab/BMED712 Project 1_Track A/results/figures/phase_time_vs_timefreq_rf.png)

## 复现与数据对象
- 0% 报告：`results/window_overlap_0_report.md`  
- 频率报告：`results/window_report_freq.md`  
- 0% 对齐表：`results_ov0/overlap_matched_compare.csv`  
- 频率汇总表：`results/window_experiments_freq_summary.csv`

