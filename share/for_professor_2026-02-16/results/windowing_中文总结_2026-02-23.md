# Windowing 工作总结（截至 2026-02-23）

本文档基于项目内现有代码与结果文件整理（主要来源：`results/window_report.md`、`results/window_report_freq.md`、`results/window_experiments_summary*.csv`、`scripts/windowing.py`）。数值均为按受试者分组的 5 折交叉验证（StratifiedGroupKFold），度量以 Balanced Accuracy（BAcc）与 Macro-F1 为主。

## 1. 我们做了哪些工作
- 建立分阶段（pre_uturn / uturn / post_uturn / gait_full）的滑动窗口（windowing）流水线：窗口长度 3/4/5/6 s，重叠率 25% 与 50%。
- 每个窗口提取时域与频域特征（均值、标准差、RMS、主频、谱质心、总功率等），并生成按阶段的特征表。
- 评估多种传感器配置（单 IMU 与全传感器 ALL），重点考察单 IMU（RF，右脚）最小化方案。
- 采用受试者分组的 5 折 CV，避免数据泄漏；模型覆盖 LR / SVM / RF，并按 BAcc 选择最佳。
- 产出两条分支实验：
  - 时间域窗口实验（`results/window_report.md` 与 `window_experiments_summary.csv`）。
  - 时间域 + 频带增强窗口实验（`results/window_report_freq.md` 与 `window_experiments_freq_summary.csv`）。

## 2. 关键发现（时间域窗口）
- 各阶段最佳（单传感器 RF）：
  - pre_uturn：4 s @ 25% → BAcc≈0.900，Macro-F1≈0.872（LR）。
  - post_uturn：4 s @ 25% → BAcc≈0.890，Macro-F1≈0.794（LR）。
  - gait_full：3 s @ 50% → BAcc≈0.891，Macro-F1≈0.809（LR）。
  - uturn：3 s（25%/50%）→ BAcc≈0.600（时间域仍具挑战）。
- 传感器最小化：RF 单 IMU 与全传感器表现相当或更优（例如 gait_full：RF 0.891 vs ALL 0.884，二者选各自最优窗口）。
- 重叠率：gait_full 受益于 50% 重叠；pre/post_uturn 更偏好 25%；u-turn 对重叠不敏感（时间域）。

说明：`window_best_per_phase.csv` 给出 RF 的阶段最优窗口（pre/post 4 s，gait_full 5 s；BAcc≈0.868）。而 `window_experiments_summary.csv` 还显示 gait_full 在 3 s @ 50% 上达到更高的 BAcc≈0.891；两者差异 < 0.03，本总结以更高者（3 s @ 50%）为推荐。

## 3. 关键发现（时间域 + 频带增强）
- uturn 阶段显著改善：6 s @ 50%（LR）→ BAcc≈0.932（RF 与 ALL 均可达到 ≈0.932）。
- 其他阶段在当前设置下未明显优于时间域最佳（例如 pre/post/gait_full 的 BAcc 反而略低），但为 uturn 提供了清晰收益。

## 4. 与“未使用 windowing（整段/无窗）”对比
- 无窗整体基线（全传感器 SVM，整段特征）：BAcc≈0.809（见 `results/report_full.md` Baseline）。
- 相对该基线，时间域窗口的绝对提升（来自 `results/window_report.md` 的对照）：
  - pre_uturn：+0.099（→ ≈0.900）。
  - post_uturn：+0.081（→ ≈0.890）。
  - gait_full：+0.082（→ ≈0.891）。
  - uturn：≈−0.180（时间域窗口对 uturn 不占优）。
- 引入频带增强后（uturn 6 s @ 50%）：相对无窗基线 +0.123（0.932 − 0.809），显著优于整段特征与时间域窗口。

## 5. 结论与建议
- “短时多步”窗口（3–6 s）能更好捕捉局部步态节律与瞬态不对称，避免整段均值稀释信息，是 pre/post/gait_full 的最优策略。
- uturn 需要更强的角速度/转弯相关频域描述与更长上下文（6 s、50% 重叠、频带增强），显著改善分类。
- 部署优先级：
  - 若以单 IMU 部署为目标，推荐 RF + 时间域窗口：pre/post 用 4 s@25%，gait_full 用 3 s@50%。
  - 对 uturn 场景，开启频带增强并用 6 s@50%。
- 试验层级：可加入“窗口→试次”的投票汇聚（多数/软投票）以获得 trial 级对比无窗整段的公平评测。

## 6. 局限与后续工作
- 数据集外部泛化尚未验证；需做留一工况 / 留一传感器硬件 / 留一步行片段等鲁棒性测试。
- uturn 仍可探索：转向曲率、航向变化速率、峰值角速度、环节耦合等特征；尝试 8–10 s 窗口。
- 与步态事件对齐的“步周期同步窗口”（step-synchronous）与 0.5–3 Hz 以内更窄频带能进一步提升解释性。

## 7. 可复现性与产出
- 主要代码：`share/for_professor_2026-02-16/scripts/windowing.py`。
- 汇总文件：
  - `results/window_experiments_summary.csv`（时间域窗口汇总）。
  - `results/window_experiments_freq_summary.csv`（时间域+频带汇总）。
  - `results/window_report.md`、`results/window_report_freq.md`（可读报告）。
  - `results/window_best_per_phase.csv`（RF 最优窗口速览）。
- 复现实验（示例）：
  - `python analysis/window_experiments.py --windows 3.0,4.0,5.0,6.0 --overlap 0.50 --sensors RF,ALL --data dataset/data --out results`
  - `python analysis/window_experiments.py --windows 3.0,4.0,5.0,6.0 --overlap 0.25 --sensors RF,ALL --data dataset/data --out results`

---

## （更新）小窗口对比与“单一最佳传感器”结论（2026-02-23）

在既有评测基础上，补充对 1.0/1.28/2.56 s 小窗（以 50% 重叠为主）的比较与 RF vs ALL 的对照：

- 关键结果（RF 单 IMU vs ALL，BAcc 为主）：
  - gait_full（2.56 s @ 50%）：RF≈0.883 > ALL≈0.811；Macro‑F1：RF≈0.816 > ALL≈0.766。
  - pre_uturn（2.56 s @ 50%）：RF≈0.873 > ALL≈0.849；Macro‑F1：RF≈0.737 < ALL≈0.792（BAcc 主指标仍优）。
  - post_uturn（2.56 s @ 50%）：RF≈0.831 > ALL≈0.793；Macro‑F1：RF≈0.782 > ALL≈0.644。
  - uturn（时间域小窗）：2.56–3.0 s 的 BAcc≈0.587–0.600，仍明显低于“6.0 s @ 50% + 频带增强”的 ≈0.932（RF/ALL 均可达）。

- 窗口最优性（综合）：
  - pre_uturn / post_uturn：4.0 s @ 25% 仍是总体最佳；小窗 2.56 s 在 50% 重叠下优于 3.0 s，但未超越 4.0 s @ 25%。
  - gait_full：3.0 s @ 50% 仍在 BAcc 上最优（≈0.891）；2.56 s 次之但 Macro‑F1 略高（≈0.816）。
  - uturn：依赖更长窗口与频域特征（6.0 s @ 50% + 频带增强）。

- “单一最佳传感器（single best sensor）”更新结论：
  - 在小窗口（2.56 s）与既有 3–6 s 窗口的对比中，RF 作为单 IMU 在 BAcc 上多数情况下≥ALL；表现稳健，继续推荐作为部署的单一最佳传感器。
  - 例外/细节：pre_uturn 在 2.56 s 时 RF 的 Macro‑F1 低于 ALL，但 BAcc（主指标）更高；若更强调 Macro‑F1，可考虑阶段性地结合更多频域特征或做窗口级到试次级的投票汇聚。
  - uturn 阶段的关键不在于传感器数量，而在于更长时间上下文与频带描述；RF 与 ALL 在 6 s + 频带时表现一致（≈0.932）。

以上数值来源：`results/windows_features_*_{2560}ms.csv` 与 `results/windows/*/features_win{win}ms_ov50.csv` 的 5 折评测（与既有流程一致）。


---

# Windowing Summary – English Addendum (as of 2026-02-23)

This English addendum mirrors the Chinese summary above. Metrics are 5-fold subject-wise CV. Balanced Accuracy (BAcc) is the primary metric; Macro‑F1 is secondary.

## 1) What we did
- Phase-wise sliding windows (pre_uturn / uturn / post_uturn / gait_full), window lengths 3/4/5/6 s with 25% and 50% overlap.
- Per-window time + spectral features (mean/std/RMS, dominant frequency, spectral centroid, total power), structured feature tables per phase.
- Evaluated sensor configs focusing on single IMU (RF) vs ALL sensors.
- Subject-wise CV, models LR/SVM/RF; report best by BAcc.
- Two branches: time-only windows; time + frequency-band augmentation.

## 2) Key findings (time-only)
- pre_uturn (RF): 4 s @ 25% → BAcc≈0.900, Macro‑F1≈0.872 (LR)
- post_uturn (RF): 4 s @ 25% → BAcc≈0.890, Macro‑F1≈0.794 (LR)
- gait_full (RF): 3 s @ 50% → BAcc≈0.891, Macro‑F1≈0.809 (LR)
- uturn (RF/ALL): 3 s → BAcc≈0.600; still challenging with time-only features
- RF single-IMU is competitive with or better than ALL in many settings; overlap helps differently by phase (50% for gait_full; 25% for pre/post_uturn).

## 3) Key findings (time + frequency bands)
- uturn improves markedly: 6 s @ 50% (LR) → BAcc≈0.932 (RF and ALL both reach ≈0.932).
- Other phases show limited gains vs time-only under current settings.

## 4) Versus no-window (trial-level) baseline
- Baseline (ALL, SVM): BAcc≈0.809.
- Time-only window gains: pre_uturn +0.099; post_uturn +0.081; gait_full +0.082; uturn −0.18.
- With frequency bands (uturn 6 s @ 50%): +0.123 over baseline to ≈0.932.

## 5) Conclusions & recommendations
- 3–6 s windows capture multi-step, quasi-stationary gait patterns better than trial-level features.
- uturn needs longer context + frequency cues (6 s @ 50% + bands).
- Deployment (RF single-IMU): pre/post_uturn = 4 s @ 25%; gait_full = 3 s @ 50%; uturn = 6 s @ 50% with bands. For a single window across phases, 3 s @ 50% is a robust compromise.

## 6) Limitations & next steps
- External generalization not yet tested; consider leave-one-condition setups.
- For uturn: add turning curvature, heading change rate, peak angular velocity; try 8–10 s windows.
- Explore step-synchronous windows and narrow bands around cadence (0.5–3 Hz).

## 7) Reproducibility & artifacts
- Code: `share/for_professor_2026-02-16/scripts/windowing.py`.
- Summaries: `results/window_experiments_summary.csv`, `results/window_experiments_freq_summary.csv`.
- Reports: `results/window_report.md`, `results/window_report_freq.md`.
- Best-by-phase (RF): `results/window_best_per_phase.csv`.
- Rerun examples:
  - `python analysis/window_experiments.py --windows 3.0,4.0,5.0,6.0 --overlap 0.50 --sensors RF,ALL --data dataset/data --out results`
  - `python analysis/window_experiments.py --windows 3.0,4.0,5.0,6.0 --overlap 0.25 --sensors RF,ALL --data dataset/data --out results`

## (Update) Small windows + single best sensor (2026-02-23)
- RF vs ALL at 2.56 s @ 50% (BAcc first):
  - gait_full: RF≈0.883 (Macro‑F1≈0.816) > ALL≈0.811 (≈0.766)
  - pre_uturn: RF≈0.873 > ALL≈0.849; Macro‑F1 RF≈0.737 < ALL≈0.792
  - post_uturn: RF≈0.831 (≈0.782) > ALL≈0.793 (≈0.644)
- Window optimality: pre/post_uturn still prefer 4.0 s @ 25%; 2.56 s @ 50% can beat 3.0 s @ 50% but not 4.0 s @ 25%. gait_full: 3.0 s @ 50% best BAcc; 2.56 s close with slightly higher Macro‑F1.
- Single best sensor: RF remains the recommended single IMU. For uturn, the key is 6 s + bands, not adding sensors; RF and ALL both ≈0.932.

