# 单一最佳传感器（Single Best Sensor）更新 — RF vs ALL（2026-02-23）

## 主要发现（以受试者分组 5 折 CV，主指标 BAcc）
- gait_full（2.56 s @ 50%）：RF BAcc≈0.883、Macro‑F1≈0.816 > ALL BAcc≈0.811、Macro‑F1≈0.766。
- pre_uturn（2.56 s @ 50%）：RF BAcc≈0.873 > ALL≈0.849；但 RF Macro‑F1≈0.737 < ALL≈0.792（若主指标取 BAcc，RF 仍占优）。
- post_uturn（2.56 s @ 50%）：RF BAcc≈0.831、Macro‑F1≈0.782 > ALL BAcc≈0.793、Macro‑F1≈0.644。
- uturn：提升关键在更长上下文 + 频带特征（6 s @ 50% + 频域）；RF 与 ALL 在该设置下均可达 ≈0.932。

结论：在小窗与既有 3–6 s 设置下，RF 作为“单一最佳传感器”的结论进一步得到强化（多数阶段 BAcc≥ALL）。

---

# Single Best Sensor Update — RF vs ALL (2026-02-23)

## Key Findings (subject‑wise 5‑fold CV; primary metric = BAcc)
- gait_full (2.56 s @ 50%): RF BAcc≈0.883, Macro‑F1≈0.816 > ALL BAcc≈0.811, Macro‑F1≈0.766.
- pre_uturn (2.56 s @ 50%): RF BAcc≈0.873 > ALL≈0.849; RF Macro‑F1≈0.737 < ALL≈0.792 (if BAcc is primary, RF still preferred).
- post_uturn (2.56 s @ 50%): RF BAcc≈0.831, Macro‑F1≈0.782 > ALL BAcc≈0.793, Macro‑F1≈0.644.
- uturn: gains come from longer context + frequency features (6 s @ 50% + bands); both RF and ALL reach ≈0.932.

Conclusion: across small windows and 3–6 s setups, the single‑IMU RF choice is reinforced (BAcc ≥ ALL in most phases).
