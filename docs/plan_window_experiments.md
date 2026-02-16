# Window Experiments — Design Plan (Week 2)

This document captures the required decisions before implementation.

## Required 5 Items (to be explicit in report)
- Window length: <TBD ms/s>
- Rationale (physiology/signal): <TBD>
- Overlap: <TBD %>
- Rationale for overlap: <TBD>
- Physiological meaning per window: <TBD>

## Guardrails
- Split subjects first, then window (no leakage).
- Keep per-window labels from event/phase segmentation.
- Evaluate at least 2–3 window sizes with fixed 50% overlap as a starting point.

## Evaluation Table (to produce)
| Version  | Window | Overlap | Accuracy | Macro-F1 | BAcc |
|---------|--------|---------|----------|----------|------|
| Baseline| None   |   -     |   —      |    —     |  —   |
| W1      | L1     |  50%    |   —      |    —     |  —   |
| W2      | L2     |  50%    |   —      |    —     |  —   |
