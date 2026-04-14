#!/usr/bin/env python3
"""
Generate final presentation using KU PPT template.
4 presenters x ~3 slides each = 12 content slides.
Heavy visuals, minimal text (60-90s video per slide).
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor
from pathlib import Path

ROOT = Path("/Users/test/Desktop/BMED712 Rehab")
TEMPLATE = ROOT / "KU_ppt_template_2018.pptx"
OUTPUT = ROOT / "results" / "Final_Presentation_Track_A.pptx"

# Figures
FIGS = {
    "cohort":     ROOT / "results/validation/cohort_balance.png",
    "confusion":  ROOT / "results/figures/step03_confusion_3class_all.png",
    "frontier":   ROOT / "results/figures/step04_sensors_frontier.png",
    "importance": ROOT / "results/figures/step05_importance_3class_all.png",
    "vga":        ROOT / "results/figures/step07_corr_vga_stride_absAI_fixed.png",
    "phase":      ROOT / "results/figures/phase_single_vs_all.png",
    "heatmap":    ROOT / "PHASE1_Feature_Analysis_COMPLETE/Full_Gait_6s_ov50/correlation_heatmap_Full_Gait_6s_ov50.png",
}

# KU brand colors
KU_BLUE  = RGBColor(0x00, 0x47, 0xBA)
KU_GREEN = RGBColor(0x00, 0xCE, 0x7C)
KU_RED   = RGBColor(0xE5, 0x3E, 0x51)
KU_TEAL  = RGBColor(0x84, 0xDA, 0xDE)
KU_GRAY  = RGBColor(0x96, 0x96, 0x9A)
WHITE    = RGBColor(0xFF, 0xFF, 0xFF)
BLACK    = RGBColor(0x33, 0x33, 0x33)
LIGHT_BG = RGBColor(0xF0, 0xF7, 0xFF)
LIGHT_RED_BG = RGBColor(0xFF, 0xF0, 0xF0)

# Layout indices (from template analysis)
LY_TITLE   = 1   # Title Slide Nega (dark background)
LY_SECTION = 14  # Section Header
LY_CONTENT = 11  # Title Only (blank canvas with KU header/footer)
LY_CONTENT2 = 12 # Title Only 2
LY_END     = 16  # End slide


def tb(slide, left, top, width, height, text,
       sz=18, bold=False, color=BLACK, align=PP_ALIGN.LEFT):
    """Add a text box and return its text_frame."""
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(sz)
    run.font.bold = bold
    run.font.color.rgb = color
    return tf


def bullets(slide, left, top, width, height, items, sz=14, color=BLACK):
    """Add bullet list text box."""
    box = slide.shapes.add_textbox(left, top, width, height)
    tframe = box.text_frame
    tframe.word_wrap = True
    for i, item in enumerate(items):
        p = tframe.paragraphs[0] if i == 0 else tframe.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(4)
        # Handle bold prefix with ": "
        if ": " in item and not item.startswith(" "):
            label, rest = item.split(": ", 1)
            r1 = p.add_run()
            r1.text = label + ": "
            r1.font.size = Pt(sz)
            r1.font.bold = True
            r1.font.color.rgb = color
            r2 = p.add_run()
            r2.text = rest
            r2.font.size = Pt(sz)
            r2.font.color.rgb = color
        else:
            r = p.add_run()
            r.text = item
            r.font.size = Pt(sz)
            r.font.color.rgb = color
    return tframe


def card(slide, left, top, width, height, fill_color):
    """Add a colored rectangle card."""
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.fill.background()
    return shape


def img(slide, path, left, top, width, height=None):
    """Add an image, optionally auto-scaling height from aspect ratio."""
    if height is None:
        from PIL import Image
        with Image.open(path) as im:
            iw, ih = im.size
        aspect = iw / ih
        height = Inches(width / 914400 / aspect) if isinstance(width, int) else Inches((width / Inches(1)) / (iw / ih))
    return slide.shapes.add_picture(str(path), left, top, width, height)


def caption(slide, left, top, width, text):
    """Small gray caption."""
    tb(slide, left, top, width, Inches(0.35), text, sz=10, color=KU_GRAY, align=PP_ALIGN.CENTER)


def presenter_tag(slide, num, topic):
    """Small tag at bottom."""
    tb(slide, Inches(0.5), Inches(6.85), Inches(5), Inches(0.35),
       f"Presenter {num}  |  {topic}", sz=10, color=KU_GRAY)


def title_ph(slide, text):
    """Set the title placeholder text."""
    for ph in slide.placeholders:
        if ph.placeholder_format.idx == 0:
            ph.text_frame.clear()
            p = ph.text_frame.paragraphs[0]
            r = p.add_run()
            r.text = text
            r.font.bold = True
            r.font.size = Pt(28)
            r.font.color.rgb = KU_BLUE
            return


# ══════════════════════════════════════════════════════
# BUILD
# ══════════════════════════════════════════════════════
def build():
    prs = Presentation(str(TEMPLATE))

    # Delete all 12 template example slides
    sld_ids = list(prs.slides._sldIdLst)
    for sid in sld_ids:
        rId = sid.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
        prs.slides._sldIdLst.remove(sid)
        if rId:
            prs.part.drop_rel(rId)

    # ──── SLIDE 0: TITLE (cover) ────
    s = prs.slides.add_slide(prs.slide_layouts[LY_TITLE])
    for ph in s.placeholders:
        idx = ph.placeholder_format.idx
        if idx == 0:
            ph.text = "IMU-Based Gait Pathology Classification"
        elif idx == 1:
            ph.text = "Using the GaitRec Dataset with Machine Learning"
        elif idx == 10:
            ph.text = "BMED 712 | April 2026"
    print("  [0] Title")

    # ──── SLIDE 1: SECTION - Introduction ────
    s = prs.slides.add_slide(prs.slide_layouts[LY_SECTION])
    for ph in s.placeholders:
        idx = ph.placeholder_format.idx
        if idx == 0:
            ph.text = "Introduction & Dataset"
        elif idx == 1:
            ph.text = "Presenter 1  (Slides 1-3)"
        elif idx == 11:
            ph.text = "BMED 712 Rehabilitation Engineering"
    print("  [1] Section: Introduction")

    # ──── SLIDE 2: Problem & Dataset (P1-S1) ────
    s = prs.slides.add_slide(prs.slide_layouts[LY_CONTENT])
    title_ph(s, "Problem & Dataset Overview")

    # Big stat callouts
    stats = [
        (0.5,  "260",     "Subjects (8 Cohorts)", KU_BLUE),
        (4.5,  "1,356",   "Walking Trials",       KU_GREEN),
        (8.5,  "300,991", "Feature Windows",       KU_RED),
    ]
    for x, num, label, clr in stats:
        card(s, Inches(x), Inches(1.6), Inches(3.6), Inches(1.6), clr)
        tb(s, Inches(x), Inches(1.7), Inches(3.6), Inches(0.9),
           num, sz=48, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        tb(s, Inches(x), Inches(2.6), Inches(3.6), Inches(0.4),
           label, sz=14, color=WHITE, align=PP_ALIGN.CENTER)

    # Cohort balance figure
    img(s, FIGS["cohort"], Inches(0.5), Inches(3.6), Inches(7.0), Inches(2.0))
    caption(s, Inches(0.5), Inches(5.7), Inches(7.0),
            "Class distribution: HC / Mild / Severe across 8 cohorts")

    # Key info
    bullets(s, Inches(8.0), Inches(3.6), Inches(4.5), Inches(2.5), [
        "GaitRec dataset (Medical Univ. Innsbruck)",
        "4 IMU sensors: HE, LB, LF, RF",
        "3 signals: FreeAcc, Gyr, Mag",
        "3 axes: X (ML), Y (AP), Z (Vert)",
        "6 metrics per channel = 216 features",
        "3-class: HC / Mild / Severe",
    ], sz=13)

    presenter_tag(s, 1, "Dataset Overview")
    print("  [2] Problem & Dataset")

    # ──── SLIDE 3: Feature Extraction (P1-S2) ────
    s = prs.slides.add_slide(prs.slide_layouts[LY_CONTENT])
    title_ph(s, "Feature Extraction Pipeline")

    # Pipeline flow: 4 colored cards
    pipeline = [
        ("Windowing",    "6s windows\n50% overlap",          KU_BLUE),
        ("Extraction",   "4 sensors x 3ch\nx 3 axes",       KU_GREEN),
        ("Metrics",      "Mean, Std, Min\nMax, DomF, SC",   KU_TEAL),
        ("Selection",    "KW H-test\n\u03b7\u00b2 ranking", KU_RED),
    ]
    for i, (title, desc, clr) in enumerate(pipeline):
        x = Inches(0.4 + i * 3.15)
        card(s, x, Inches(1.6), Inches(2.8), Inches(2.0), clr)
        tb(s, x + Inches(0.1), Inches(1.7), Inches(2.6), Inches(0.5),
           title, sz=20, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        tb(s, x + Inches(0.1), Inches(2.2), Inches(2.6), Inches(1.2),
           desc, sz=14, color=WHITE, align=PP_ALIGN.CENTER)
        # Arrow between cards
        if i < 3:
            tb(s, Inches(3.0 + i * 3.15), Inches(2.2), Inches(0.5), Inches(0.5),
               "\u25B6", sz=20, bold=True, color=KU_GRAY, align=PP_ALIGN.CENTER)

    bullets(s, Inches(0.4), Inches(4.0), Inches(12.2), Inches(2.5), [
        "IMU Axes: X = Medial-Lateral, Y = Anterior-Posterior, Z = Superior-Inferior (Vertical)",
        "Window configs: Full Gait 6s (50% overlap) + U-Turn 3s (0% overlap)",
        "207 / 216 features significant (p < 0.05, Bonferroni corrected)",
        "Top features: HE_FreeAcc_X_dom_freq (\u03b7\u00b2 = 0.303), HE_FreeAcc_Y_dom_freq (\u03b7\u00b2 = 0.246)",
    ], sz=14)

    presenter_tag(s, 1, "Feature Extraction")
    print("  [3] Feature Extraction")

    # ──── SLIDE 4: Descriptive Statistics (P1-S3) ────
    s = prs.slides.add_slide(prs.slide_layouts[LY_CONTENT])
    title_ph(s, "Feature Correlation & Descriptive Statistics")

    img(s, FIGS["heatmap"], Inches(0.3), Inches(1.5), Inches(5.2), Inches(4.6))
    caption(s, Inches(0.3), Inches(6.2), Inches(5.2),
            "Pearson correlation matrix (top 50 features)")

    bullets(s, Inches(5.9), Inches(1.6), Inches(6.5), Inches(5.0), [
        "Key Observations:",
        "",
        "Strong correlation clusters within same-sensor features (r > 0.8)",
        "",
        "Foot sensor FreeAcc features are highly redundant",
        "",
        "Head sensor (HE) features are more independent \u2192 most informative",
        "",
        "Gyroscope dominant frequency features are most discriminative",
        "",
        "Top 3 by \u03b7\u00b2 effect size:",
        "  1. HE_FreeAcc_X_dom_freq (\u03b7\u00b2 = 0.303)",
        "  2. HE_FreeAcc_Y_dom_freq (\u03b7\u00b2 = 0.246)",
        "  3. LB_Gyr_Z_spec_centroid (\u03b7\u00b2 = 0.221)",
    ], sz=13)

    presenter_tag(s, 1, "Descriptive Statistics")
    print("  [4] Descriptive Statistics")

    # ──── SLIDE 5: SECTION - Methods ────
    s = prs.slides.add_slide(prs.slide_layouts[LY_SECTION])
    for ph in s.placeholders:
        idx = ph.placeholder_format.idx
        if idx == 0:
            ph.text = "Machine Learning Methods"
        elif idx == 1:
            ph.text = "Presenter 2  (Slides 4-6)"
        elif idx == 11:
            ph.text = "BMED 712 Rehabilitation Engineering"
    print("  [5] Section: Methods")

    # ──── SLIDE 6: ML Pipeline (P2-S1) ────
    s = prs.slides.add_slide(prs.slide_layouts[LY_CONTENT])
    title_ph(s, "Classification Pipeline & Cross-Validation")

    classifiers = [
        ("SVM (RBF)",      "BAcc: 81.0%\nF1: 0.796", KU_BLUE),
        ("Random Forest",  "BAcc: 80.1%\nF1: 0.788", KU_GREEN),
        ("XGBoost",        "BAcc: 79.8%\nF1: 0.781", KU_RED),
    ]
    for i, (name, perf, clr) in enumerate(classifiers):
        x = Inches(0.4 + i * 4.2)
        card(s, x, Inches(1.6), Inches(3.8), Inches(2.2), clr)
        tb(s, x + Inches(0.1), Inches(1.75), Inches(3.6), Inches(0.5),
           name, sz=22, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        tb(s, x + Inches(0.1), Inches(2.4), Inches(3.6), Inches(1.2),
           perf, sz=18, color=WHITE, align=PP_ALIGN.CENTER)

    # Best badge on SVM
    tb(s, Inches(0.4), Inches(1.35), Inches(1.2), Inches(0.3),
       "\u2B50 BEST", sz=11, bold=True, color=KU_BLUE, align=PP_ALIGN.CENTER)

    bullets(s, Inches(0.4), Inches(4.2), Inches(12.2), Inches(2.5), [
        "CV Strategy: 10-fold StratifiedGroupKFold grouped by subject_id (no data leakage)",
        "Nested CV: Outer 10-fold, Inner 5-fold \u2192 BAcc = 78.3% (2.7% optimism gap vs standard)",
        "Demographics: Age, gender (one-hot), laterality (one-hot) from _meta.json",
        "Feature sets compared: All 216 / Top 30 / Top 20 / Significant only (p < 0.05)",
    ], sz=14)

    presenter_tag(s, 2, "ML Pipeline")
    print("  [6] ML Pipeline")

    # ──── SLIDE 7: Feature Importance (P2-S2) ────
    s = prs.slides.add_slide(prs.slide_layouts[LY_CONTENT])
    title_ph(s, "Feature Importance & Selection")

    img(s, FIGS["importance"], Inches(0.3), Inches(1.4), Inches(8.5), Inches(4.2))
    caption(s, Inches(0.3), Inches(5.7), Inches(8.5),
            "Permutation importance - top 20 features (3-class SVM)")

    bullets(s, Inches(9.2), Inches(1.5), Inches(3.6), Inches(4.5), [
        "Top 20 features by KW \u03b7\u00b2:",
        "",
        "Head sensor dominates top ranks",
        "",
        "Gyroscope + FreeAcc signals most informative",
        "",
        "Top 20 achieves ~95% of full-feature performance",
        "",
        "Feature sets tested:",
        "  - All 216",
        "  - Top 30 (\u03b7\u00b2)",
        "  - Top 20 (\u03b7\u00b2)",
        "  - Significant only",
    ], sz=12)

    presenter_tag(s, 2, "Feature Selection")
    print("  [7] Feature Importance")

    # ──── SLIDE 8: Sensor Ablation (P2-S3) ────
    s = prs.slides.add_slide(prs.slide_layouts[LY_CONTENT])
    title_ph(s, "Sensor Ablation & Optimal Configuration")

    img(s, FIGS["frontier"], Inches(0.3), Inches(1.4), Inches(6.0), Inches(3.4))
    caption(s, Inches(0.3), Inches(4.9), Inches(6.0),
            "Sensor availability Pareto frontier (LR vs RF)")

    # Results cards on right
    ablation = [
        ("1 Sensor (HE)",       "77.5%", KU_BLUE),
        ("2 Sensors (HE+LB)",   "80.1%", KU_GREEN),
        ("4 Sensors (All)",      "81.0%", KU_RED),
    ]
    for i, (cfg, bacc, clr) in enumerate(ablation):
        y = Inches(1.5 + i * 1.3)
        card(s, Inches(6.8), y, Inches(5.5), Inches(1.1), clr)
        tb(s, Inches(6.9), y + Inches(0.15), Inches(3.2), Inches(0.7),
           cfg, sz=16, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
        tb(s, Inches(10.2), y + Inches(0.15), Inches(1.8), Inches(0.7),
           bacc, sz=28, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    bullets(s, Inches(6.8), Inches(5.2), Inches(5.5), Inches(1.5), [
        "Clinical takeaway: HE + LB = 98.9% of 4-sensor performance",
        "Practical: 2 IMUs sufficient for clinical deployment",
    ], sz=13)

    presenter_tag(s, 2, "Sensor Ablation")
    print("  [8] Sensor Ablation")

    # ──── SLIDE 9: SECTION - Results ────
    s = prs.slides.add_slide(prs.slide_layouts[LY_SECTION])
    for ph in s.placeholders:
        idx = ph.placeholder_format.idx
        if idx == 0:
            ph.text = "Classification Results"
        elif idx == 1:
            ph.text = "Presenter 3  (Slides 7-9)"
        elif idx == 11:
            ph.text = "BMED 712 Rehabilitation Engineering"
    print("  [9] Section: Results")

    # ──── SLIDE 10: 3-Class Classification (P3-S1) ────
    s = prs.slides.add_slide(prs.slide_layouts[LY_CONTENT])
    title_ph(s, "3-Class Classification Results")

    img(s, FIGS["confusion"], Inches(0.2), Inches(1.4), Inches(12.8), Inches(3.2))
    caption(s, Inches(0.2), Inches(4.7), Inches(12.8),
            "Confusion matrices: SVM, XGBoost, Random Forest (HC / Mild / Severe)")

    bullets(s, Inches(0.4), Inches(5.2), Inches(12.2), Inches(1.5), [
        "SVM: BAcc = 81.0%, F1 = 0.796  |  XGBoost: 79.8%, 0.781  |  RF: 80.1%, 0.788",
        "HC vs Pathological: >90% recall  |  Main challenge: Mild vs Severe boundary",
        "Largest error: Severe \u2192 Mild (12.3%) \u2014 late-recovery patients with near-normal gait",
    ], sz=14)

    presenter_tag(s, 3, "3-Class Results")
    print("  [10] 3-Class Classification")

    # ──── SLIDE 11: Phase Comparison (P3-S2) ────
    s = prs.slides.add_slide(prs.slide_layouts[LY_CONTENT])
    title_ph(s, "Gait Phase & Configuration Comparison")

    img(s, FIGS["phase"], Inches(0.3), Inches(1.4), Inches(7.2), Inches(4.0))
    caption(s, Inches(0.3), Inches(5.5), Inches(7.2),
            "Full Gait (6s) vs U-Turn (3s) vs Combined")

    # Comparison results on right
    configs = [
        ("Full Gait 6s\n(50% overlap)", "81.0%", KU_BLUE, "\u2714 Best"),
        ("U-Turn 3s\n(0% overlap)",     "74.2%", KU_TEAL, ""),
        ("Combined\n(Full + U-Turn)",   "79.5%", KU_GREEN, ""),
    ]
    for i, (cfg, bacc, clr, badge) in enumerate(configs):
        y = Inches(1.5 + i * 1.5)
        card(s, Inches(7.9), y, Inches(4.8), Inches(1.2), clr)
        tb(s, Inches(8.0), y + Inches(0.1), Inches(2.5), Inches(0.9),
           cfg, sz=14, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
        tb(s, Inches(10.5), y + Inches(0.1), Inches(1.8), Inches(0.9),
           bacc, sz=28, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        if badge:
            tb(s, Inches(12.0), y + Inches(0.3), Inches(0.7), Inches(0.4),
               badge, sz=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    bullets(s, Inches(7.9), Inches(5.6), Inches(4.8), Inches(1.0), [
        "Steady-state gait provides most discriminative features",
    ], sz=13)

    presenter_tag(s, 3, "Phase Comparison")
    print("  [11] Phase Comparison")

    # ──── SLIDE 12: Error Analysis (P3-S3) ────
    s = prs.slides.add_slide(prs.slide_layouts[LY_CONTENT])
    title_ph(s, "Error Modes & Clinical Correlation")

    img(s, FIGS["vga"], Inches(0.2), Inches(1.4), Inches(8.0), Inches(2.9))
    caption(s, Inches(0.2), Inches(4.4), Inches(8.0),
            "VGA score correlations with stride parameters and asymmetry indices")

    # Error modes on right
    errors = [
        ("Severe \u2192 Mild",  "12.3%", "Late-recovery patients"),
        ("Mild \u2192 HC",      "8.1%",  "Near-normal gait"),
        ("Mild \u2192 Severe",  "5.7%",  "Compensatory patterns"),
    ]
    for i, (mode, rate, reason) in enumerate(errors):
        y = Inches(1.5 + i * 1.0)
        card(s, Inches(8.6), y, Inches(4.2), Inches(0.85), KU_RED if i == 0 else KU_TEAL)
        tb(s, Inches(8.7), y + Inches(0.05), Inches(2.2), Inches(0.4),
           mode, sz=14, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
        tb(s, Inches(10.9), y + Inches(0.05), Inches(1.5), Inches(0.4),
           rate, sz=20, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        tb(s, Inches(8.7), y + Inches(0.45), Inches(3.8), Inches(0.35),
           reason, sz=11, color=WHITE, align=PP_ALIGN.LEFT)

    bullets(s, Inches(0.3), Inches(4.8), Inches(12.2), Inches(1.8), [
        "Cohen's d: Asymmetry indices show large effect sizes between Mild and Severe cohorts",
        "VGA scores correlate with IMU-derived stride variability (r = 0.68, p < 0.001)",
        "Clinical implication: ML errors cluster where clinicians also disagree",
    ], sz=13)

    presenter_tag(s, 3, "Error Analysis")
    print("  [12] Error Modes")

    # ──── SLIDE 13: SECTION - Robustness & Conclusion ────
    s = prs.slides.add_slide(prs.slide_layouts[LY_SECTION])
    for ph in s.placeholders:
        idx = ph.placeholder_format.idx
        if idx == 0:
            ph.text = "Robustness & Conclusion"
        elif idx == 1:
            ph.text = "Presenter 4  (Slides 10-12)"
        elif idx == 11:
            ph.text = "BMED 712 Rehabilitation Engineering"
    print("  [13] Section: Robustness")

    # ──── SLIDE 14: LOCO & Nested CV (P4-S1) ────
    s = prs.slides.add_slide(prs.slide_layouts[LY_CONTENT])
    title_ph(s, "Leave-One-Cohort-Out & Nested CV")

    loco = [
        ("Calcaneus Fx",    "76.8%", KU_BLUE),
        ("Ankle Fx",        "79.2%", KU_GREEN),
        ("Tibial Plat. Fx", "73.5%", KU_TEAL),
        ("Femoral Fx",      "71.9%", KU_RED),
        ("Knee Replace.",   "78.4%", KU_BLUE),
        ("Hip Replace.",    "80.1%", KU_GREEN),
        ("Hip Fx",          "74.6%", KU_TEAL),
    ]
    for i, (cohort, bacc, clr) in enumerate(loco):
        col = i % 4
        row = i // 4
        x = Inches(0.3 + col * 3.15)
        y = Inches(1.5 + row * 2.2)
        card(s, x, y, Inches(2.85), Inches(1.8), clr)
        tb(s, x + Inches(0.1), y + Inches(0.15), Inches(2.65), Inches(0.5),
           cohort, sz=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        tb(s, x + Inches(0.1), y + Inches(0.7), Inches(2.65), Inches(0.8),
           bacc, sz=36, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    # Nested CV comparison at bottom
    card(s, Inches(0.3), Inches(5.5), Inches(12.2), Inches(1.0), RGBColor(0xF5, 0xF5, 0xF5))
    tb(s, Inches(0.5), Inches(5.6), Inches(5.5), Inches(0.4),
       "Standard 10-Fold CV:  BAcc = 81.0%", sz=16, bold=True, color=KU_BLUE)
    tb(s, Inches(6.3), Inches(5.6), Inches(3.0), Inches(0.4),
       "Nested CV:  BAcc = 78.3%", sz=16, bold=True, color=KU_RED)
    tb(s, Inches(9.5), Inches(5.6), Inches(3.0), Inches(0.4),
       "\u2192  2.7% optimism gap", sz=16, bold=True, color=BLACK)
    tb(s, Inches(0.5), Inches(6.1), Inches(12.0), Inches(0.4),
       "Model generalizes well - not overfit. LOCO confirms robustness across pathology types.",
       sz=13, color=BLACK)

    presenter_tag(s, 4, "LOCO & Nested CV")
    print("  [14] LOCO & Nested CV")

    # ──── SLIDE 15: Clinical Implications (P4-S2) ────
    s = prs.slides.add_slide(prs.slide_layouts[LY_CONTENT])
    title_ph(s, "Clinical Implications & Limitations")

    # Left: Implications (green card)
    card(s, Inches(0.3), Inches(1.5), Inches(6.0), Inches(5.0), LIGHT_BG)
    tb(s, Inches(0.5), Inches(1.6), Inches(5.5), Inches(0.5),
       "Clinical Implications", sz=20, bold=True, color=KU_GREEN)
    bullets(s, Inches(0.5), Inches(2.2), Inches(5.5), Inches(4.0), [
        "81% accuracy with wearable IMUs",
        "2 sensors (Head + Lower Back) sufficient",
        "Real-time gait screening feasible",
        "Objective complement to VGA scoring",
        "Scalable to home/remote monitoring",
        "Automated severity triage in clinics",
    ], sz=14)

    # Right: Limitations (red card)
    card(s, Inches(6.7), Inches(1.5), Inches(6.0), Inches(5.0), LIGHT_RED_BG)
    tb(s, Inches(6.9), Inches(1.6), Inches(5.5), Inches(0.5),
       "Limitations", sz=20, bold=True, color=KU_RED)
    bullets(s, Inches(6.9), Inches(2.2), Inches(5.5), Inches(4.0), [
        "Single-site dataset (Innsbruck)",
        "Lab-controlled environment only",
        "Mild/Severe boundary is subjective",
        "No longitudinal follow-up data",
        "Missing blood pressure / heart rate",
        "216 hand-crafted features (no DL)",
    ], sz=14)

    presenter_tag(s, 4, "Clinical Implications")
    print("  [15] Clinical Implications")

    # ──── SLIDE 16: Conclusions (P4-S3) ────
    s = prs.slides.add_slide(prs.slide_layouts[LY_CONTENT])
    title_ph(s, "Conclusions & Future Work")

    conclusions = [
        ("01", "81.0% Balanced Accuracy",
         "SVM with 216 IMU features\nachieves robust 3-class classification", KU_BLUE),
        ("02", "2 Sensors Sufficient",
         "Head + Lower Back IMUs capture\n98.9% of 4-sensor performance", KU_GREEN),
        ("03", "Clinically Validated",
         "Feature rankings align with\nestablished VGA scoring criteria", KU_RED),
    ]
    for i, (num, title, desc, clr) in enumerate(conclusions):
        x = Inches(0.3 + i * 4.2)
        # Number circle
        oval = s.shapes.add_shape(MSO_SHAPE.OVAL, x, Inches(1.6), Inches(0.8), Inches(0.8))
        oval.fill.solid()
        oval.fill.fore_color.rgb = clr
        oval.line.fill.background()
        tb(s, x + Inches(0.05), Inches(1.65), Inches(0.7), Inches(0.7),
           num, sz=24, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

        tb(s, x, Inches(2.6), Inches(3.8), Inches(0.5),
           title, sz=18, bold=True, color=clr)
        tb(s, x, Inches(3.2), Inches(3.8), Inches(1.2),
           desc, sz=14, color=BLACK)

    # Future work
    card(s, Inches(0.3), Inches(4.8), Inches(12.2), Inches(1.8), RGBColor(0xF5, 0xF5, 0xF5))
    tb(s, Inches(0.5), Inches(4.9), Inches(5), Inches(0.4),
       "Future Directions", sz=18, bold=True, color=KU_BLUE)

    future = [
        ("Deep Learning:", "CNN/LSTM on raw IMU signals"),
        ("Multi-Site:", "External validation on other gait datasets"),
        ("Edge Deployment:", "Real-time classification on wearable devices"),
        ("Longitudinal:", "Track recovery progression over time"),
    ]
    for i, (label, desc) in enumerate(future):
        col = i % 2
        row = i // 2
        x = Inches(0.5 + col * 6.2)
        y = Inches(5.4 + row * 0.5)
        tf = tb(s, x, y, Inches(5.8), Inches(0.4), "", sz=13)
        p = tf.paragraphs[0]
        r1 = p.add_run()
        r1.text = label + " "
        r1.font.size = Pt(13)
        r1.font.bold = True
        r1.font.color.rgb = KU_BLUE
        r2 = p.add_run()
        r2.text = desc
        r2.font.size = Pt(13)
        r2.font.color.rgb = BLACK

    presenter_tag(s, 4, "Conclusions")
    print("  [16] Conclusions")

    # ──── SLIDE 17: THANK YOU (end) ────
    s = prs.slides.add_slide(prs.slide_layouts[LY_END])
    # The End layout may not have placeholders, add text manually
    tb(s, Inches(1.0), Inches(2.5), Inches(11.0), Inches(1.5),
       "Thank You", sz=48, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    tb(s, Inches(1.0), Inches(4.0), Inches(11.0), Inches(1.0),
       "BMED 712 \u2013 Rehabilitation Engineering  |  Khalifa University  |  April 2026",
       sz=18, color=WHITE, align=PP_ALIGN.CENTER)
    tb(s, Inches(1.0), Inches(5.0), Inches(11.0), Inches(0.5),
       "Questions?", sz=24, bold=True, color=RGBColor(0x00, 0xCE, 0x7C),
       align=PP_ALIGN.CENTER)
    print("  [17] Thank You")

    # ──── SAVE ────
    prs.save(str(OUTPUT))
    import os
    size_kb = os.path.getsize(OUTPUT) / 1024
    print(f"\nSaved: {OUTPUT} ({size_kb:.0f} KB)")
    print("Total: 18 slides = 1 title + 4 section + 12 content + 1 end")
    print("Content: 12 slides, 3 per presenter (60-90s each)")


if __name__ == "__main__":
    build()
