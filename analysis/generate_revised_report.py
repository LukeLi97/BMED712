"""Generate revised Progress Report PDF incorporating all professor feedback.

Changes from original:
  1. Softer language (robust biomarkers → potential indicators, etc.)
  2. Abstract leads with two explicit contributions
  3. ML framing: emphasize clinical insight over model improvement
  4. Sensor ablation key-takeaway added
  5. Fig 7: use fixed version (no regression line, ordinal boxplots)
  6. Table II: signed Cohen's d (all pathological groups negative)
  7. Discussion streamlined
"""

from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIG_DIR = REPO / "results" / "figures"
ART_DIR = REPO / "results" / "artifacts"
OUT_PDF = REPO / "results" / "Progress_Report_Revised.pdf"

# ── reportlab imports ────────────────────────────────────────────────────
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate,
    Paragraph, Spacer, Table, TableStyle, Image,
    KeepTogether, HRFlowable, PageBreak,
)
from reportlab.platypus.flowables import Flowable


# ── Two-column page layout ────────────────────────────────────────────────
PAGE_W, PAGE_H = letter
MARGIN = 0.75 * inch
COL_GAP = 0.25 * inch
COL_W = (PAGE_W - 2 * MARGIN - COL_GAP) / 2


def make_doc():
    """Create BaseDocTemplate with two-column layout."""
    doc = BaseDocTemplate(
        str(OUT_PDF),
        pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=0.85 * inch, bottomMargin=0.75 * inch,
    )

    # Title area (full-width, first page only)
    title_frame = Frame(MARGIN, PAGE_H - 1.8 * inch, PAGE_W - 2 * MARGIN,
                        1.0 * inch, id="title", showBoundary=0)
    # Two columns (rest of page)
    col_y = 0.75 * inch
    col_h = PAGE_H - 2.65 * inch
    left_frame = Frame(MARGIN, col_y, COL_W, col_h, id="left", showBoundary=0)
    right_frame = Frame(MARGIN + COL_W + COL_GAP, col_y, COL_W, col_h,
                        id="right", showBoundary=0)

    # Full-width column (for wide tables/figures)
    full_frame = Frame(MARGIN, col_y, PAGE_W - 2 * MARGIN, col_h,
                       id="full", showBoundary=0)

    # Subsequent pages: two columns
    two_col_left = Frame(MARGIN, 0.75 * inch, COL_W,
                         PAGE_H - 1.6 * inch, id="left2", showBoundary=0)
    two_col_right = Frame(MARGIN + COL_W + COL_GAP, 0.75 * inch, COL_W,
                          PAGE_H - 1.6 * inch, id="right2", showBoundary=0)

    first_page = PageTemplate(
        id="FirstPage",
        frames=[title_frame, left_frame, right_frame],
        onPage=page_footer,
    )
    normal_page = PageTemplate(
        id="NormalPage",
        frames=[two_col_left, two_col_right],
        onPage=page_footer,
    )
    doc.addPageTemplates([first_page, normal_page])
    return doc


def page_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.grey)
    footer = ("BMED 712 — Track A Progress Report (Revised)  |  "
              "Habib Farweh, Li, Khattab, Ali  |  Spring 2026")
    canvas.drawCentredString(PAGE_W / 2, 0.45 * inch, footer)
    canvas.drawRightString(PAGE_W - MARGIN, 0.45 * inch, f"Page {doc.page}")
    canvas.restoreState()


# ── Styles ────────────────────────────────────────────────────────────────
def make_styles():
    base = getSampleStyleSheet()

    S = {}

    S["title"] = ParagraphStyle(
        "title", fontSize=14, fontName="Helvetica-Bold",
        alignment=TA_CENTER, spaceAfter=4,
    )
    S["authors"] = ParagraphStyle(
        "authors", fontSize=9, fontName="Helvetica",
        alignment=TA_CENTER, spaceAfter=2,
    )
    S["affil"] = ParagraphStyle(
        "affil", fontSize=8, fontName="Helvetica-Oblique",
        alignment=TA_CENTER, spaceAfter=6,
    )
    S["abstract_head"] = ParagraphStyle(
        "abstract_head", fontSize=9, fontName="Helvetica-Bold",
        spaceAfter=2,
    )
    S["abstract"] = ParagraphStyle(
        "abstract", fontSize=8, fontName="Helvetica",
        alignment=TA_JUSTIFY, leading=11, spaceAfter=8,
    )
    S["section"] = ParagraphStyle(
        "section", fontSize=9, fontName="Helvetica-Bold",
        spaceBefore=8, spaceAfter=3, textColor=colors.HexColor("#1a1a1a"),
    )
    S["subsection"] = ParagraphStyle(
        "subsection", fontSize=8.5, fontName="Helvetica-Bold",
        spaceBefore=5, spaceAfter=2,
    )
    S["body"] = ParagraphStyle(
        "body", fontSize=8, fontName="Helvetica",
        alignment=TA_JUSTIFY, leading=11, spaceAfter=4,
    )
    S["body_indent"] = ParagraphStyle(
        "body_indent", parent=S["body"], leftIndent=10,
    )
    S["caption"] = ParagraphStyle(
        "caption", fontSize=7.5, fontName="Helvetica-Oblique",
        alignment=TA_CENTER, spaceAfter=4, spaceBefore=2,
    )
    S["table_header"] = ParagraphStyle(
        "table_header", fontSize=7, fontName="Helvetica-Bold",
        alignment=TA_CENTER,
    )
    S["table_cell"] = ParagraphStyle(
        "table_cell", fontSize=7, fontName="Helvetica",
        alignment=TA_CENTER,
    )
    S["key_finding"] = ParagraphStyle(
        "key_finding", fontSize=8, fontName="Helvetica-Oblique",
        alignment=TA_LEFT, leading=11,
        leftIndent=8, rightIndent=8,
        borderPad=4, backColor=colors.HexColor("#f0f4f8"),
        spaceAfter=4, spaceBefore=4,
    )
    return S


# ── Table helpers ─────────────────────────────────────────────────────────
HEADER_COLOR = colors.HexColor("#2c3e50")
ROW_COLOR = colors.HexColor("#f7f9fb")
GRID_COLOR = colors.HexColor("#cccccc")


def styled_table(data, col_widths, header_rows=1):
    t = Table(data, colWidths=col_widths, repeatRows=header_rows)
    n = len(data)
    style = [
        ("BACKGROUND", (0, 0), (-1, header_rows - 1), HEADER_COLOR),
        ("TEXTCOLOR", (0, 0), (-1, header_rows - 1), colors.white),
        ("FONTNAME", (0, 0), (-1, header_rows - 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.3, GRID_COLOR),
        ("ROWBACKGROUNDS", (0, header_rows), (-1, -1),
         [colors.white, ROW_COLOR]),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]
    t.setStyle(TableStyle(style))
    return t


# ── Image helper ──────────────────────────────────────────────────────────
def fig(path, width=None, caption=""):
    if width is None:
        width = COL_W - 0.1 * inch
    try:
        img = Image(str(path), width=width,
                    height=width * 0.55, kind="proportional")
    except Exception:
        img = Paragraph(f"[Figure: {Path(path).name}]",
                        make_styles()["caption"])
    if caption:
        return [img, Paragraph(caption, make_styles()["caption"])]
    return [img]


# ── Content builders ──────────────────────────────────────────────────────
def build_story(S):
    story = []

    # ── Title block (full-width) ─────────────────────────────────────────
    story.append(Paragraph(
        "Gait Phenotyping Across Pathologies Using Wearable IMUs:"
        " Temporal Asymmetry, Sensor Optimization, and Clinical Screening",
        S["title"]))
    story.append(Paragraph(
        "Fatima Habib Farweh (100066893), Liang Li (100065824), "
        "Yasmine Khattab (100067769), Zehara Ali (100058079)",
        S["authors"]))
    story.append(Paragraph(
        "BMED712 — Rehabilitation and Augmentation of Human Performance | "
        "Spring 2026 | Instructors: Dr. Kinda Khalaf and Dr. Mohamed Elgendi",
        S["affil"]))

    # ── Abstract ─────────────────────────────────────────────────────────
    story.append(Paragraph("Abstract", S["abstract_head"]))
    story.append(Paragraph(
        "<i>Wearable inertial measurement units (IMUs) enable scalable, "
        "real-world gait analysis for diverse clinical populations. This study "
        "makes two primary contributions: (1) characterizing stride and step "
        "temporal asymmetry across neurological and orthopaedic subtypes "
        "using the multi-pathology dataset of Voisard et al. (2025), and "
        "(2) evaluating IMU sensor placement trade-offs for gait-based "
        "pathology classification. A parameter extraction pipeline computing "
        "seven clinically validated temporal metrics was developed, and a "
        "sensor ablation study was conducted using SVM, XGBoost, and Random "
        "Forest models. Results suggest that healthy subjects exhibit greater "
        "temporal asymmetry than pathological groups (Cohen's d = 0.77, "
        "95% CI: 0.50–1.07, p &lt; 0.001), consistent with a motor "
        "lateralization hypothesis. Across all models, the right foot (RF) "
        "sensor consistently provided the highest single-sensor performance "
        "(up to 88.9% balanced accuracy), and foot-only configurations "
        "retained approximately 93% of full-sensor accuracy—suggesting "
        "potential for a practical two-sensor clinical wearable. These "
        "findings indicate a possible role for IMU-derived temporal asymmetry "
        "as a clinical indicator, with further validation needed before "
        "clinical translation.</i>",
        S["abstract"]))

    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#cccccc"), spaceAfter=4))

    # ── I. Introduction ──────────────────────────────────────────────────
    story.append(Paragraph("I. INTRODUCTION", S["section"]))
    story.append(Paragraph(
        "Human walking is among the most precisely regulated motor behaviors "
        "executed by the central nervous system. Its bilateral coordination "
        "reflects the integrity of a multilevel neuromotor architecture, and "
        "its degradation is a sensitive indicator of diverse neurological and "
        "orthopaedic conditions [1, 2].",
        S["body"]))
    story.append(Paragraph(
        "Quantitative gait analysis has long been a clinical tool, but its "
        "reach has historically been limited by the cost and spatial "
        "constraints of laboratory-based force-platform and motion-capture "
        "systems. Inertial measurement units (IMUs) have disrupted this "
        "paradigm: lightweight, wireless, and low-cost, they enable gait "
        "measurement in clinics, rehabilitation wards, and home environments. "
        "The dataset published by Voisard et al. (2025) in Scientific Data "
        "provides 1,356 trials from 260 participants across three clinical "
        "categories and seven diagnostic subtypes, recorded with four "
        "synchronised IMUs [3].",
        S["body"]))
    story.append(Paragraph(
        "This paper reports two contributions: (1) statistical characterization "
        "of temporal asymmetry across all seven diagnostic subtypes, and "
        "(2) sensor ablation analysis identifying cost-effective IMU "
        "configurations. We frame our findings as preliminary evidence "
        "warranting further validation rather than established clinical "
        "guidelines.",
        S["body"]))

    # ── II. Research Questions ───────────────────────────────────────────
    story.append(Paragraph("II. RESEARCH QUESTIONS", S["section"]))
    story.append(Paragraph(
        "<b>RQ1 (Sensor Optimization):</b> Which sensor configurations "
        "provide optimal cost-performance trade-offs for pathology "
        "classification across gait phases?",
        S["body_indent"]))
    story.append(Paragraph(
        "<b>RQ2 (Temporal Asymmetry):</b> Does temporal gait asymmetry "
        "differ between healthy and pathological populations, and if so, "
        "in which direction?",
        S["body_indent"]))

    # ── III. Magnitude of the Problem ────────────────────────────────────
    story.append(Paragraph("III. MAGNITUDE OF THE PROBLEM", S["section"]))
    story.append(Paragraph(
        "Gait disorders affect hundreds of millions of people worldwide. "
        "Neurological conditions alone affect more than one billion people "
        "globally, with motor and coordination deficits that directly impair "
        "walking function [4]. Parkinson's disease affects approximately "
        "8.5 million people worldwide [5]. Stroke affects ~13 million people "
        "annually, with 30–40% of survivors exhibiting persistent gait "
        "abnormalities [6]. On the orthopaedic side, knee osteoarthritis "
        "affects ~250 million people globally and is the leading cause of "
        "pain-related mobility impairment in adults over 50 [8].",
        S["body"]))
    story.append(Paragraph(
        "Falls are the leading cause of injury-related mortality among "
        "adults aged 65 and older. The CDC estimates ~36 million falls "
        "annually among older adults in the U.S., resulting in over 32,000 "
        "deaths and costs exceeding $50 billion per year [10]. Objective "
        "gait monitoring may support earlier detection and intervention.",
        S["body"]))

    # ── IV. Relevance ────────────────────────────────────────────────────
    story.append(Paragraph("IV. RELEVANCE", S["section"]))
    story.append(Paragraph(
        "Wearable IMU-based gait analysis potentially addresses three "
        "clinical needs: screening, longitudinal monitoring, and "
        "rehabilitation assessment. The Voisard et al. (2025) dataset "
        "enables direct validation of computational gait features against "
        "established clinical measures including UPDRS-III, mRS, FAC, TNS, "
        "Kellgren–Lawrence grading, and Visual Gait Assessment (VGA).",
        S["body"]))

    # ── V. What Has Been Done ────────────────────────────────────────────
    story.append(Paragraph("V. WHAT HAS BEEN DONE WITH THE DATASET",
                            S["section"]))
    story.append(Paragraph(
        "The dataset creators validated automatic gait event detection "
        "from healthy to severely impaired patients [12]. Al-Harthi et al. "
        "(2026) applied G-MASA-TCN deep learning to all 1,356 trials, with "
        "Integrated Gradients explainability identifying foot-mounted IMU "
        "channels as the most attribution-weighted inputs—converging with our "
        "sensor ablation finding that RF is the single most informative "
        "placement [3].",
        S["body"]))
    story.append(Paragraph(
        "Sadeghsalehi (2025) applied a multi-stream attention model to "
        "binary classification tasks and identified a critical laterality "
        "confound: all 15 HOA patients had right-sided pathology, and 47 of "
        "49 CVA patients had right-dominant motor deficits. This means "
        "right-foot sensor dominance in those tasks reflects dataset-specific "
        "laterality bias rather than a general superiority of RF sensing [11]. "
        "We flag this where relevant in our subtype results.",
        S["body"]))

    # ── VI. Methodology ──────────────────────────────────────────────────
    story.append(Paragraph("VI. METHODOLOGY", S["section"]))

    story.append(Paragraph("a. Data Preparation", S["subsection"]))
    story.append(Paragraph(
        "All 1,356 trials from 260 subjects were downloaded from Figshare "
        "(DOI: 10.6084/m9.figshare.28806086). U-turn segments were excluded "
        "from parameter computations as they represent biomechanically "
        "distinct, non-steady-state gait (heel-strikes outside [uturn_start, "
        "uturn_end] were retained). After exclusion, 974 valid straight-"
        "walking trials from 216 subjects were retained.",
        S["body"]))

    story.append(Paragraph("b. Parameter Extraction", S["subsection"]))
    story.append(Paragraph(
        "Seven clinically validated gait metrics were computed from "
        "pre-annotated bilateral heel-strike timestamps per trial: "
        "stride time, step time, cadence, stance phase, swing phase, "
        "stride CV, and symmetry index. All parameters were aggregated "
        "at the subject level (mean across retained trials).",
        S["body"]))

    story.append(Paragraph("c. Sensor Ablation", S["subsection"]))
    story.append(Paragraph(
        "A sensor ablation study was performed using XGBoost, SVM, and "
        "Random Forest models across gait phases (full gait, pre-uturn, "
        "post-uturn, u-turn), window sizes (1–6 s), and overlap "
        "percentages (0%, 25%, 50%). Performance was assessed using "
        "balanced accuracy and macro-F1 to account for class imbalance.",
        S["body"]))

    # ── VII. Preliminary Results ─────────────────────────────────────────
    story.append(Paragraph("VII. PRELIMINARY RESULTS", S["section"]))

    story.append(Paragraph("a. Gait Parameters Across Groups", S["subsection"]))
    story.append(Paragraph(
        "Neurological groups (PD, CVA, RIL) exhibit elevated stride times, "
        "decreased cadence, increased temporal variability (CV), and reduced "
        "symmetry index compared to healthy subjects. Orthopaedic groups "
        "generally show intermediate or near-healthy values, with ACL "
        "patients indistinguishable from healthy controls on most temporal "
        "parameters, consistent with an intact central motor program driving "
        "a mechanically altered but neurally intact gait.",
        S["body"]))

    # Table I
    story.append(Paragraph(
        "TABLE I. SUBJECT-LEVEL TEMPORAL ASYMMETRY METRICS. "
        "95% Bootstrap CI shown for primary statistic. "
        "***p<0.001, **p<0.01 (Welch's t-test, Healthy vs. Combined Pathological).",
        S["caption"]))
    t1_data = [
        ["Metric", "Healthy (n=70)", "Ortho (n=35)", "Neuro (n=111)"],
        ["Stride |AI|", "0.052 ± 0.036", "0.039 ± 0.027", "0.029 ± 0.018"],
        ["Stride |L-R| (s)", "0.069 ± 0.054", "0.054 ± 0.037", "0.038 ± 0.023"],
        ["Step |AI|", "0.148 ± 0.091", "0.088 ± 0.075", "0.118 ± 0.093"],
        ["Step CV (L)", "0.284 ± 0.236", "0.176 ± 0.153", "0.187 ± 0.152"],
        ["Mean step time (s)", "0.605 ± 0.035", "0.628 ± 0.053", "0.608 ± 0.057"],
    ]
    cw1 = [1.3 * inch, 0.85 * inch, 0.85 * inch, 0.85 * inch]
    story.append(styled_table(t1_data, cw1))
    story.append(Spacer(1, 4))

    story.append(Paragraph("b. Temporal Asymmetry", S["subsection"]))
    story.append(Paragraph(
        "<b>(1) Healthy vs. Pathological.</b> "
        "The central finding is that healthy subjects exhibit significantly "
        "larger temporal asymmetry than pathological groups (Table I). "
        "Stride |AI| yields Cohen's d = 0.77 (95% Bootstrap CI: 0.50–1.07) "
        "with p &lt; 0.001. AUC = 0.716 (95% CI: 0.635–0.792) at threshold "
        "0.049 delivers 83% specificity and 59% sensitivity. Mean step time "
        "does not differ between groups (p = 0.19), ruling out walking speed "
        "as a confound.",
        S["body"]))
    story.append(Paragraph(
        "In healthy individuals, the signed AI is consistently positive "
        "(left stride slightly longer, mean = +0.009), consistent with "
        "dominant-leg lateralization. In neurological patients it approaches "
        "zero and becomes directionally unpredictable. The orthopaedic group "
        "is not significantly different from healthy (LME p = 1.00), "
        "suggesting mechanical joint pathology does not disrupt central "
        "lateralization.",
        S["body"]))

    story.append(Paragraph(
        "<b>(2) Cross-Pathology Comparison.</b> "
        "Cohen's d (signed: pathological − healthy) indicates that "
        "neurological conditions with central motor involvement show the "
        "largest asymmetry reduction (Table II). RIL (d = −0.87) and PD "
        "(d = −0.77) show large effects; CVA (d = −0.73) shows a large "
        "effect, though interpretation must account for the right-dominant "
        "laterality confound [11]. CIPN shows a medium effect (d = −0.53). "
        "KOA reaches statistical significance (d = −0.45, p = 0.034), while "
        "HOA and ACL are not significantly different from healthy.",
        S["body"]))

    # Table II — CORRECTED (signed d)
    story.append(Paragraph(
        "TABLE II. STRIDE |AI| BY SUBTYPE (REVISED). "
        "Cohen's d = (pathological − healthy) / pooled SD; negative values "
        "indicate lower |AI| than healthy. "
        "†CVA and HOA subject to right-dominant laterality sampling confound; "
        "interpret with caution. ***p<0.001, **p<0.01, *p<0.05.",
        S["caption"]))
    t2_data = [
        ["Subtype", "Category", "Stride |AI|", "d vs Healthy", "p-value"],
        ["RIL (n=14)", "Neurological", "0.024", "−0.87", "<0.001***"],
        ["PD (n=17)", "Neurological", "0.025", "−0.77", "<0.001***"],
        ["CVA (n=44)†", "Neurological", "0.027", "−0.73", "<0.001***"],
        ["CIPN (n=36)", "Neurological", "0.033", "−0.53", "0.003**"],
        ["KOA (n=14)", "Orthopaedic", "0.036", "−0.45", "0.034*"],
        ["HOA (n=12)†", "Orthopaedic", "0.042", "−0.27", "0.226 ns"],
        ["ACL (n=9)", "Orthopaedic", "0.049", "−0.09", "0.779 ns"],
    ]
    cw2 = [0.9 * inch, 0.9 * inch, 0.75 * inch, 0.85 * inch, 0.75 * inch]
    story.append(styled_table(t2_data, cw2))
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "<b>(3) ROC Screening Performance and Clinical Validation.</b> "
        "Stride |AI| achieves AUC = 0.716 [0.635–0.792] at threshold 0.049, "
        "sensitivity 0.59, specificity 0.83. Spearman correlation between "
        "VGA severity and stride |AI| is ρ = −0.206 (p &lt; 0.001, n = 927), "
        "indicating convergent validity with established clinical assessment. "
        "This correlation is weak-to-moderate (r² ≈ 0.042), suggesting IMU "
        "captures information beyond coarse visual rating but that the "
        "relationship is not strong.",
        S["body"]))

    # Figure 7 — fixed version
    fig7_path = FIG_DIR / "step07_corr_vga_stride_absAI_fixed.png"
    story.append(Spacer(1, 4))
    story.extend(fig(
        fig7_path, width=COL_W - 0.05 * inch,
        caption=(
            "Fig. 7: VGA score vs stride |AI|. Panel A: scatter colored by "
            "group; linear regression not shown (VGA is an ordinal variable). "
            "Spearman ρ = −0.206, p < 0.001, n = 927. Panel B: stride |AI| "
            "distribution per VGA category (0 = normal, 4 = severe)."
        )))

    story.append(Paragraph("c. ML Feature Integration", S["subsection"]))
    story.append(Paragraph(
        "Adding 11 asymmetry features to the 217-feature sensor set produced "
        "a modest but consistent improvement (Table III). On the matched "
        "974-trial subset, macro-F1 improved +0.007 and BAcc +0.009. "
        "Asymmetry features alone achieve ~50% F1 (vs. 33% chance). "
        "No asymmetry feature appears in the RF top-25, confirming spectral "
        "features already implicitly encode temporal lateralization. "
        "The AUC of 0.716 represents a modest result; the primary "
        "contribution of this work is the <i>clinical characterization</i> "
        "of asymmetry patterns across subtypes, not incremental model "
        "performance improvement.",
        S["body"]))

    # Table III
    story.append(Paragraph(
        "TABLE III. 5-FOLD STRATIFIEDGROUPKFOLD CV (SUBJECT-GROUPED). "
        "*Values after slash = matched 974-trial subset.",
        S["caption"]))
    t3_data = [
        ["Configuration", "Features", "LR F1", "RF F1", "SVM F1"],
        ["Sensor only", "217", "0.748", "0.810", "0.816 / 0.806*"],
        ["Sensor + Asymmetry", "228", "0.741", "0.810", "0.822 / 0.812*"],
        ["Feet + Asymmetry", "120", "0.735", "0.770", "0.791"],
        ["Asymmetry only", "11", "0.447", "0.466", "0.491"],
    ]
    cw3 = [1.3 * inch, 0.65 * inch, 0.6 * inch, 0.6 * inch, 0.85 * inch]
    story.append(styled_table(t3_data, cw3))
    story.append(Spacer(1, 4))

    story.append(Paragraph("d. Feature Directionality", S["subsection"]))
    story.append(Paragraph(
        "Of 182 features significant at p &lt; 0.05 (Kruskal-Wallis), the "
        "dominant pattern is monotonic decrease H>O>N (67 features). Spectral "
        "centroid frequencies and signal dynamics decrease systematically from "
        "healthy to neurological, reflecting slower and less variable gait. "
        "All top-20 features show 93–100% cross-subject concordance, "
        "confirming effects are present in nearly every individual.",
        S["body"]))

    # ── Sensor Ablation Results ──────────────────────────────────────────
    story.append(Paragraph("e. Sensor Ablation Results", S["subsection"]))

    # Key finding box
    story.append(Paragraph(
        "<b>Key Finding:</b> Foot-only sensors (LF+RF) retain approximately "
        "93% of full-sensor balanced accuracy (68.2% vs 73.4%), suggesting "
        "a practical two-sensor wearable configuration for clinical use. "
        "The RF sensor is consistently the top single-sensor recommendation "
        "across all three models and gait phases.",
        S["key_finding"]))
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "<b>(1) SVM.</b> The RF sensor achieved the highest single-sensor "
        "SVM performance. Best results: full gait 85.97% BAcc / 85.4% F1 "
        "(5s, 50% overlap); pre-uturn 83.2% / 80.8% (4s, 25%); "
        "post-uturn 87.6% / 80.8% (6s, 25%). Best two-sensor configuration: "
        "LB+HE for post-uturn (89.8% / 84.4%, 6s, 50%).",
        S["body"]))

    # Table SVM (abbreviated)
    story.append(Paragraph(
        "TABLE IV. BEST SENSOR CONFIGURATIONS — SVM.",
        S["caption"]))
    t_svm = [
        ["Phase", "Best Single", "Win", "OL%", "BAcc", "F1"],
        ["Full gait", "RF", "5s", "50%", "85.97%", "85.4%"],
        ["Pre-uturn", "RF", "4s", "25%", "83.2%", "80.8%"],
        ["Post-uturn", "RF", "6s", "25%", "87.6%", "80.8%"],
        ["U-turn", "RF", "1.28s", "50%", "76.0%", "75.3%"],
    ]
    cw_svm = [0.7 * inch, 0.65 * inch, 0.5 * inch, 0.5 * inch,
              0.65 * inch, 0.65 * inch]
    story.append(styled_table(t_svm, cw_svm))
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "<b>(2) XGBoost.</b> Best single-sensor: full gait RF 88.7% "
        "(5s, 50%); post-uturn LB 90.7% (3s, 25%); pre-uturn HE 90.0% "
        "(6s, 50%); u-turn RF 78.2% (1s, 50%). Best two-sensor: "
        "post-uturn HE+LB 93.9% (6s, 25%); full gait LB+RF 91.8% (5s, 50%).",
        S["body"]))

    story.append(Paragraph(
        "<b>(3) Random Forest.</b> Best single-sensor: pre-uturn RF "
        "88.9% / 85.6% (3s, 25%); post-uturn RF 88.5% / 82.0% (6s, 50%); "
        "full gait RF 85.1% / 84.2% (5s, 50%). Best two-sensor: "
        "post-uturn LF+RF 89.1%, full gait LB+HE 87.0%.",
        S["body"]))

    story.append(Paragraph(
        "Window size and overlap significantly impact performance. Larger "
        "windows (5–6 s) are generally optimal for continuous gait phases "
        "across models, while the u-turn phase requires smaller windows "
        "(1–1.28 s) to capture its biomechanical signature.",
        S["body"]))

    # ── VIII. Discussion ─────────────────────────────────────────────────
    story.append(Paragraph("VIII. DISCUSSION", S["section"]))
    story.append(Paragraph(
        "This study makes two contributions. First, we characterize temporal "
        "asymmetry across seven gait pathology subtypes, finding that healthy "
        "subjects exhibit greater unsigned asymmetry than pathological groups "
        "(d = 0.77, p &lt; 0.001). This is consistent with Sadeghi et al.'s "
        "meta-analysis showing reliable left-right differences in able-bodied "
        "adults driven by motor lateralization [14]. Our data suggest that "
        "neurological conditions disrupt the lateralized motor program, "
        "causing convergence toward zero asymmetry. Notably, RIL and PD "
        "show the largest effects (d = −0.87 and −0.77), while orthopaedic "
        "subtypes show smaller or non-significant effects, consistent with "
        "preserved central motor control in mechanical joint conditions.",
        S["body"]))
    story.append(Paragraph(
        "Second, the sensor ablation study indicates that the right foot "
        "sensor is consistently the most informative single placement. "
        "Foot-only configurations retain approximately 93% of full-sensor "
        "accuracy, suggesting potential for a practical two-sensor wearable. "
        "However, the laterality confound in CVA and HOA (Sadeghsalehi et al.) "
        "means that RF dominance in those tasks may partly reflect dataset "
        "structure rather than biological generalizability [11].",
        S["body"]))
    story.append(Paragraph(
        "Clinical Screening: AUC = 0.716 at threshold 0.049 provides 83% "
        "specificity, meaning negative screens reliably identify healthy "
        "individuals. Sensitivity is modest (59%), and the AUC itself "
        "represents a modest classification result. The ML improvement from "
        "adding asymmetry features (+0.007 F1) is small and should not be "
        "over-interpreted. The primary value of this analysis is the "
        "characterization of asymmetry patterns across pathology subtypes, "
        "not a state-of-the-art classification system.",
        S["body"]))
    story.append(Paragraph(
        "The weak-to-moderate VGA–IMU correlation (ρ = −0.206, r² ≈ 0.042) "
        "indicates that IMU-derived temporal asymmetry captures information "
        "beyond coarse visual assessment, but that VGA alone explains only "
        "~4% of the variance in stride |AI|. Future work should examine "
        "whether combining VGA with IMU metrics improves clinical utility.",
        S["body"]))

    # ── IX. Limitations ──────────────────────────────────────────────────
    story.append(Paragraph("IX. LIMITATIONS", S["section"]))
    story.append(Paragraph(
        "The dataset has several important limitations. (1) The right-dominant "
        "laterality confound in CVA (47/49) and HOA (15/15) means that "
        "asymmetry directionality results for these subtypes are not "
        "generalizable. (2) Small subtype sample sizes (ACL n=9, RIL n=14, "
        "KOA n=14) limit statistical power for per-subtype comparisons; "
        "bootstrap CIs reflect this uncertainty. (3) Results are from a "
        "single cohort in a controlled 10-metre walk protocol; ecological "
        "validity in real-world settings is unknown. (4) The ML models "
        "use 5-fold StratifiedGroupKFold with subject grouping to prevent "
        "data leakage, but replication in an independent dataset is necessary "
        "before any clinical claims.",
        S["body"]))

    # ── X. Conclusion ────────────────────────────────────────────────────
    story.append(Paragraph("X. CONCLUSION", S["section"]))
    story.append(Paragraph(
        "We analyzed temporal gait asymmetry and sensor placement efficiency "
        "in a multi-pathology IMU dataset. Our results suggest that: "
        "(1) healthy subjects exhibit greater temporal asymmetry than "
        "pathological groups, consistent with a motor lateralization "
        "hypothesis; (2) neurological subtypes (RIL, PD, CVA) show the "
        "largest asymmetry reductions; and (3) foot-only sensors may offer "
        "a practical two-sensor clinical configuration. "
        "These findings indicate potential clinical utility of IMU-derived "
        "temporal asymmetry as a gait indicator, subject to validation "
        "in larger independent cohorts and real-world settings.",
        S["body"]))

    # ── References ───────────────────────────────────────────────────────
    story.append(Paragraph("REFERENCES", S["section"]))
    refs = [
        "[1] J. M. Hausdorff, \"Gait dynamics, fractals and falls,\" "
        "J. Neuroeng. Rehabil., 2007.",
        "[2] B. R. Bloem et al., \"Parkinson's disease,\" "
        "Lancet, vol. 397, pp. 2284–2303, 2021.",
        "[3] F. Al-Harthi et al., \"G-MASA-TCN,\" arXiv, 2026.",
        "[4] WHO, \"Neurological Disorders,\" Global Burden of Disease, 2006.",
        "[5] GBD 2016 Parkinson's Disease Collaborators, Lancet Neurol., 2018.",
        "[6] G. A. Donnan et al., \"Stroke,\" Lancet, vol. 371, pp. 1612–1623, 2008.",
        "[7] T. J. Dougherty et al., \"CIPN,\" J. Peripher. Nerv. Syst., 2004.",
        "[8] D. J. Hunter & S. Bierma-Zeinstra, \"Osteoarthritis,\" "
        "Lancet, vol. 393, pp. 1745–1759, 2019.",
        "[9] N. A. Mall et al., \"ACL reconstruction trends,\" "
        "Am. J. Sports Med., vol. 42, 2014.",
        "[10] Y. K. Haddad et al., \"Healthcare spending for falls,\" "
        "Inj. Prev., 2024.",
        "[11] H. Sadeghsalehi, \"A dual-use framework for clinical gait "
        "analysis,\" arXiv:2511.02047, 2025.",
        "[12] C. Voisard et al., \"Automatic gait events detection,\" "
        "J. Neuroeng. Rehabil., 2024.",
        "[13] A. Bonci et al., \"Semiogram: IMU-derived gait visualisation,\" "
        "Front. Neurol., 2021.",
        "[14] R. Sadeghi et al., \"Symmetry and limb dominance in gait,\" "
        "Gait Posture, vol. 12, 2000.",
        "[15] W. Herzog et al., \"Asymmetry in healthy human gait,\" "
        "Clin. Biomech., 1989.",
        "[16] M. Mancini et al., \"Recommended gait domains for PD monitoring,\" "
        "npj Parkinsons Dis., 2025.",
    ]
    for ref in refs:
        story.append(Paragraph(ref, S["body"]))

    return story


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    S = make_styles()
    doc = make_doc()
    story = build_story(S)
    doc.build(story)
    print(f"Saved revised Progress Report → {OUT_PDF}")


if __name__ == "__main__":
    main()
