"""Convert week_report_week3.md to a clean PDF using reportlab."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, PageBreak,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

# Register CJK font for Chinese text
pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
CJK_FONT = "STSong-Light"

REPO = Path(__file__).resolve().parents[1]
OUT_PDF = REPO / "results" / "week_report_week3.pdf"

# ── Colours ───────────────────────────────────────────────────────────────
NAVY = colors.HexColor("#1a2332")
TEAL = colors.HexColor("#1c7293")
LIGHT_BG = colors.HexColor("#f0f4f8")
GRID = colors.HexColor("#cccccc")
GREEN_CHECK = colors.HexColor("#27ae60")


# ── Styles ────────────────────────────────────────────────────────────────
def _styles():
    S = {}
    S["title"] = ParagraphStyle("title", fontSize=16, fontName="Helvetica-Bold",
                                 alignment=TA_CENTER, spaceAfter=2, textColor=NAVY)
    S["subtitle"] = ParagraphStyle("subtitle", fontSize=9, fontName="Helvetica",
                                    alignment=TA_CENTER, spaceAfter=6,
                                    textColor=colors.HexColor("#555555"))
    S["h1"] = ParagraphStyle("h1", fontSize=12, fontName="Helvetica-Bold",
                              spaceBefore=14, spaceAfter=4, textColor=NAVY)
    S["h2"] = ParagraphStyle("h2", fontSize=10, fontName="Helvetica-Bold",
                              spaceBefore=8, spaceAfter=3, textColor=TEAL)
    S["h3"] = ParagraphStyle("h3", fontSize=9, fontName="Helvetica-Bold",
                              spaceBefore=6, spaceAfter=2)
    S["body"] = ParagraphStyle("body", fontSize=8.5, fontName="Helvetica",
                                alignment=TA_JUSTIFY, leading=12, spaceAfter=4)
    S["body_bold"] = ParagraphStyle("body_bold", parent=S["body"],
                                     fontName="Helvetica-Bold")
    S["bullet"] = ParagraphStyle("bullet", parent=S["body"], leftIndent=14,
                                  bulletIndent=4)
    S["key"] = ParagraphStyle("key", fontSize=8.5, fontName="Helvetica-Oblique",
                               alignment=TA_LEFT, leading=12,
                               leftIndent=8, rightIndent=8, borderPadding=4,
                               backColor=LIGHT_BG, spaceAfter=6, spaceBefore=4)
    S["caption"] = ParagraphStyle("caption", fontSize=7, fontName="Helvetica-Oblique",
                                   alignment=TA_CENTER, spaceAfter=4, spaceBefore=2)
    S["cn_h1"] = ParagraphStyle("cn_h1", fontSize=12, fontName=CJK_FONT,
                                 spaceBefore=14, spaceAfter=4, textColor=NAVY)
    S["cn_h2"] = ParagraphStyle("cn_h2", fontSize=10, fontName=CJK_FONT,
                                 spaceBefore=8, spaceAfter=3, textColor=TEAL)
    S["cn_body"] = ParagraphStyle("cn_body", fontSize=8.5, fontName=CJK_FONT,
                                   leading=13, spaceAfter=4)
    return S


# ── Table helper ──────────────────────────────────────────────────────────
HDR_COLOR = colors.HexColor("#2c3e50")
ROW_ALT = colors.HexColor("#f7f9fb")


def styled_table(data, col_widths=None, font_size=7):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), HDR_COLOR),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.3, GRID),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ROW_ALT]),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]
    t.setStyle(TableStyle(style))
    return t


# ── Build story ───────────────────────────────────────────────────────────
def build(S):
    story = []
    sp = lambda pts: Spacer(1, pts)
    hr = lambda: HRFlowable(width="100%", thickness=0.5, color=GRID, spaceAfter=6)

    # ── Title ─────────────────────────────────────────────────────────────
    story.append(Paragraph("BMED 712 - Track A | Week 3 Report", S["title"]))
    story.append(Paragraph(
        "Date: 2026-04-06  |  Team: Fatima Habib Farweh, Liang Li, "
        "Yasmine Khattab, Zehara Ali", S["subtitle"]))
    story.append(Paragraph(
        "Period: Week 3 (post-submission - revision, debugging, retraining)",
        S["subtitle"]))
    story.append(hr())

    # ── 1. Completed Tasks ────────────────────────────────────────────────
    story.append(Paragraph("1. Summary of Completed Tasks", S["h1"]))
    tasks = [
        ["#", "Task", "Status"],
        ["1", "Address all 6 professor feedback items", "Done"],
        ["2", "Identify and document feature extraction bug", "Done"],
        ["3", "Validate corrected frequency-sheet features (300k windows)", "Done"],
        ["4", "Retrain 3-class models (SVM / XGBoost / RF)", "Done"],
        ["5", "First-ever 8-class (subtype) classification", "Done"],
        ["6", "Sensor ablation with new features", "Done"],
        ["7", "Phase-specific asymmetry analysis", "Done"],
        ["8", "VGA-IMU ordinal correlation analysis", "Done"],
        ["9", "Expanded feature experiment (proxy features)", "Done"],
        ["10", "Revised Progress Report PDF", "Done"],
    ]
    story.append(styled_table(tasks, [0.3 * inch, 4.0 * inch, 0.6 * inch]))
    story.append(sp(6))

    # ── 2. Bug Analysis ───────────────────────────────────────────────────
    story.append(Paragraph("2. Feature Extraction Bug - Root Cause Analysis", S["h1"]))

    story.append(Paragraph("Bug 1 - Missing Acc channel", S["h3"]))
    story.append(Paragraph(
        "The extraction pipeline iterated over FreeAcc and Gyr signals only, "
        "<b>silently skipping the raw Acc channel</b>. Each of the 4 IMUs provides "
        "3 signal types (Acc, FreeAcc, Gyr) x 3 axes, so 1/3 of available signals "
        "were never processed. The resulting file had 168 features instead of 216+.",
        S["body"]))

    story.append(Paragraph("Bug 2 - Trial-level aggregation", S["h3"]))
    story.append(Paragraph(
        "The old code averaged all time windows within each trial into a single row "
        "(1,356 rows = one per trial), collapsing within-trial temporal variability "
        "- a critical source of discriminative information for gait classification.",
        S["body"]))

    story.append(Paragraph("Bug 3 - No subtype label", S["h3"]))
    story.append(Paragraph(
        "The original CSV had only a 3-class label (Healthy / Neuro / Ortho). The "
        "8-class cohort column (HS / PD / CVA / RIL / CIPN / KOA / HOA / ACL) was "
        "never written, making subtype analysis impossible.",
        S["body"]))

    story.append(Paragraph("Resolution", S["h3"]))
    story.append(Paragraph(
        "Teammate Fatemah re-extracted features from scratch: all 3 channels included, "
        "window-level rows (300,991 windows), both 3-class and 8-class labels, "
        "216 features per window (4 sensors x 3 channels x 3 axes x 6 feature types). "
        "Missing-value rate: &lt; 0.12%.",
        S["body"]))
    story.append(hr())

    # ── 3. ML Results ─────────────────────────────────────────────────────
    story.append(Paragraph("3. ML Results - Corrected Features", S["h1"]))

    story.append(Paragraph("3.1 Three-class classification", S["h2"]))
    t3c = [
        ["Phase", "Window", "Overlap", "Best Model", "BAcc", "F1", "vs Old"],
        ["post_uturn", "5 s", "50%", "XGBoost", "79.2%", "80.4%", "+7.6%"],
        ["pre_uturn", "5 s", "50%", "XGBoost", "76.7%", "76.2%", "+5.1%"],
        ["full_gait", "3 s", "50%", "XGBoost", "76.1%", "75.9%", "+4.5%"],
        ["full_gait", "5 s", "50%", "SVM", "75.8%", "75.0%", "+4.2%"],
        ["uturn", "1 s", "50%", "SVM", "76.5%", "75.3%", "+4.9%"],
    ]
    story.append(styled_table(t3c, [0.8*inch, 0.5*inch, 0.5*inch, 0.7*inch,
                                     0.55*inch, 0.55*inch, 0.55*inch]))
    story.append(Paragraph("Old best (buggy features): BAcc = 71.6%.", S["caption"]))

    story.append(Paragraph("3.2 Eight-class classification (subtype-level) - NEW",
                            S["h2"]))
    t8c = [
        ["Phase", "Model", "BAcc", "F1", "Note"],
        ["full_gait 5s/50%", "SVM", "41.5%", "35.9%", "Chance = 12.5%"],
        ["pre_uturn 5s/50%", "SVM", "41.0%", "36.9%", ""],
        ["post_uturn 6s/50%", "SVM", "39.9%", "35.2%", ""],
    ]
    story.append(styled_table(t8c, [1.2*inch, 0.6*inch, 0.6*inch, 0.5*inch, 1.0*inch]))
    story.append(Paragraph(
        "BAcc of 41.5% is 3.3x chance level, confirming IMU features carry "
        "subtype-discriminative signal.", S["key"]))

    story.append(Paragraph("3.3 Sensor ablation (full_gait 5s/50%, 3-class)", S["h2"]))
    tsa = [
        ["Sensor Set", "SVM", "XGBoost", "RF"],
        ["All (HE+LB+LF+RF)", "75.8%", "75.7%", "75.3%"],
        ["HE+LB", "72.9%", "75.1%", "73.5%"],
        ["HE only", "72.7%", "74.0%", "73.2%"],
        ["Feet (LF+RF)", "70.4%", "69.5%", "67.1%"],
        ["RF only", "67.5%", "67.9%", "63.7%"],
        ["LB only", "71.0%", "69.4%", "64.8%"],
    ]
    story.append(styled_table(tsa, [1.3*inch, 0.7*inch, 0.7*inch, 0.7*inch]))
    story.append(Paragraph(
        "Key surprise: HE (head) sensor alone achieves 74.0% BAcc with XGBoost "
        "- retaining 97.8% of full-sensor performance.", S["key"]))

    story.append(Paragraph("3.4 Expanded feature experiment", S["h2"]))
    story.append(Paragraph(
        "Added 5 derived features per sensor-channel-axis (energy, DC ratio, "
        "relative variability, spectral complexity, normalized spectral power): "
        "216 to 396 features. <b>Result: no improvement</b> (SVM 73.9% vs 75.8% original). "
        "Proxy features are correlated with existing stats. Raw signal access is "
        "needed for genuinely new features (kurtosis, ZCR, wavelet).",
        S["body"]))
    story.append(hr())

    # ── 4. Professor Feedback ─────────────────────────────────────────────
    story.append(Paragraph("4. Professor's Feedback - All 6 Items Addressed", S["h1"]))
    tfb = [
        ["#", "Feedback", "Fix Applied"],
        ["1", "Tone down language", "\"potential indicators\", \"suggests\""],
        ["2", "Streamline narrative", "Abstract: 2 contributions; Discussion: 3 paragraphs"],
        ["3", "ML improvement small", "AUC labeled \"modest\"; clinical insight emphasized"],
        ["4", "Sensor ablation key-takeaway", "Key-finding box: foot-only retains 93%"],
        ["5", "Fig 7: remove regression line", "Spearman rho annotated; per-VGA boxplots"],
        ["6", "Table II: signed Cohen's d", "All d negative: RIL -0.87, PD -0.77, ..."],
    ]
    story.append(styled_table(tfb, [0.3*inch, 1.8*inch, 2.8*inch]))
    story.append(hr())

    # ── 5. Key Insights ───────────────────────────────────────────────────
    story.append(Paragraph("5. Key Insights This Week", S["h1"]))

    insights = [
        ("<b>Feature extraction matters more than model tuning.</b> "
         "Fixing the missing Acc channel and using window-level data improved BAcc by "
         "+7.6% - more than any hyperparameter search could achieve."),
        ("<b>Head sensor is underrated.</b> "
         "HE alone (74.0%) nearly matches all 4 sensors (75.7%). A single "
         "head-mounted IMU could be a practical clinical screening device."),
        ("<b>Subtype discrimination is feasible.</b> "
         "8-class BAcc of 41.5% (vs 12.5% chance) shows IMU features carry "
         "subtype-specific signatures."),
        ("<b>Straight-line walking carries the signal.</b> "
         "Pre-U-turn phase shows significant group differences (p = 0.026); "
         "yet ML performs best on post_uturn (79.2%) - the return corridor has "
         "cleaner signal."),
        ("<b>VGA is a coarse proxy.</b> "
         "Spearman rho = -0.206 means VGA explains only ~4% of IMU asymmetry "
         "variance. IMU captures information invisible to clinical visual assessment."),
    ]
    for i, txt in enumerate(insights, 1):
        story.append(Paragraph(f"{i}. {txt}", S["bullet"]))
    story.append(hr())

    # ── 6. Next Steps ─────────────────────────────────────────────────────
    story.append(Paragraph("6. Proposed Next Steps (Week 4)", S["h1"]))
    nexts = [
        "Access raw IMU signals to compute genuinely new features "
        "(kurtosis, skewness, zero-crossing rate, wavelet energy)",
        "Re-run 8-class with SMOTE or class-weighted loss to address "
        "RIL/ACL imbalance",
        "Phase-feature fusion: concatenate pre_uturn + post_uturn features per trial",
        "Discuss with professor: is head-sensor-only IMU a publishable finding?",
        "Begin final report structure planning",
    ]
    for n in nexts:
        story.append(Paragraph(f"- {n}", S["bullet"]))
    story.append(hr())

    # ── 7. Files ──────────────────────────────────────────────────────────
    story.append(Paragraph("7. Files Delivered This Week", S["h1"]))
    tfiles = [
        ["File", "Description"],
        ["analysis/validate_new_features.py", "Feature validation (300k windows)"],
        ["analysis/train_new_features.py", "3-class, 8-class, sensor ablation"],
        ["analysis/expand_features.py", "Expanded feature experiment"],
        ["analysis/fix_fig7_table2.py", "Corrected Fig 7 + signed Table II"],
        ["results/ml_new_features/", "All ML result CSVs and plots"],
        ["results/validation/", "Validation outputs (heatmaps, KDE)"],
        ["results/Progress_Report_Revised.pdf", "Revised report (all feedback)"],
    ]
    story.append(styled_table(tfiles, [2.5 * inch, 3.0 * inch]))

    # ── Page break → Chinese version ─────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Week 3 Report (Chinese)", S["cn_h1"]))
    story.append(hr())

    story.append(Paragraph("一、核心发现：特征提取 Bug 分析", S["cn_h2"]))
    story.append(Paragraph(
        "旧版 master_features.csv 存在两个复合错误：<br/>"
        "<b>错误1 - Acc 通道缺失：</b>旧代码只处理了 FreeAcc 和 Gyr，"
        "完全跳过了原始 Acc 通道。每个 IMU 有 3 种信号 x 3 轴，旧版丢失了 "
        "1/3 的信号源（168 vs 216 特征）。<br/>"
        "<b>错误2 - 聚合粒度错误：</b>旧代码将每个 trial 内窗口取均值（1,356 行），"
        "丢失了 trial 内时间变异性。<br/>"
        "<b>错误3 - 无亚型标签：</b>只有 3 类标签，无 8 类 cohort 标签。<br/>"
        "<b>修复：</b>Fatemah 重新提取全部特征：300,991 个窗口，"
        "3 通道 + 8 类标签，缺失率 &lt; 0.12%。",
        S["cn_body"]))

    story.append(Paragraph("二、修正后 ML 结果", S["cn_h2"]))
    tcn = [
        ["Experiment", "Best Model", "BAcc", "vs Old"],
        ["3-class (post_uturn 5s/50%)", "XGBoost", "79.2%", "+7.6%"],
        ["3-class (full_gait 5s/50%)", "SVM", "75.8%", "+4.2%"],
        ["8-class (full_gait 5s/50%)", "SVM", "41.5%", "New (chance 12.5%)"],
    ]
    story.append(styled_table(tcn, [1.8*inch, 0.8*inch, 0.7*inch, 1.2*inch]))
    story.append(Paragraph(
        "传感器消融新发现：头部 (HE) 单传感器 XGBoost 达到 74.0% BAcc，"
        "接近全部 4 传感器 (75.7%)。扩展特征实验无提升 (396 vs 216 特征)。",
        S["cn_body"]))

    story.append(Paragraph("三、导师反馈 - 全部 6 条已回应", S["cn_h2"]))
    story.append(Paragraph(
        "1. 语气软化 (\"robust biomarkers\" -> \"potential indicators\")<br/>"
        "2. 叙事精简 (摘要重写，贡献前置)<br/>"
        "3. ML 定位调整 (AUC 0.716 标注 \"modest\")<br/>"
        "4. 传感器消融关键结论 (foot-only 保留 93%)<br/>"
        "5. 图7修正 (删除回归线，改用 Spearman + 箱线图)<br/>"
        "6. 表II修正 (Cohen's d 全部改为负值)",
        S["cn_body"]))

    story.append(Paragraph("四、本周关键洞见", S["cn_h2"]))
    story.append(Paragraph(
        "1. <b>特征工程 > 模型调参：</b>修复 Acc 缺失 + 窗口级数据带来 +7.6% BAcc。<br/>"
        "2. <b>头部传感器被低估：</b>HE 单传感器 (74.0%) 几乎匹配全部 4 传感器。<br/>"
        "3. <b>亚型分类可行：</b>8 类 BAcc 41.5% (随机 12.5%) 证明 IMU 携带亚型信号。<br/>"
        "4. <b>直线行走 = 主信号源：</b>pre-U-turn p=0.026 显著；ML 在 post-uturn "
        "最佳 (79.2%)。<br/>"
        "5. <b>VGA 是粗糙代理：</b>Spearman rho = -0.206，仅解释 ~4% 方差。",
        S["cn_body"]))

    story.append(Paragraph("五、下周计划", S["cn_h2"]))
    story.append(Paragraph(
        "- 获取原始 IMU 信号，提取峰度/过零率/小波能量等新特征<br/>"
        "- 8 类分类使用 SMOTE 或类别加权损失缓解不平衡<br/>"
        "- 尝试 pre_uturn + post_uturn 阶段特征融合<br/>"
        "- 与导师讨论：头部单传感器发现是否可作为可发表亮点<br/>"
        "- 开始规划 final report 结构",
        S["cn_body"]))

    return story


# ── Footer ────────────────────────────────────────────────────────────────
def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.grey)
    w, _ = letter
    canvas.drawCentredString(
        w / 2, 0.4 * inch,
        f"BMED 712 - Track A Week 3 Report  |  Spring 2026  |  Page {doc.page}")
    canvas.restoreState()


def main():
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT_PDF), pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.7 * inch, bottomMargin=0.65 * inch,
    )
    S = _styles()
    story = build(S)
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    print(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
