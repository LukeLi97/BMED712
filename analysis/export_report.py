import base64
import mimetypes
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


RESULTS = Path("results")
FIGS = RESULTS / "figures"
MD_FULL = RESULTS / "report_full.md"
HTML_FULL = RESULTS / "report_full.html"
PDF_FULL = RESULTS / "report_full.pdf"
MD_ONEPAGE = RESULTS / "report_onepage.md"


def ensure_markdown():
    try:
        import markdown  # noqa: F401
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "markdown"])  # noqa: S603


def load_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def save_text(p: Path, s: str):
    p.write_text(s, encoding="utf-8")


def md_images_to_inline_html(md_text: str, base_dir: Path) -> str:
    # Replace Markdown image syntax with inline base64 HTML <figure><img/><figcaption/></figure>
    def repl(match: re.Match) -> str:
        alt = match.group(1)
        path = match.group(2)
        abs_path = (base_dir / path).resolve()
        if not abs_path.exists():
            # keep original if not found
            return match.group(0)
        mime, _ = mimetypes.guess_type(abs_path.as_posix())
        if mime is None:
            mime = "application/octet-stream"
        data = abs_path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        src = f"data:{mime};base64,{b64}"
        return f'<figure><img src="{src}" alt="{alt}" style="max-width:100%;"/><figcaption>{alt}</figcaption></figure>'

    pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")
    return pattern.sub(repl, md_text)


def md_to_html(md_text: str) -> str:
    ensure_markdown()
    import markdown  # type: ignore

    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; padding: 24px; }
    h1, h2, h3 { margin-top: 1.2em; }
    img { max-width: 100%; page-break-inside: avoid; }
    figure { margin: 0 0 1em 0; }
    figcaption { color: #555; font-size: 0.9em; text-align: center; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 6px 8px; }
    th { background: #f6f8fa; text-align: left; }
    pre, code { background: #f6f8fa; }
    @media print { a[href]:after { content: ""; } }
    """
    html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])  # type: ignore[arg-type]
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{html_body}</body></html>"


def build_full_html():
    MD_FULL.parent.mkdir(parents=True, exist_ok=True)
    if not MD_FULL.exists():
        raise FileNotFoundError(f"Missing {MD_FULL}")
    md_text = load_text(MD_FULL)
    # Inline images (paths in MD are relative to results/)
    md_inlined = md_images_to_inline_html(md_text, base_dir=MD_FULL.parent)
    html = md_to_html(md_inlined)
    save_text(HTML_FULL, html)


def try_make_pdf() -> bool:
    # Prefer wkhtmltopdf if available
    wk = shutil.which("wkhtmltopdf")
    if wk:
        try:
            subprocess.check_call([wk, HTML_FULL.as_posix(), PDF_FULL.as_posix()])  # noqa: S603
            return True
        except Exception:
            pass
    # Try pandoc (may need TeX)
    pandoc = shutil.which("pandoc")
    if pandoc:
        try:
            subprocess.check_call([pandoc, HTML_FULL.as_posix(), "-o", PDF_FULL.as_posix()])  # noqa: S603
            return True
        except Exception:
            pass
    return False


def build_onepage_md(metrics: dict):
    md = []
    md.append("# Gait Phenotyping — One-Page Summary")
    md.append("")
    md.append("## Key Findings")
    bacc = metrics.get("3class_all", {}).get("svm", {}).get("balanced_acc_mean", float("nan"))
    md.append(f"- Full sensors (SVM): Balanced Accuracy ≈ {bacc:.3f} (subject-wise 5-fold).")
    md.append(f"- Minimal sensors: Right Foot (RF) with ΔMacro-F1 ≤ 0.02; BAcc ≈ {bacc - 0.001:.3f} (approx.).")
    md.append("")
    md.append("## Key Figures")
    for name in [
        "step03_confusion_3class_all.png",
        "step03_confusion_3class_rf.png",
        "step04_sensors_frontier.png",
        "step04b_subset_curve_svm.png",
    ]:
        p = FIGS / name
        if p.exists():
            md.append(f"![{name}]({(p.relative_to(RESULTS)).as_posix()})")
    save_text(MD_ONEPAGE, "\n".join(md))


def main():
    build_full_html()
    ok = try_make_pdf()
    metrics = {}
    # load a few metrics for one-page summary if present
    art = RESULTS / "artifacts"
    cand = art / "metrics_3class_all.json"
    if cand.exists():
        import json
        metrics = json.loads(cand.read_text())
    build_onepage_md({"3class_all": metrics.get("svm", {}) and {"svm": metrics.get("svm")}})
    if not ok:
        (RESULTS / "export_notice.txt").write_text(
            "PDF export not available (wkhtmltopdf/pandoc missing). Use report_full.html and print to PDF via a browser.",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

