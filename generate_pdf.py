# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "weasyprint>=61",
#     "markdown>=3.5",
#     "pymdown-extensions>=10",
# ]
# ///
"""
Generate a colorful, study-friendly PDF of the RL exam answers.

Each top-level chapter gets its own accent color, applied to its banner,
question "pills", equation boxes, tables, table-of-contents entry, and the
running page header — so the document is easy to navigate and learn from.

Usage:
    uv run generate_pdf.py                 # answers.md -> answers.pdf
    uv run generate_pdf.py in.md out.pdf   # custom input/output
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import html as _html
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown
import weasyprint
from weasyprint import HTML

HERE = Path(__file__).resolve().parent
EQ_DIR = HERE / "eq_svg"          # cache of rendered LaTeX equations (SVG)

# --------------------------------------------------------------------------- #
# LaTeX → SVG rendering. Each $...$ / $$...$$ formula is compiled with `latex`  #
# and converted to a tight, font-free (paths-only) SVG via `dvisvgm`, then      #
# embedded in the HTML. Results are cached by content hash so rebuilds are fast.#
# --------------------------------------------------------------------------- #
TEX_TEMPLATE = r"""\documentclass[12pt]{article}
\usepackage{amsmath,amssymb,amsfonts,bm}
\usepackage[active,tightpage]{preview}
\setlength\PreviewBorder{2pt}
\pagestyle{empty}
\begin{document}\begin{preview}$%s$\end{preview}\end{document}
"""

MATH_DISPLAY_RE = re.compile(r"\$\$(.+?)\$\$", re.S)
MATH_INLINE_RE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", re.S)


def _compile_latex(formula: str, display: bool, out: Path) -> bool:
    """Compile a single formula to SVG at `out`. Returns True on success."""
    snippet = (r"\displaystyle " if display else "") + formula
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "e.tex").write_text(TEX_TEMPLATE % snippet, encoding="utf-8")
        r = subprocess.run(
            ["latex", "-interaction=nonstopmode", "-halt-on-error", "e.tex"],
            cwd=d, capture_output=True, text=True,
        )
        if not (d / "e.dvi").exists():
            sys.stderr.write(f"  ! latex failed for: {formula[:60]}\n")
            return False
        r = subprocess.run(
            ["dvisvgm", "--no-fonts", "--exact-bbox", "e.dvi", "-o", str(out)],
            cwd=d, capture_output=True, text=True,
        )
        return out.exists()


def render_latex(formula: str, display: bool) -> str | None:
    """Return an <img> tag for the formula, compiling+caching as needed."""
    formula = formula.strip()
    key = hashlib.sha1((("D" if display else "I") + formula).encode()).hexdigest()
    svg = EQ_DIR / f"{key}.svg"
    if not svg.exists():
        EQ_DIR.mkdir(exist_ok=True)
        if not _compile_latex(formula, display, svg):
            return None
    cls = "eqimg" if display else "inlimg"
    return f'<img class="{cls}" src="eq_svg/{svg.name}" alt="{_html.escape(formula)}">'


def extract_math(text: str):
    """Pull $$display$$ and $inline$ math out of the Markdown before conversion
    (so Markdown can't mangle backslashes/underscores). Returns text with
    placeholder tokens plus the lists of extracted formulas."""
    disp, inl = [], []

    def d(m):
        disp.append(m.group(1))
        return f"\n\nZZDISP{len(disp) - 1}ZZ\n\n"

    def f(m):
        inl.append(m.group(1))
        return f"ZZINL{len(inl) - 1}ZZ"

    text = MATH_DISPLAY_RE.sub(d, text)
    text = MATH_INLINE_RE.sub(f, text)
    return text, disp, inl


def restore_math(body: str, disp, inl) -> str:
    """Replace placeholder tokens in the converted HTML with rendered SVGs."""
    for i, formula in enumerate(disp):
        img = render_latex(formula, display=True) \
            or f'<span class="eqfail">{_html.escape(formula)}</span>'
        block = f'<div class="eq">{img}</div>'
        body = body.replace(f"<p>ZZDISP{i}ZZ</p>", block).replace(f"ZZDISP{i}ZZ", block)
    for i, formula in enumerate(inl):
        img = render_latex(formula, display=False) \
            or f'<span class="eqfail">{_html.escape(formula)}</span>'
        body = body.replace(f"ZZINL{i}ZZ", f'<span class="math">{img}</span>')
    return body

# --------------------------------------------------------------------------- #
# One palette entry per chapter (in document order). accent = strong color,    #
# tint = very light wash for boxes, icon = monochrome glyph tinted with accent. #
# Colors cycle if there are more chapters than entries.                         #
# --------------------------------------------------------------------------- #
PALETTE = [
    {"accent": "#2d3142", "tint": "#e9eaef", "icon": "★"},  # intro / title
    {"accent": "#e63946", "tint": "#fde3e6", "icon": "●"},  # RL basics
    {"accent": "#f3722c", "tint": "#fde7dc", "icon": "◆"},  # value-based
    {"accent": "#b8860b", "tint": "#f6edcf", "icon": "▲"},  # function approx.
    {"accent": "#2a9d8f", "tint": "#def0ed", "icon": "✦"},  # policy gradient
    {"accent": "#1d6fb8", "tint": "#e0ecf7", "icon": "✸"},  # exploration
    {"accent": "#8338ec", "tint": "#eee3fb", "icon": "⬢"},  # model-based
    {"accent": "#d23391", "tint": "#fbe1f0", "icon": "❖"},  # advanced topics
    {"accent": "#3a45c4", "tint": "#e4e6f8", "icon": "◉"},  # distributional
]

# --------------------------------------------------------------------------- #
# Base stylesheet (chapter-specific rules are appended programmatically).       #
# --------------------------------------------------------------------------- #
BASE_CSS = r"""
@page {
    size: A4;
    margin: 20mm 18mm 18mm 18mm;
    @bottom-center {
        content: "Page " counter(page) " of " counter(pages);
        font-family: "DejaVu Sans", sans-serif;
        font-size: 9pt;
        color: #9aa0aa;
    }
}
@page :first {           /* cover page: no header/footer */
    margin: 0;
    @bottom-center { content: none; }
}

html {
    font-family: "Noto Serif", "DejaVu Serif", Georgia, serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #222428;
}
body { margin: 0; }
p { margin: 7px 0; text-align: left; }
strong { color: #11131a; }
em { color: #444; }

/* ------------------------------ Cover page ---------------------------- */
.cover {
    height: 100vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    color: #fff;
    background: #2d3142;
    page-break-after: always;
}
.cover .kicker {
    font-family: "DejaVu Sans", sans-serif;
    letter-spacing: 6px; text-transform: uppercase;
    font-size: 11pt; opacity: .9; margin-bottom: 16px;
}
.cover h1 {
    font-family: "DejaVu Sans", sans-serif;
    font-size: 40pt; font-weight: 800; line-height: 1.1;
    margin: 0 40px 8px; color: #fff;
    background: none; border: none; padding: 0;
    text-shadow: 0 2px 10px rgba(0,0,0,.25);
}
.cover .rule { width: 110px; height: 4px; background: #fff; opacity:.85; margin: 22px 0; border-radius: 2px; }
.cover .subtitle { font-size: 15pt; opacity: .95; margin: 0 60px; }
.cover .dots { margin-top: 28px; font-size: 20pt; letter-spacing: 8px; }
.cover .meta {
    position: absolute; bottom: 24mm;
    font-family: "DejaVu Sans", sans-serif; font-size: 10pt; opacity: .85;
}

/* --------------------------- Table of contents ------------------------ */
.toc { page-break-after: always; }
.toc h2 {
    font-family: "DejaVu Sans", sans-serif; font-size: 24pt; color: #2d3142;
    margin: 0 0 6px; padding: 0; border: none; background: none;
}
.toc .lead { color:#888; font-family:"DejaVu Sans",sans-serif; font-size:10pt; margin-bottom:18px; }
.toc ul { list-style: none; padding-left: 0; margin: 0; }
.toc .toc-section { margin-top: 14px; }
.toc .toc-section > a {
    font-family: "DejaVu Sans", sans-serif; font-weight: 700; font-size: 12.5pt;
}
.toc .toc-section .tico { font-family:"Symbola","Noto Sans Symbols2","DejaVu Sans"; margin-right:7px; }
.toc .toc-item { margin-left: 26px; font-size: 10.5pt; }
.toc a { text-decoration: none; color: #333; }
.toc li a::after {
    content: " " leader('.') " " target-counter(attr(href), page);
    color: #b7bcc4; font-family: "DejaVu Sans", sans-serif;
}

/* ------------------------- Chapter-scoped styles ---------------------- */
/* Each .ch section sets --accent and --tint inline; everything reads them. */
.ch { page-break-before: always; }

h1 {                                   /* chapter banner */
    font-family: "DejaVu Sans", sans-serif;
    font-size: 21pt; font-weight: 800; color: #fff;
    background: var(--accent, #2d3142);
    padding: 14px 18px; border-radius: 10px;
    margin: 0 0 18px; page-break-after: avoid;
}
h1 .ico {
    font-family: "Symbola","Noto Sans Symbols2","DejaVu Sans";
    color: #fff; opacity: .95; margin-right: 10px;
}

h2 {                                   /* a question — colored "pill" */
    font-family: "DejaVu Sans", sans-serif;
    font-size: 12.5pt; font-weight: 700;
    color: var(--accent, #2d3142);
    background: var(--tint, #eee);
    border-left: 6px solid var(--accent, #2d3142);
    padding: 8px 12px; border-radius: 0 6px 6px 0;
    margin: 22px 0 12px; page-break-after: avoid;
}

h3 {
    font-family: "DejaVu Sans", sans-serif; font-size: 11.5pt;
    color: var(--accent, #2d3142); margin: 16px 0 6px; page-break-after: avoid;
}

/* Equations: tinted card with an accent edge. */
.eq {
    font-family: "DejaVu Sans", "Cambria Math", sans-serif;
    background: var(--tint, #f3f4f6);
    border-left: 4px solid var(--accent, #888);
    padding: 9px 14px; margin: 11px 0; text-align: center;
    font-size: 11pt; border-radius: 6px; page-break-inside: avoid;
}
.eq sub, .eq sup { font-size: 72%; }
.eq img.eqimg { max-width: 100%; height: auto; }
.math img.inlimg { height: 1em; vertical-align: -0.18em; }
.eqfail { font-family: "DejaVu Sans Mono", monospace; color: #b00020; font-size: 9.5pt; }

/* Tables: accent header, tinted zebra. */
table {
    border-collapse: collapse; width: 100%; margin: 12px 0;
    font-size: 10pt; page-break-inside: avoid;
}
th, td { border: 1px solid #d4d8de; padding: 6px 9px; text-align: left; vertical-align: top; }
th { background: var(--accent, #2d3142); color: #fff; font-family: "DejaVu Sans", sans-serif; }
tr:nth-child(even) td { background: var(--tint, #f5f6f8); }

ul, ol { margin: 7px 0; padding-left: 24px; }
li { margin: 4px 0; }
li::marker { color: var(--accent, #555); }

code {
    font-family: "DejaVu Sans Mono", monospace;
    background: #eef1f4; padding: 0 3px; border-radius: 3px; font-size: 9.5pt;
}
hr { border: none; border-top: 2px dotted var(--tint, #e0e0e0); margin: 20px 0; }
a { color: var(--accent, #1d6fb8); }
"""

COVER_HTML = """
<section class="cover">
    <div class="kicker">MIM UW &middot; 2025/2026</div>
    <h1>Reinforcement&nbsp;Learning</h1>
    <div class="rule"></div>
    <div class="subtitle">Comprehensive Exam Answers</div>
    <div class="meta">Generated {date}</div>
</section>
"""

H1_RE = re.compile(r"(<h1\b.*?</h1>)", re.S)
H1_PARSE = re.compile(r'<h1[^>]*\bid="([^"]+)"[^>]*>(.*?)</h1>', re.S)
HEADING_RE = re.compile(r'<(h[12])[^>]*\bid="([^"]+)"[^>]*>(.*?)</\1>', re.S)


def _strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s).strip()


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def load_questions(path: Path):
    """Parse questions.md into an ordered list of sections.

    Returns [{"title": str, "q": {num: exact_text}}, ...] in file order.
    """
    sections = []
    cur = None
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("## "):
            cur = {"title": s[3:].strip().rstrip(":").strip(), "q": {}}
            sections.append(cur)
        elif cur is not None:
            m = re.match(r"^(\d+)\.\s*(.+)$", s)
            if m:
                cur["q"][int(m.group(1))] = m.group(2).strip()
    return sections


def apply_exact_questions(body: str, sections) -> str:
    """Replace each '## N. <paraphrase>' heading text with the exact question
    text from questions.md, matched by chapter order and question number.

    The first chapter (the title/intro page) has no questions and is skipped.
    """
    parts = H1_RE.split(body)            # [pre, h1, content, h1, content, ...]
    headers = H1_RE.findall(body)
    out = [parts[0]]
    for i, h1 in enumerate(headers):
        content = parts[2 * i + 2] if 2 * i + 2 < len(parts) else ""
        sec = sections[i - 1] if 1 <= i <= len(sections) else None
        if sec:
            def repl(m, sec=sec):
                num = int(m.group(2))
                exact = sec["q"].get(num)
                if exact is None:
                    return m.group(0)
                return f"{m.group(1)}{num}. {_html_escape(exact)}{m.group(3)}"

            content = re.sub(
                r"(<h2[^>]*>)\s*(\d+)\.\s*.*?(</h2>)", repl, content, flags=re.S
            )
        out.extend([h1, content])
    return "".join(out)


def colorize_chapters(body: str):
    """Wrap each <h1>..next <h1> span in a colored .ch section.

    Returns (new_body_html, chapters) where chapters lists per-chapter
    metadata used to build the TOC and the per-chapter @page headers.
    """
    pieces = H1_RE.split(body)          # [pre, h1, content, h1, content, ...]
    headers = H1_RE.findall(body)
    pre = pieces[0]

    out = [pre]
    chapters = []
    for i, h1 in enumerate(headers):
        content = pieces[2 * i + 2] if 2 * i + 2 < len(pieces) else ""
        pal = PALETTE[i % len(PALETTE)]
        m = H1_PARSE.search(h1)
        anchor = m.group(1) if m else f"ch{i}"
        title = _strip_tags(m.group(2)) if m else f"Chapter {i}"

        # Inject the chapter icon right after the <h1 ...> opening tag.
        h1_iconed = re.sub(
            r"(<h1[^>]*>)",
            rf'\1<span class="ico">{pal["icon"]}</span>',
            h1, count=1,
        )
        out.append(
            f'<section class="ch ch{i}" '
            f'style="--accent:{pal["accent"]};--tint:{pal["tint"]}">'
            f"{h1_iconed}{content}</section>"
        )
        chapters.append({"idx": i, "anchor": anchor, "title": title, **pal})
    return "".join(out), chapters


def build_toc(body: str) -> str:
    """Two-level clickable TOC, colored per chapter."""
    items = ['<section class="toc"><h2>Contents</h2>'
             '<div class="lead">Click any entry to jump to it.</div><ul>']
    ch = -1
    for tag, anchor, raw in HEADING_RE.findall(body):
        text = _strip_tags(raw)
        if tag == "h1":
            ch += 1
            pal = PALETTE[ch % len(PALETTE)]
            items.append(
                f'<li class="toc-section">'
                f'<a href="#{anchor}" style="color:{pal["accent"]}">'
                f'<span class="tico" style="color:{pal["accent"]}">{pal["icon"]}</span>'
                f"{text}</a></li>"
            )
        else:
            pal = PALETTE[max(ch, 0) % len(PALETTE)]
            items.append(
                f'<li class="toc-item"><a href="#{anchor}">{text}</a></li>'
            )
    items.append("</ul></section>")
    return "".join(items)


def chapter_page_css(chapters) -> str:
    """A named @page per chapter so every page shows a colored chapter header."""
    rules = []
    for c in chapters:
        rules.append(f".ch{c['idx']} {{ page: chap{c['idx']}; }}")
        rules.append(
            f"@page chap{c['idx']} {{ "
            f'@top-left {{ content: "{c["icon"]}  {c["title"]}"; '
            f'color: {c["accent"]}; font-weight: 700; '
            f'font-family: "Symbola","Noto Sans Symbols2","DejaVu Sans",sans-serif; '
            f"font-size: 8.5pt; }} "
            f'@top-right {{ content: "RL — Exam Answers"; color: #c2c7cf; '
            f'font-family: "DejaVu Sans",sans-serif; font-size: 8pt; }} }}'
        )
    return "\n".join(rules)


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "answers.md"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "answers.pdf"

    if not src.exists():
        print(f"error: input file not found: {src}", file=sys.stderr)
        return 1

    md = markdown.Markdown(
        extensions=["extra", "toc", "sane_lists", "smarty", "md_in_html"],
        extension_configs={"toc": {"permalink": False}},
    )
    text, disp, inl = extract_math(src.read_text(encoding="utf-8"))
    print(f"  rendering {len(disp)} display + {len(inl)} inline formulas via LaTeX…")
    raw_body = md.convert(text)
    raw_body = restore_math(raw_body, disp, inl)

    questions_md = HERE / "questions.md"
    if questions_md.exists():
        raw_body = apply_exact_questions(raw_body, load_questions(questions_md))

    body, chapters = colorize_chapters(raw_body)
    toc = build_toc(raw_body)
    cover = COVER_HTML.format(date=_dt.date.today().strftime("%B %d, %Y"))
    css = BASE_CSS + "\n" + chapter_page_css(chapters)

    document = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'></head><body>"
        f"{cover}{toc}{body}</body></html>"
    )

    HTML(string=document, base_url=str(HERE)).write_pdf(
        str(out), stylesheets=[weasyprint.CSS(string=css)]
    )
    print(f"✓ Wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
