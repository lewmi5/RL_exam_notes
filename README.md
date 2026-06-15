# Reinforcement Learning — Exam Answers (MIM UW, 2025/2026)

Comprehensive, self-contained answers to the RL exam questions, with the
formulas explained step by step. The Markdown source is rendered into a
colorful, study-friendly PDF (per-chapter accent colors, clickable table of
contents, LaTeX-typeset equations).

## Contents

| File | What it is |
|------|------------|
| [answers.md](answers.md) | The answers — one section per topic, with step-by-step formula explanations. |
| [questions.md](questions.md) | The exam questions, used to insert exact question text into the PDF. |
| [generate_pdf.py](generate_pdf.py) | Build script: `answers.md` → `rl-exam-answers.pdf`. |
| [rl-exam-answers.pdf](rl-exam-answers.pdf) | The generated PDF (regenerate with the script). |
| `eq_svg/` | Cache of rendered equations (one SVG per formula). Safe to delete; it will be rebuilt. |

## Building the PDF

**Option A — [uv](https://docs.astral.sh/uv/) (no manual install):**

```bash
uv run generate_pdf.py
```

**Option B — pip:**

```bash
pip install -r requirements.txt
python generate_pdf.py
```

Custom input/output:

```bash
python generate_pdf.py input.md output.pdf
```

## Prerequisites

- **Python ≥ 3.10** and the packages in [requirements.txt](requirements.txt).
- **A LaTeX distribution (TeX Live)** providing `latex` and `dvisvgm` — each
  `$…$` / `$$…$$` formula is compiled to SVG and embedded in the PDF.
  On Debian/Ubuntu:

  ```bash
  sudo apt install texlive-latex-extra dvisvgm
  ```

The first build compiles every equation (slower); subsequent builds reuse the
`eq_svg/` cache and are fast.
