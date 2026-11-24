# Submissions

One folder per venue. Each is **self-contained**: it carries its own `figures/`,
`references.bib`, style files, and `.bbl`, so it can be zipped and uploaded
as-is, or compiled without reaching outside its own directory.

| Folder | Venue | Style files | Status |
|---|---|---|---|
| [`neurips/`](neurips) | NeurIPS 2026 | `neurips_2026.sty` (`[preprint]`) | **main** — see [`neurips/reviews/`](neurips/reviews) |
| [`colm/`](colm) | COLM 2026 | `COLM_2026_Template/colm2026_conference.sty` (`[submission]`) | |
| [`aamas/`](aamas) | AAMAS 2026 | `aamas.cls` (`[sigconf]`), `ACM-Reference-Format.bst` | camera-ready format |

The body text and all figure references are shared across the three; they differ
in preamble, bibliography backend, and float sizing. `neurips/` additionally
includes `checklist.tex` (the NeurIPS paper checklist).

## Building

Every folder uses the same entry point, so:

```sh
cd neurips        # or colm, aamas
latexmk -pdf main.tex
```

Or by hand — note the bibliography backend differs per venue:

```sh
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

A pre-built `main.bbl` ships in each folder, so a plain `pdflatex` pass works
without running BibTeX (required for arXiv and most submission portals).

## Figures

`figures/` is duplicated per folder rather than shared, because LaTeX resolves
`\includegraphics{figures/...}` relative to the compile directory. Regenerate
the plots from `../analysis/` and copy into all three folders to keep them in
sync:

| File | Kind |
|---|---|
| `mini-mafia.tex`, `benchmark.tex`, `benchmark-background.tex` | TikZ, `\input`-ed |
| `scores_theoretical_combined.png`, `scores_z_combined.png` | results |
| `cv_scatter.png`, `calibration_plot.png` | validation |
| `overlay_deceive_z_scores.png` | prompt-sensitivity appendix |

The Beamer talk lives in [`../presentation/`](../presentation) with its own
`figures/` copy.
