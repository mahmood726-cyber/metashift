# MetaShift

[![ci](https://github.com/mahmood726-cyber/metashift/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/mahmood726-cyber/metashift/actions/workflows/ci.yml) [![codeql](https://github.com/mahmood726-cyber/metashift/actions/workflows/codeql.yml/badge.svg?branch=master)](https://github.com/mahmood726-cyber/metashift/actions/workflows/codeql.yml) [![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

At what point during the accumulation of primary studies does a cumulative meta-analysis pooled estimate undergo detectable trajectory shifts? We built cumulative DerSimonian-Laird random-effects meta-analyses for ten Cochrane reviews with per-study data from Pairwise70, plus aggregate summaries for 393 additional reviews. Three changepoint algorithms were applied: CUSUM charts for mean shifts, PELT segmentation for variance changes, and a significance-flip detector tracking p-value crossings. Of ten fully traced reviews, the median number of changepoints was two (95% CI 1-4), seven exhibited heterogeneity spikes, and only two stabilized before the accumulation midpoint. CUSUM and PELT agreed on spike location in six of seven cases, and higher final heterogeneity predicted late instability across the full 403-review corpus. Changepoint methods reveal that most cumulative meta-analyses undergo detectable regime shifts that standard forest plots obscure from readers and guideline developers. However, this approach is limited to chronological ordering and cannot account for selective reporting of interim cumulative results by review authors.

**Live dashboard:** <https://mahmood726-cyber.github.io/metashift/>

## Run

Open `metashift.html` (or `index.html`) in any modern browser. No build step.

For local development:

```bash
python -m http.server 8000
# then open http://localhost:8000/
```

## Test

```bash
python -m pytest -q
```

The suite under `tests/` includes 2 test file(s).

## Repo layout

| Path | Purpose |
|---|---|
| `metashift.html` | the dashboard (main artifact) |
| `index.html` | landing page |
| `tests/` | pytest tests |
| `e156-submission/` | E156 micro-paper bundle |
| `E156-PROTOCOL.md` | project metadata (E156 entry #107) |

## License

See `LICENSE` (MIT).
