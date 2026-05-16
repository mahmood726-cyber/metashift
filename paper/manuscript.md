# MetaShift: Changepoint Detection Reveals Hidden Regime Shifts in Cumulative Meta-Analysis

## Overview

CUSUM and PELT changepoint algorithms detect hidden trajectory shifts in cumulative meta-analyses. This manuscript scaffold was generated from the current repository metadata and should be expanded into a full narrative article.

## Study Profile

Type: methods
Primary estimand: Changepoint detection rate
App: MetaShift v1.0
Data: Pairwise70 dataset (10 reviews with trajectories, 393 aggregate)
Code: https://github.com/mahmood726-cyber/metashift

## E156 Capsule

At what point during the accumulation of primary studies does a cumulative meta-analysis pooled estimate undergo detectable trajectory shifts? We built cumulative DerSimonian-Laird random-effects meta-analyses for ten Cochrane reviews with per-study data from Pairwise70, plus aggregate summaries for 393 additional reviews. Three changepoint algorithms were applied: CUSUM charts for mean shifts, PELT segmentation for variance changes, and a significance-flip detector tracking p-value crossings. Of ten fully traced reviews, the median number of changepoints was two (95% CI 1-4), seven exhibited heterogeneity spikes, and only two stabilized before the accumulation midpoint. CUSUM and PELT agreed on spike location in six of seven cases, and higher final heterogeneity predicted late instability across the full 403-review corpus. Changepoint methods reveal that most cumulative meta-analyses undergo detectable regime shifts that standard forest plots obscure from readers and guideline developers. However, this approach is limited to chronological ordering and cannot account for selective reporting of interim cumulative results by review authors.

## Expansion Targets

1. Expand the background and rationale into a full introduction.
2. Translate the E156 capsule into detailed methods, results, and discussion sections.
3. Add figures, tables, and a submission-ready reference narrative around the existing evidence object.
