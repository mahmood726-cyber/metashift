Mahmood Ahmad
Tahir Heart Institute
mahmood.ahmad2@nhs.net

MetaShift: Changepoint Detection Reveals Hidden Regime Shifts in Cumulative Meta-Analysis

At what point during the accumulation of primary studies does a cumulative meta-analysis pooled estimate undergo detectable trajectory shifts? We built cumulative DerSimonian-Laird random-effects meta-analyses for ten Cochrane reviews with per-study data from Pairwise70, plus aggregate summaries for 393 additional reviews. Three changepoint algorithms were applied: CUSUM charts for mean shifts, PELT segmentation for variance changes, and a significance-flip detector tracking p-value crossings. Of ten fully traced reviews, the median number of changepoints was two (95% CI 1-4), seven exhibited heterogeneity spikes, and only two stabilized before the accumulation midpoint. CUSUM and PELT agreed on spike location in six of seven cases, and higher final heterogeneity predicted late instability across the full 403-review corpus. Changepoint methods reveal that most cumulative meta-analyses undergo detectable regime shifts that standard forest plots obscure from readers and guideline developers. However, this approach is limited to chronological ordering and cannot account for selective reporting of interim cumulative results by review authors.

Outside Notes

Type: methods
Primary estimand: Changepoint detection rate
App: MetaShift v1.0
Data: Pairwise70 dataset (10 reviews with trajectories, 393 aggregate)
Code: https://github.com/mahmood726-cyber/metashift
Version: 1.0
Validation: DRAFT

References

1. Borenstein M, Hedges LV, Higgins JPT, Rothstein HR. Introduction to Meta-Analysis. 2nd ed. Wiley; 2021.
2. Higgins JPT, Thompson SG, Deeks JJ, Altman DG. Measuring inconsistency in meta-analyses. BMJ. 2003;327(7414):557-560.
3. Cochrane Handbook for Systematic Reviews of Interventions. Version 6.4. Cochrane; 2023.

AI Disclosure

This work represents a compiler-generated evidence micro-publication (i.e., a structured, pipeline-based synthesis output). AI is used as a constrained synthesis engine operating on structured inputs and predefined rules, rather than as an autonomous author. Deterministic components of the pipeline, together with versioned, reproducible evidence capsules (TruthCert), are designed to support transparent and auditable outputs. All results and text were reviewed and verified by the author, who takes full responsibility for the content. The workflow operationalises key transparency and reporting principles consistent with CONSORT-AI/SPIRIT-AI, including explicit input specification, predefined schemas, logged human-AI interaction, and reproducible outputs.
