# MetaShift Design Spec — Changepoint Detection for Cumulative Meta-Analysis

**Date:** 2026-03-26
**Target:** `C:\MetaShift\metashift.html` (new single-file HTML app)
**Build step:** `C:\MetaShift\build_cumulative.py` (new Python script)
**Data sources:** FragilityAtlas specifications (394K rows) + r_validation_inputs.json (10 analyses with per-study data)

## Background

Cumulative meta-analysis adds studies one-by-one (usually chronologically) and re-pools after each addition. Currently, researchers eyeball the cumulative forest plot to judge when the evidence "stabilized." No tool formally detects where the trajectory shifted.

We apply changepoint detection algorithms from manufacturing quality control (CUSUM) and signal processing (PELT) to identify three types of shifts: (1) when the pooled estimate stabilized, (2) when significance status flipped, (3) when heterogeneity spiked. Applied to 403 Cochrane reviews.

## 1. Data Pipeline

### `build_cumulative.py`

For each of the 403 reviews, construct a cumulative meta-analysis trajectory. Two data strategies:

**Strategy A — Per-study data (10 reviews):**
From `C:\FragilityAtlas\data\output\r_validation_inputs.json`, we have actual per-study yi/sei arrays for 10 analyses. For these, compute the true cumulative DL meta-analysis: at step t, pool studies 1..t.

**Strategy B — Specification trajectory (403 reviews):**
From `C:\FragilityAtlas\data\output\fragility_atlas_specifications.csv` (394K rows), each review has ~1,000 specifications (different estimator/CI/bias-correction combinations). These aren't chronological, but we can extract the *default specification* (DL estimator, Wald CI, no bias correction, no leave-out) for each review to get the pooled estimate.

For the cumulative trajectory, use the leave-one-out specifications: the file includes rows where `leave_out` is set to each study index. By comparing the full-data estimate to the leave-k-out estimates, we can reconstruct the *marginal contribution* of each study, which approximates the cumulative trajectory in reverse.

**Alternatively**, compute cumulative meta-analysis directly in Python for the 10 reviews with per-study data, and use the review-level summary (k, theta, tau2, I2, p_value) for the remaining 393 reviews as single-point data (no trajectory — these reviews get a "trajectory unavailable" flag).

**Recommended approach:** Use Strategy A for the 10 reviews with full per-study data to demonstrate the tool. For the corpus-level analysis, compute summary statistics across all 403 reviews using available aggregate data (k, theta, p_value, I2). The custom mode allows users to enter their own study-level data for any meta-analysis.

### Source files

```
C:\FragilityAtlas\data\output\r_validation_inputs.json         (10 analyses, per-study yi/sei)
C:\FragilityAtlas\data\output\fragility_atlas_results.csv      (403 reviews, aggregates)
C:\PredictionGap\data\output\prediction_gap_results.csv        (403 reviews, theta/p/tau2/I2)
```

### Output: `data/cumulative.json`

```json
{
  "reviews_with_trajectory": [
    {
      "review_id": "CD006140",
      "analysis_name": "Acceptability",
      "k": 8,
      "studies": [
        {"idx": 1, "yi": -0.821, "sei": 0.329},
        {"idx": 2, "yi": -0.433, "sei": 0.163}
      ],
      "cumulative": [
        {"step": 1, "est": -0.821, "ci_lo": -1.466, "ci_hi": -0.176, "pval": 0.013, "tau2": 0, "I2": 0, "sig": true},
        {"step": 2, "est": -0.553, "ci_lo": -0.923, "ci_hi": -0.183, "pval": 0.003, "tau2": 0.02, "I2": 12.3, "sig": true}
      ],
      "changepoints": {
        "effect": {"position": 4, "type": "stabilized"},
        "significance": {"position": 2, "type": "became_significant"},
        "heterogeneity": {"position": 6, "type": "spike"}
      },
      "stability_class": "Stable Early"
    }
  ],
  "reviews_aggregate_only": [
    {
      "review_id": "CD000028",
      "analysis_name": "Cause of cardiovascular mortality",
      "k": 21,
      "theta": -0.2945,
      "p_value": 0.000003,
      "I2": 42.3,
      "tau2": 0.05,
      "has_trajectory": false
    }
  ],
  "corpus_summary": {
    "n_with_trajectory": 10,
    "n_aggregate_only": 393,
    "n_total": 403
  }
}
```

## 2. Changepoint Algorithms

All algorithms operate on a numeric sequence of length k (one value per cumulative step).

### 2.1 CUSUM — Pooled estimate stability

Detects the point after which the cumulative pooled estimate stopped drifting.

**Algorithm:**
1. Compute the final pooled estimate `mu_final` as the target
2. For each step t, compute deviation: `d_t = |est_t - mu_final|`
3. CUSUM+: `S_t = max(0, S_{t-1} + d_t - k_slack)` where `k_slack = 0.5 * std(d)`
4. Changepoint: last step where `S_t > h` (threshold `h = 4 * std(d)`)
5. Studies after this point did not meaningfully shift the estimate

**Interpretation:** The changepoint divides the trajectory into "drifting" (before) and "stable" (after).

### 2.2 Binary segmentation — Significance flip

Detects the study that caused the cumulative result to become (or stop being) statistically significant.

**Algorithm:**
1. Create binary sequence: `sig_t = 1 if p_t < 0.05, else 0`
2. For each candidate changepoint c (2..k-1):
   - Compute log-likelihood ratio: `LLR(c) = n1*log(p1_hat) + n2*log(p2_hat) - n*log(p_hat)`
   - Where p1_hat = mean(sig[1..c]), p2_hat = mean(sig[c+1..k]), p_hat = mean(sig[1..k])
3. Changepoint: c that maximizes |LLR(c)|, if max LLR exceeds `3.84` (chi2_{0.05,1})

**Interpretation:** The first study after which significance was consistently maintained (or lost).

### 2.3 PELT-lite — Heterogeneity spike

Detects the point where I² or tau² jumped.

**Algorithm (simplified PELT for single changepoint):**
1. For each candidate changepoint c (2..k-1):
   - Cost of no-change model: `C0 = k * log(var(I2[1..k]))`
   - Cost of change model: `C1 = c * log(var(I2[1..c])) + (k-c) * log(var(I2[c+1..k]))`
   - Save `C0 - C1` as the improvement at c
2. Changepoint: c that maximizes improvement, if improvement > `penalty = 3 * log(k)` (modified BIC)

**Interpretation:** The study whose inclusion caused heterogeneity to spike.

### 2.4 Stability classification

Based on the effect changepoint position relative to total k:
- **Stable Early**: CP < 0.3 * k (evidence settled quickly)
- **Stable Mid**: CP at 0.3-0.6 * k
- **Stable Late**: CP at 0.6-0.9 * k
- **Never Stabilized**: no CP detected, or CP > 0.9 * k

## 3. Dashboard Layout

Single scrollable page, dark mode toggle. Two modes: Corpus (embedded data) and Custom (user input).

### Header

- Title: "MetaShift — When Did the Evidence Change?"
- Subtitle: "Changepoint detection in cumulative meta-analysis"
- Mode toggle: [Corpus | Custom]

### Corpus Mode

**Section 1 — Corpus Summary**: 4 stat boxes (from the 10 reviews with trajectories + 403 aggregate summaries):
1. Reviews with trajectories (10)
2. % with effect changepoint detected
3. % with significance flip
4. % with heterogeneity spike

**Section 2 — Stability Distribution**: Bar chart showing stability classes (Early/Mid/Late/Never) for the 10 trajectory reviews. For the full 403, show aggregate k distribution as context.

**Section 3 — Review Explorer**: Sortable table. For the 10 trajectory reviews: full changepoint data. For the 393 aggregate-only reviews: k, theta, p, I2 (no trajectory). Column: Has Trajectory (yes/no). Click to expand.

**Section 4 — Triple-Panel Trajectory Chart** (accordion detail for reviews with trajectories):

Three vertically stacked SVG charts sharing the same x-axis (study number 1..k):

**Top — Effect trajectory**: Line chart of cumulative pooled estimate with CI band (shaded). Vertical dashed line at effect changepoint. Color: blue before CP, green after.

**Middle — Significance timeline**: Horizontal colored bar at each step: green if p<0.05, grey if not. Vertical dashed line at significance flip point.

**Bottom — Heterogeneity trajectory**: Line chart of I² (0-100%). Vertical dashed line at het changepoint. Optional: tau² on secondary y-axis.

All three share x-axis labels (study 1, 2, ... k).

### Custom Mode

- Input form: rows of [Study name, Year, Effect, SE]
- "Add Study" button, "Load Example" with a built-in dataset
- "Run Analysis" button
- Same triple-panel chart renders on the entered data
- Changepoint results displayed below chart

### Footer

- "MetaShift v1.0 — Browser-based, no data leaves your device."
- Export: [CSV] screening table, [SVG] trajectory chart, [PNG] trajectory chart
- localStorage prefix: `ms_`

## 4. Visual Design

Same portfolio CSS pattern: custom properties, light/dark theming, card layout, responsive.

## 5. Integration Map

| File | Purpose |
|------|---------|
| `C:\MetaShift\build_cumulative.py` | Compute cumulative MA trajectories + changepoints for 10 reviews |
| `C:\MetaShift\data\cumulative.json` | Build artifact |
| `C:\MetaShift\metashift.html` | Single-file HTML dashboard (~1,800 lines) |

## 6. Out of Scope

- Bayesian online changepoint detection
- Multiple changepoints per trajectory (just the most significant one)
- Sequential monitoring boundaries (O'Brien-Fleming, alpha-spending)
- Comparison with living review update triggers
- Trial sequential analysis (TSA)

## 7. Validation

- Cumulative DL for 10 reviews: verify step-1 equals single-study estimate, step-k equals full pooled estimate
- CUSUM: verify on synthetic data (constant sequence → no CP; step-change → CP at step)
- Binary segmentation: verify on alternating 0/1 sequence → CP at transition
- PELT: verify on [low, low, high, high] I² → CP at transition
- Stability classes produce reasonable distribution across 10 reviews
- Custom mode: enter 5 studies, verify trajectory chart and changepoints
