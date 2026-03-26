# MetaShift Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-file HTML dashboard that applies changepoint detection (CUSUM, binary segmentation, PELT) to cumulative meta-analysis trajectories, identifying when evidence stabilized, significance flipped, and heterogeneity spiked.

**Architecture:** Two-phase build: (1) Python script computes cumulative DL meta-analysis for 10 reviews with per-study data, runs changepoint detection, outputs JSON. (2) Single-file HTML dashboard with triple-panel trajectory charts, corpus summary, per-review screening table, and custom-input mode.

**Tech Stack:** Python 3.x (json, math, csv), vanilla HTML/CSS/JS, SVG for charts.

**Spec:** `C:\MetaShift\docs\superpowers\specs\2026-03-26-metashift-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `C:\MetaShift\build_cumulative.py` | Compute cumulative MA + changepoints for 10 per-study reviews |
| `C:\MetaShift\data\cumulative.json` | Build artifact: trajectories + changepoints |
| `C:\MetaShift\metashift.html` | Single-file HTML dashboard (~1,800 lines) |

---

### Task 1: Build `build_cumulative.py` — cumulative MA + changepoints

**Files:**
- Create: `C:\MetaShift\build_cumulative.py`
- Output: `C:\MetaShift\data\cumulative.json`

**Context:** Load per-study data from `r_validation_inputs.json` (10 analyses with yi/sei arrays). For each, compute the cumulative DerSimonian-Laird meta-analysis (pool studies 1..t for t=1..k). Then run three changepoint algorithms on the trajectories. Also load review-level aggregates for the remaining 393 reviews.

- [ ] **Step 1: Write build_cumulative.py**

```python
"""Build cumulative meta-analysis trajectories + changepoint detection.

For 10 reviews with per-study data: full cumulative DL + 3 changepoint algorithms.
For 393 reviews: aggregate data only (no trajectory).

Output: data/cumulative.json
"""

import json
import math
import csv
from pathlib import Path


def dl_meta(yi, sei):
    """DerSimonian-Laird random-effects meta-analysis."""
    k = len(yi)
    if k == 0:
        return None
    if k == 1:
        return {
            'est': yi[0], 'se': sei[0],
            'ci_lo': yi[0] - 1.96 * sei[0], 'ci_hi': yi[0] + 1.96 * sei[0],
            'pval': 2 * (1 - normal_cdf(abs(yi[0] / sei[0]))) if sei[0] > 0 else 1,
            'tau2': 0, 'I2': 0, 'Q': 0,
        }

    vi = [s * s for s in sei]
    wi = [1.0 / v for v in vi]
    sum_w = sum(wi)
    mu_fe = sum(w * y for w, y in zip(wi, yi)) / sum_w

    Q = sum(w * (y - mu_fe) ** 2 for w, y in zip(wi, yi))
    sum_w2 = sum(w * w for w in wi)
    C = sum_w - sum_w2 / sum_w
    tau2 = max(0, (Q - (k - 1)) / C) if C > 0 else 0

    wi_re = [1.0 / (v + tau2) for v in vi]
    sum_w_re = sum(wi_re)
    mu_re = sum(w * y for w, y in zip(wi_re, yi)) / sum_w_re
    se_re = math.sqrt(1.0 / sum_w_re)

    ci_lo = mu_re - 1.96 * se_re
    ci_hi = mu_re + 1.96 * se_re
    z = abs(mu_re / se_re) if se_re > 0 else 0
    pval = 2 * (1 - normal_cdf(z))
    I2 = max(0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0

    return {
        'est': round(mu_re, 6), 'se': round(se_re, 6),
        'ci_lo': round(ci_lo, 6), 'ci_hi': round(ci_hi, 6),
        'pval': round(pval, 8), 'tau2': round(tau2, 6),
        'I2': round(I2, 1), 'Q': round(Q, 4),
    }


def normal_cdf(x):
    """Standard normal CDF (Abramowitz & Stegun 26.2.17)."""
    if x < -8:
        return 0
    if x > 8:
        return 1
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989422802 * math.exp(-x * x / 2)
    p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
    return 1 - p if x > 0 else p


def cumulative_ma(yi, sei):
    """Compute cumulative DL meta-analysis: pool studies 1..t for t=1..k."""
    trajectory = []
    for t in range(1, len(yi) + 1):
        result = dl_meta(yi[:t], sei[:t])
        result['step'] = t
        result['sig'] = result['pval'] < 0.05
        trajectory.append(result)
    return trajectory


def cusum_changepoint(values, target=None):
    """CUSUM changepoint detection on numeric sequence.

    Detects last point where the sequence was still drifting away from target.
    Returns changepoint index (0-based) or None.
    """
    n = len(values)
    if n < 3:
        return None

    if target is None:
        target = values[-1]  # final value is the "stable" target

    deviations = [abs(v - target) for v in values]
    sd = max(1e-10, (sum(d * d for d in deviations) / n) ** 0.5)
    k_slack = 0.5 * sd
    h = 4 * sd

    # Forward CUSUM
    S = 0
    last_exceed = None
    for i in range(n):
        S = max(0, S + deviations[i] - k_slack)
        if S > h:
            last_exceed = i

    return last_exceed


def binary_segmentation_cp(sig_sequence):
    """Find the changepoint in a binary significance sequence.

    Returns index of the most likely change in Bernoulli parameter, or None.
    """
    n = len(sig_sequence)
    if n < 4:
        return None

    total_p = sum(sig_sequence) / n
    if total_p == 0 or total_p == 1:
        return None  # all same, no change

    best_llr = 0
    best_c = None

    for c in range(2, n - 1):
        n1 = c
        n2 = n - c
        p1 = sum(sig_sequence[:c]) / n1
        p2 = sum(sig_sequence[c:]) / n2

        # Avoid log(0)
        if p1 <= 0 or p1 >= 1 or p2 <= 0 or p2 >= 1:
            continue

        llr = (n1 * (p1 * math.log(p1 / total_p) + (1 - p1) * math.log((1 - p1) / (1 - total_p))) +
               n2 * (p2 * math.log(p2 / total_p) + (1 - p2) * math.log((1 - p2) / (1 - total_p))))

        if abs(llr) > best_llr:
            best_llr = abs(llr)
            best_c = c

    # Threshold: chi2(1, 0.05) = 3.84
    if best_llr > 3.84:
        return best_c
    return None


def pelt_changepoint(values):
    """Simplified PELT for single changepoint in I2/tau2 trajectory.

    Finds the split that maximally reduces total variance, penalized by BIC.
    Returns changepoint index or None.
    """
    n = len(values)
    if n < 4:
        return None

    total_var = sum((v - sum(values) / n) ** 2 for v in values) / n
    if total_var < 1e-10:
        return None  # constant sequence

    total_cost = n * math.log(max(total_var, 1e-10))
    penalty = 3 * math.log(n)

    best_improvement = 0
    best_c = None

    for c in range(2, n - 1):
        seg1 = values[:c]
        seg2 = values[c:]

        var1 = sum((v - sum(seg1) / len(seg1)) ** 2 for v in seg1) / len(seg1)
        var2 = sum((v - sum(seg2) / len(seg2)) ** 2 for v in seg2) / len(seg2)

        cost_split = len(seg1) * math.log(max(var1, 1e-10)) + len(seg2) * math.log(max(var2, 1e-10))
        improvement = total_cost - cost_split

        if improvement > best_improvement:
            best_improvement = improvement
            best_c = c

    if best_improvement > penalty:
        return best_c
    return None


def classify_stability(cp_pos, k):
    """Classify stability based on effect changepoint position."""
    if cp_pos is None or cp_pos > 0.9 * k:
        return 'Never Stabilized'
    ratio = cp_pos / k
    if ratio < 0.3:
        return 'Stable Early'
    if ratio < 0.6:
        return 'Stable Mid'
    return 'Stable Late'


def safe_float(val):
    if val is None or val == '':
        return None
    try:
        v = float(val)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def main():
    print("Building MetaShift cumulative trajectories...\n")

    # Load per-study data (10 analyses)
    json_path = Path(r'C:\FragilityAtlas\data\output\r_validation_inputs.json')
    if not json_path.exists():
        print(f"ERROR: {json_path} not found")
        return

    with open(json_path, encoding='utf-8') as f:
        study_data = json.load(f)
    print(f"  Loaded {len(study_data)} analyses with per-study data")

    # Compute cumulative MA + changepoints
    reviews_with_trajectory = []
    for analysis in study_data:
        rid = analysis.get('review_id', '')
        name = analysis.get('analysis_name', '')
        yi = analysis.get('yi', [])
        sei = analysis.get('sei', [])
        k = len(yi)

        if k < 2:
            continue

        # Cumulative meta-analysis
        traj = cumulative_ma(yi, sei)

        # Extract sequences for changepoint detection
        est_seq = [t['est'] for t in traj]
        sig_seq = [1 if t['sig'] else 0 for t in traj]
        i2_seq = [t['I2'] for t in traj]

        # Run changepoint algorithms
        effect_cp = cusum_changepoint(est_seq)
        sig_cp = binary_segmentation_cp(sig_seq)
        het_cp = pelt_changepoint(i2_seq)

        stability = classify_stability(effect_cp, k)

        reviews_with_trajectory.append({
            'review_id': rid,
            'analysis_name': name,
            'k': k,
            'studies': [{'idx': i + 1, 'yi': round(yi[i], 6), 'sei': round(sei[i], 6)} for i in range(k)],
            'cumulative': traj,
            'changepoints': {
                'effect': {'position': effect_cp, 'type': 'stabilized'} if effect_cp is not None else None,
                'significance': {'position': sig_cp, 'type': 'flipped'} if sig_cp is not None else None,
                'heterogeneity': {'position': het_cp, 'type': 'spike'} if het_cp is not None else None,
            },
            'stability_class': stability,
        })

    # Load aggregate data for remaining 393 reviews
    fa_path = Path(r'C:\FragilityAtlas\data\output\fragility_atlas_results.csv')
    pg_path = Path(r'C:\PredictionGap\data\output\prediction_gap_results.csv')

    fa_map = {}
    with open(fa_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            fa_map[row['review_id']] = row

    pg_map = {}
    with open(pg_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            pg_map[row['review_id']] = row

    traj_ids = set(r['review_id'] for r in reviews_with_trajectory)
    reviews_aggregate = []
    for rid in sorted(fa_map.keys()):
        if rid in traj_ids:
            continue
        fa = fa_map[rid]
        pg = pg_map.get(rid, {})
        reviews_aggregate.append({
            'review_id': rid,
            'analysis_name': fa.get('analysis_name', ''),
            'k': int(float(fa.get('k', 0))),
            'theta': safe_float(pg.get('theta')),
            'p_value': safe_float(pg.get('p_value')),
            'I2': safe_float(pg.get('I2')),
            'tau2': safe_float(pg.get('tau2')),
            'has_trajectory': False,
        })

    # Output
    output = {
        'reviews_with_trajectory': reviews_with_trajectory,
        'reviews_aggregate_only': reviews_aggregate,
        'corpus_summary': {
            'n_with_trajectory': len(reviews_with_trajectory),
            'n_aggregate_only': len(reviews_aggregate),
            'n_total': len(reviews_with_trajectory) + len(reviews_aggregate),
        }
    }

    out_path = Path(r'C:\MetaShift\data\cumulative.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f)

    # Summary
    print(f"\n{'='*50}")
    print("METASHIFT CUMULATIVE TRAJECTORIES")
    print(f"{'='*50}")
    print(f"  Reviews with trajectory: {len(reviews_with_trajectory)}")
    print(f"  Reviews aggregate only:  {len(reviews_aggregate)}")
    for r in reviews_with_trajectory:
        cp = r['changepoints']
        e = f"step {cp['effect']['position']}" if cp['effect'] else 'none'
        s = f"step {cp['significance']['position']}" if cp['significance'] else 'none'
        h = f"step {cp['heterogeneity']['position']}" if cp['heterogeneity'] else 'none'
        print(f"  {r['review_id']} (k={r['k']}): effect CP={e}, sig flip={s}, het spike={h} -> {r['stability_class']}")
    print(f"\n  Output: {out_path}")
    print(f"  Size: {out_path.stat().st_size / 1024:.0f} KB")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run build script**

```bash
cd /c/MetaShift && python build_cumulative.py
```

Expected: 10 reviews with trajectories, ~393 aggregate-only, changepoint positions printed.

- [ ] **Step 3: Verify JSON**

```bash
python -c "import json; d=json.load(open('data/cumulative.json')); print(f'Trajectory: {d[\"corpus_summary\"][\"n_with_trajectory\"]}'); print(f'Aggregate: {d[\"corpus_summary\"][\"n_aggregate_only\"]}'); r=d['reviews_with_trajectory'][0]; print(f'{r[\"review_id\"]} k={r[\"k\"]} steps={len(r[\"cumulative\"])} stability={r[\"stability_class\"]}')"
```

- [ ] **Step 4: Commit**

```bash
cd /c/MetaShift && git add build_cumulative.py data/cumulative.json && git commit -m "feat: cumulative MA + changepoint detection for 10 reviews"
```

---

### Task 2: Dashboard scaffold + embedded data

**Files:**
- Create: `C:\MetaShift\metashift.html`

**Context:** Full HTML shell with CSS theming, mode toggle, section placeholders, embedded JSON. Same portfolio pattern.

- [ ] **Step 1: Create metashift.html**

CSS: light/dark theming (`ms_` localStorage prefix), `.stat-grid`, `.card`, stability class badges (`.cls-early` green, `.cls-mid` blue, `.cls-late` amber, `.cls-never` red), `.sortable-table`, `.detail-panel`, `.trajectory-chart` containers, print styles, responsive.

HTML: Header "MetaShift — When Did the Evidence Change?", dark mode toggle, mode toggle (Corpus/Custom), Sections 1-4 placeholders, Custom mode with study input form + "Load Example" button, footer with export buttons and caveat.

JS: Embed `data/cumulative.json` as `var DATA = <json>;`, `escapeHtml()`, `applyMode()`, dark mode toggle, stubs for render functions.

- [ ] **Step 2: Embed JSON data, verify no `</script>` issues**

- [ ] **Step 3: Verify — page loads, mode toggle works, no errors**

- [ ] **Step 4: Commit**

```bash
cd /c/MetaShift && git add metashift.html && git commit -m "feat: dashboard scaffold — HTML, CSS, dark mode, embedded data"
```

---

### Task 3: Statistics engine + summary + stability chart

**Files:**
- Modify: `C:\MetaShift\metashift.html` — JS section

**Context:** Port the changepoint algorithms to JS (for custom mode), implement summary cards and stability distribution chart.

- [ ] **Step 1: Implement JS changepoint algorithms**

Port from Python: `dlMeta(yi, sei)`, `cumulativeMeta(yi, sei)`, `cusumChangepoint(values)`, `binarySegmentationCP(sigSeq)`, `peltChangepoint(values)`, `classifyStability(cpPos, k)`, `normalCDF(x)`.

These are used by Custom mode — the Corpus mode uses pre-computed results from the embedded JSON.

- [ ] **Step 2: Implement `renderSummary()`**

4 stat boxes from the trajectory reviews:
1. Reviews with trajectories (n)
2. % with effect CP detected
3. % with significance flip detected
4. % with heterogeneity spike detected

- [ ] **Step 3: Implement `renderStabilityChart()`**

SVG horizontal bar chart: 4 bars (Early/Mid/Late/Never) colored by class. Count from the trajectory reviews.

Below: histogram of k values across all 403 reviews (to show context of how many studies each review has).

- [ ] **Step 4: Verify — summary shows counts, stability chart renders**

- [ ] **Step 5: Commit**

```bash
cd /c/MetaShift && git add metashift.html && git commit -m "feat: changepoint algorithms + summary cards + stability chart"
```

---

### Task 4: Triple-panel trajectory chart (accordion detail)

**Files:**
- Modify: `C:\MetaShift\metashift.html` — `renderTrajectoryChart(review)`, `toggleDetail(idx)`

**Context:** The core visualization. Three vertically stacked SVG charts sharing the same x-axis (study number 1..k). This renders inside the accordion when a table row is clicked.

- [ ] **Step 1: Implement `renderTrajectoryChart(review)`**

Returns HTML string containing 3 SVG charts stacked vertically:

**Top panel — Effect trajectory** (~250px tall):
- X-axis: study number 1..k
- Y-axis: pooled effect estimate
- Line chart of cumulative estimate
- Shaded CI band (light fill between ci_lo and ci_hi)
- Vertical dashed line at effect changepoint (if detected)
- Color: line blue before CP, green after; CP line red

**Middle panel — Significance timeline** (~60px tall):
- X-axis: same as top
- Colored rectangles for each step: green if sig, grey if not
- Vertical dashed red line at significance flip point

**Bottom panel — Heterogeneity trajectory** (~200px tall):
- X-axis: same as top
- Y-axis: I² (0-100%)
- Line chart of cumulative I²
- Vertical dashed line at het changepoint
- Color: line grey, CP line red

All share x-axis with study number labels at bottom of lowest panel.

SVG dimensions: ~700px wide, ~550px total height. Use shared `xScale(step)` function for alignment.

- [ ] **Step 2: Implement review table + accordion**

Sortable table of trajectory reviews. Columns: Review ID, Analysis, k, Effect CP, Sig Flip, Het Spike, Stability Class. Click to expand triple-panel chart.

For aggregate-only reviews (393): show in a separate collapsed section with just k, theta, p, I2 (no trajectory available).

- [ ] **Step 3: Verify — click a review row, triple-panel chart appears with CP markers**

- [ ] **Step 4: Commit**

```bash
cd /c/MetaShift && git add metashift.html && git commit -m "feat: triple-panel trajectory chart + review table with accordion"
```

---

### Task 5: Custom mode + export + polish

**Files:**
- Modify: `C:\MetaShift\metashift.html`

**Context:** Custom mode lets users enter study-level data. Export buttons. Final polish.

- [ ] **Step 1: Implement Custom mode**

Input form: table with rows [Study Name, Year, Effect, SE]. "Add Row" button, "Clear" button, "Load Example" button (pre-fills with a 10-study example showing a clear changepoint).

"Run Analysis" button: collects data, sorts by year, runs `cumulativeMeta()` + 3 changepoint algorithms, renders triple-panel chart + summary stats below the form.

- [ ] **Step 2: Implement "Load Example" dataset**

Built-in example: 10 studies showing a clear pattern — first 4 studies non-significant, study 5 tips to significance, studies 6-10 stable. Heterogeneity spikes at study 7.

```javascript
var EXAMPLE_STUDIES = [
  {name: 'Smith 2010', year: 2010, yi: 0.15, sei: 0.20},
  {name: 'Jones 2011', year: 2011, yi: 0.30, sei: 0.25},
  {name: 'Lee 2012', year: 2012, yi: 0.22, sei: 0.18},
  {name: 'Brown 2013', year: 2013, yi: 0.40, sei: 0.22},
  {name: 'Garcia 2014', year: 2014, yi: 0.55, sei: 0.15},
  {name: 'Wilson 2015', year: 2015, yi: 0.35, sei: 0.12},
  {name: 'Chen 2016', year: 2016, yi: 0.80, sei: 0.30},
  {name: 'Patel 2017', year: 2017, yi: 0.42, sei: 0.14},
  {name: 'Kim 2018', year: 2018, yi: 0.38, sei: 0.11},
  {name: 'Ahmed 2019', year: 2019, yi: 0.41, sei: 0.13},
];
```

- [ ] **Step 3: Implement export**

- CSV: export review screening table (trajectory reviews with CP data)
- SVG: download the triple-panel chart for the currently expanded review
- PNG: same via canvas rendering

Use csvSafe() for formula injection protection. Revoke Blob URLs.

- [ ] **Step 4: Full integration test**

Corpus mode:
1. Summary cards show trajectory count and CP percentages
2. Stability chart renders
3. Table shows trajectory reviews, accordion expands with triple-panel chart
4. CPs marked correctly on charts

Custom mode:
1. Click "Load Example" — form fills with 10 studies
2. Click "Run Analysis" — triple-panel chart renders, CPs detected
3. Clear and enter 3 studies manually — chart updates

Exports work, dark mode works, div balance, no `</script>`.

- [ ] **Step 5: Commit**

```bash
cd /c/MetaShift && git add metashift.html && git commit -m "feat: custom mode, export, example data, polish"
```
