"""Build cumulative meta-analysis trajectories + changepoint detection.

For 10 reviews with per-study data: full cumulative DL + 3 changepoint algorithms.
For 393 reviews: aggregate data only (no trajectory).

Output: data/cumulative.json
"""

import json
import math
import csv
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def normal_cdf(x):
    """Standard normal CDF (Abramowitz & Stegun 26.2.17)."""
    if x < -8:
        return 0.0
    if x > 8:
        return 1.0
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989422802 * math.exp(-x * x / 2)
    p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
    return 1.0 - p if x > 0 else p


def dl_meta(yi, sei):
    """DerSimonian-Laird random-effects meta-analysis."""
    k = len(yi)
    if k == 0:
        return None

    # Guard for non-finite values
    if not all(math.isfinite(y) and math.isfinite(s) and s > 0 for y, s in zip(yi, sei)):
        return None

    if k == 1:
        z = abs(yi[0] / sei[0]) if sei[0] > 0 else 0.0
        pval = 2 * (1 - normal_cdf(z))
        return {
            'est': yi[0],
            'se': sei[0],
            'ci_lo': yi[0] - 1.96 * sei[0],
            'ci_hi': yi[0] + 1.96 * sei[0],
            'pval': pval,
            'tau2': 0.0,
            'I2': 0.0,
            'Q': 0.0,
        }

    vi = [s * s for s in sei]
    wi = [1.0 / v for v in vi]
    sum_w = sum(wi)
    mu_fe = sum(w * y for w, y in zip(wi, yi)) / sum_w

    Q = sum(w * (y - mu_fe) ** 2 for w, y in zip(wi, yi))
    sum_w2 = sum(w * w for w in wi)
    C = sum_w - sum_w2 / sum_w
    tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0

    wi_re = [1.0 / (v + tau2) for v in vi]
    sum_w_re = sum(wi_re)
    mu_re = sum(w * y for w, y in zip(wi_re, yi)) / sum_w_re
    se_re = math.sqrt(1.0 / sum_w_re)

    ci_lo = mu_re - 1.96 * se_re
    ci_hi = mu_re + 1.96 * se_re
    z = abs(mu_re / se_re) if se_re > 0 else 0.0
    pval = 2 * (1 - normal_cdf(z))
    I2 = max(0.0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0.0

    return {
        'est': round(mu_re, 6),
        'se': round(se_re, 6),
        'ci_lo': round(ci_lo, 6),
        'ci_hi': round(ci_hi, 6),
        'pval': round(pval, 8),
        'tau2': round(tau2, 6),
        'I2': round(I2, 1),
        'Q': round(Q, 4),
    }


def cumulative_ma(yi, sei):
    """Compute cumulative DL meta-analysis: pool studies 1..t for t=1..k."""
    trajectory = []
    for t in range(1, len(yi) + 1):
        result = dl_meta(yi[:t], sei[:t])
        if result is None:
            continue
        result['step'] = t
        result['sig'] = result['pval'] < 0.05
        trajectory.append(result)
    return trajectory


def cusum_changepoint(values, target=None):
    """CUSUM changepoint detection on numeric sequence.

    Detects last point where the sequence was still drifting away from target.
    Returns changepoint index (1-based, matching step numbers) or None.
    k < 3: returns None gracefully.
    """
    n = len(values)
    if n < 3:
        return None

    if target is None:
        target = values[-1]  # final value is the "stable" target

    deviations = [abs(v - target) for v in values]
    mean_dev = sum(deviations) / n
    sd = max(1e-10, (sum((d - mean_dev) ** 2 for d in deviations) / n) ** 0.5)
    k_slack = 0.5 * sd
    h = 4 * sd

    # Forward CUSUM
    S = 0.0
    last_exceed = None
    for i in range(n):
        S = max(0.0, S + deviations[i] - k_slack)
        if S > h:
            last_exceed = i + 1  # 1-based step number

    return last_exceed


def binary_segmentation_cp(sig_sequence):
    """Find the changepoint in a binary significance sequence.

    Returns 1-based step index of the most likely change in Bernoulli parameter, or None.
    k < 4: returns None gracefully.
    """
    n = len(sig_sequence)
    if n < 4:
        return None

    total_p = sum(sig_sequence) / n
    if total_p == 0 or total_p == 1:
        return None  # all same, no change

    best_llr = 0.0
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
    if best_llr > 3.84 and best_c is not None:
        return best_c + 1  # convert to 1-based step number
    return None


def pelt_changepoint(values):
    """Simplified PELT for single changepoint in I2/tau2 trajectory.

    Finds the split that maximally reduces total variance, penalized by BIC.
    Returns 1-based step index or None.
    k < 4: returns None gracefully.
    """
    n = len(values)
    if n < 4:
        return None

    mean_all = sum(values) / n
    total_var = sum((v - mean_all) ** 2 for v in values) / n
    if total_var < 1e-10:
        return None  # constant sequence

    total_cost = n * math.log(max(total_var, 1e-10))
    penalty = 3 * math.log(n)

    best_improvement = 0.0
    best_c = None

    for c in range(2, n - 1):
        seg1 = values[:c]
        seg2 = values[c:]

        mean1 = sum(seg1) / len(seg1)
        mean2 = sum(seg2) / len(seg2)
        var1 = sum((v - mean1) ** 2 for v in seg1) / len(seg1)
        var2 = sum((v - mean2) ** 2 for v in seg2) / len(seg2)

        cost_split = (len(seg1) * math.log(max(var1, 1e-10)) +
                      len(seg2) * math.log(max(var2, 1e-10)))
        improvement = total_cost - cost_split

        if improvement > best_improvement:
            best_improvement = improvement
            best_c = c

    if best_improvement > penalty and best_c is not None:
        return best_c + 1  # convert to 1-based step number
    return None


def classify_stability(cp_pos, k):
    """Classify stability based on effect changepoint position (1-based)."""
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
            print(f"  Skipping {rid} (k={k} < 2)")
            continue

        # Cumulative meta-analysis
        traj = cumulative_ma(yi, sei)
        if len(traj) < 2:
            print(f"  Skipping {rid}: trajectory has < 2 valid steps")
            continue

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
            'studies': [
                {
                    'idx': i + 1,
                    'yi': round(yi[i], 6),
                    'sei': round(sei[i], 6)
                }
                for i in range(k)
            ],
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
    if fa_path.exists():
        with open(fa_path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                fa_map[row['review_id']] = row
        print(f"  Loaded {len(fa_map)} rows from fragility_atlas_results.csv")
    else:
        print(f"  WARNING: {fa_path} not found — aggregate data will be empty")

    pg_map = {}
    if pg_path.exists():
        with open(pg_path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                pg_map[row['review_id']] = row
        print(f"  Loaded {len(pg_map)} rows from prediction_gap_results.csv")
    else:
        print(f"  WARNING: {pg_path} not found — aggregate fields will be null")

    traj_ids = set(r['review_id'] for r in reviews_with_trajectory)
    reviews_aggregate = []
    for rid in sorted(fa_map.keys()):
        if rid in traj_ids:
            continue
        fa = fa_map[rid]
        pg = pg_map.get(rid, {})
        k_val = fa.get('k', '0')
        try:
            k_int = int(float(k_val)) if k_val else 0
        except (ValueError, TypeError):
            k_int = 0
        reviews_aggregate.append({
            'review_id': rid,
            'analysis_name': fa.get('analysis_name', ''),
            'k': k_int,
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
    print(f"\n{'='*55}")
    print("METASHIFT CUMULATIVE TRAJECTORIES")
    print(f"{'='*55}")
    print(f"  Reviews with trajectory: {len(reviews_with_trajectory)}")
    print(f"  Reviews aggregate only:  {len(reviews_aggregate)}")
    print(f"  Total:                   {output['corpus_summary']['n_total']}")
    print()
    for r in reviews_with_trajectory:
        cp = r['changepoints']
        e = f"step {cp['effect']['position']}" if cp['effect'] else 'none'
        s = f"step {cp['significance']['position']}" if cp['significance'] else 'none'
        h = f"step {cp['heterogeneity']['position']}" if cp['heterogeneity'] else 'none'
        print(f"  {r['review_id']} (k={r['k']}, steps={len(r['cumulative'])}): "
              f"effect CP={e}, sig flip={s}, het spike={h} -> {r['stability_class']}")
    size_kb = out_path.stat().st_size / 1024
    print(f"\n  Output: {out_path}")
    print(f"  Size:   {size_kb:.0f} KB")


if __name__ == '__main__':
    main()
