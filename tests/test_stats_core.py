"""Direct unit + reference-value tests for the statistical / changepoint core.

Covers dl_meta (including an independently re-derived reference value and the
high-risk guard branches) and the three changepoint detectors + classifier.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_cumulative import (
    binary_segmentation_cp,
    classify_stability,
    cumulative_ma,
    cusum_changepoint,
    dl_meta,
    pelt_changepoint,
)


def _reference_dl(yi, sei):
    """Independent from-scratch DerSimonian-Laird implementation for cross-check.

    Deliberately written separately from build_cumulative.dl_meta so a formula
    regression in the module is not mirrored here.
    """
    k = len(yi)
    vi = [s * s for s in sei]
    wi = [1.0 / v for v in vi]
    sw = sum(wi)
    mu_fe = sum(w * y for w, y in zip(wi, yi)) / sw
    Q = sum(w * (y - mu_fe) ** 2 for w, y in zip(wi, yi))
    C = sw - sum(w * w for w in wi) / sw
    tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0
    wre = [1.0 / (v + tau2) for v in vi]
    swre = sum(wre)
    mu = sum(w * y for w, y in zip(wre, yi)) / swre
    se = math.sqrt(1.0 / swre)
    I2 = max(0.0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0.0
    return {"est": mu, "se": se, "tau2": tau2, "Q": Q, "I2": I2}


def test_dl_meta_exact_equal_sei_anchor():
    # Hand-derivable: equal sei=0.1 -> vi=0.01, wi=100 each, mu_fe=mean=0.15,
    # Q = 100*(0.05^2) + 0 + 100*(0.05^2) = 0.5 < k-1=2 -> tau2=0, I2=0,
    # se = sqrt(1/300) = 0.0577350.
    res = dl_meta([0.1, 0.15, 0.2], [0.1, 0.1, 0.1])
    assert res["est"] == 0.15
    assert res["Q"] == 0.5
    assert res["tau2"] == 0.0
    assert res["I2"] == 0.0
    assert abs(res["se"] - math.sqrt(1.0 / 300.0)) < 1e-6


def test_dl_meta_matches_independent_reference_with_heterogeneity():
    yi = [0.2, 0.5, 0.9]
    sei = [0.1, 0.2, 0.15]
    res = dl_meta(yi, sei)
    ref = _reference_dl(yi, sei)
    assert abs(res["est"] - ref["est"]) < 1e-6
    assert abs(res["se"] - ref["se"]) < 1e-6
    assert abs(res["tau2"] - ref["tau2"]) < 1e-6
    assert abs(res["Q"] - ref["Q"]) < 1e-4
    assert abs(res["I2"] - round(ref["I2"], 1)) < 0.1
    assert res["tau2"] > 0  # this dataset is genuinely heterogeneous


def test_dl_meta_k1_branch_returns_single_study():
    res = dl_meta([0.3], [0.2])
    assert res["est"] == 0.3
    assert res["se"] == 0.2
    assert res["tau2"] == 0.0
    assert res["I2"] == 0.0


def test_dl_meta_none_for_zero_sei():
    assert dl_meta([0.1, 0.2], [0.0, 0.1]) is None


def test_dl_meta_none_for_nan():
    assert dl_meta([0.1, float("nan")], [0.1, 0.1]) is None


def test_dl_meta_none_for_empty():
    assert dl_meta([], []) is None


def test_cumulative_ma_produces_one_step_per_study():
    traj = cumulative_ma([0.1, 0.15, 0.2], [0.1, 0.1, 0.1])
    assert [t["step"] for t in traj] == [1, 2, 3]
    assert traj[0]["est"] == 0.1  # first step is just study 1


def test_cusum_below_minimum_and_constant():
    assert cusum_changepoint([0.1, 0.2]) is None  # n < 3
    assert cusum_changepoint([0.5, 0.5, 0.5, 0.5]) is None  # constant


def test_binseg_below_minimum_and_all_same():
    assert binary_segmentation_cp([0, 1, 0]) is None  # n < 4
    assert binary_segmentation_cp([1, 1, 1, 1]) is None  # no change


def test_pelt_below_minimum_and_constant():
    assert pelt_changepoint([0.1, 0.2, 0.3]) is None  # n < 4
    assert pelt_changepoint([0.5, 0.5, 0.5, 0.5]) is None  # constant


def test_classify_stability_boundaries():
    k = 10
    assert classify_stability(None, k) == "Never Stabilized"
    assert classify_stability(10, k) == "Never Stabilized"  # > 0.9*k
    assert classify_stability(2, k) == "Stable Early"  # ratio 0.2 < 0.3
    assert classify_stability(3, k) == "Stable Mid"    # ratio 0.3, not < 0.3
    assert classify_stability(5, k) == "Stable Mid"    # ratio 0.5 < 0.6
    assert classify_stability(6, k) == "Stable Late"   # ratio 0.6, not < 0.6
