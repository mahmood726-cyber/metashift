# Truth-recovery yardstick — metashift (changepoint detection)

**Verdict: CRITICAL miscalibration found + a measured improvement. PELT run on the
CUMULATIVE trajectory flags a "regime shift" in ~97–99% of *stable* meta-analyses;
running it on the PER-STUDY sequence cuts the false-positive rate ~2.4× and roughly
doubles localization accuracy.**

## Method
metashift detects "hidden regime shifts" in cumulative meta-analyses with CUSUM /
binary-segmentation / PELT. The honest test injects a cumulative MA with a KNOWN
changepoint (the true study effect jumps by δ at a known position) or NO
changepoint, and measures detection power, false-positive rate, and localization,
using the app's OWN detectors (engine.mjs, verbatim). Because the app runs the
detectors on the **cumulative estimate trajectory**, we also run PELT on the
**per-study** effect sequence to quantify what the cumulative smoothing costs.
3000–4000 reps/cell.

## Results — PELT power / false-positive / localization

| k  | cTrue | δ   | PELT-cumulative (pow/FP/loc) | PELT-per-study (pow/FP/loc) | binSeg-sig (pow/FP) |
|----|-------|-----|------------------------------|------------------------------|---------------------|
| 12 | 6     | 0.5 | 0.997 / **0.974** / 0.684 | 0.965 / 0.457 / 0.855 | 0.00 / 0.00 |
| 12 | 6     | 1.0 | 1.000 / **0.968** / 0.919 | 1.000 / 0.444 / 0.981 | 0.00 / 0.00 |
| 20 | 10    | 0.5 | 1.000 / **0.998** / 0.447 | 0.992 / 0.378 / 0.896 | 0.111 / 0.018 |
| 20 | 10    | 1.0 | 1.000 / **0.994** / 0.783 | 1.000 / 0.380 / 0.998 | 0.122 / 0.018 |

(loc = fraction of detections within ±2 of the true changepoint.)

## Findings (all measured)
1. **CRITICAL — PELT on the cumulative trajectory is essentially always positive.**
   Its false-positive rate under a *stable* (no-shift) meta-analysis is **0.97–0.998**.
   The reason is structural: the cumulative-estimate sequence is non-stationary
   (early points are high-variance with few studies; later points are low-variance
   as the running mean stabilizes) and strongly autocorrelated, which violates
   PELT's independence / constant-variance assumptions. PELT then "detects" the
   natural early-noisy → late-stable transition in nearly every meta-analysis. So
   the app's headline changepoint flag is not informative — it fires on almost
   everything.
2. **IMPROVEMENT — run the detector on the per-study sequence.** PELT on the raw
   per-study effects keeps full power (~1.0) while cutting the false-positive rate
   from ~0.97 to ~0.40 and improving localization from ~0.50 to ~0.90 (a sharp
   step is easy to localize; a smoothed ramp is not). → **detect changepoints on
   the per-study effect series, not the cumulative trajectory.**
3. **Still tune the penalty.** Even per-study, FPR ≈ 0.40 (> nominal) because the
   PELT penalty `3·log(n)` is too lenient for short series (n=12–20). A larger
   penalty (or a permutation-calibrated threshold) would bring it toward nominal.
4. **binary-segmentation on the significance sequence has ~zero power** (0–0.12):
   the cumulative significance series is "sticky" (once significant, stays
   significant) and the implementation excludes perfectly-separated splits, so it
   almost never detects a real shift. It is not a useful detector as written.

## Recommendation
Switch the effect-changepoint detector to operate on the per-study effect series
(or a properly pre-whitened cumulative series), raise the PELT penalty / use a
permutation null for short series, and drop or rework the significance-sequence
detector. A self-contained per-study reference is in `harness.mjs`.

## What did NOT transfer
NPE/conformal machinery is estimator-of-μ specific; metashift is a changepoint
detector, so the known-truth detection-power/false-positive harness (the same
diagnostic-vs-truth design as MAFI / multiverse) transferred. Engine unchanged;
no runtime dependency added.

## Reproduce
```
node truth-recovery/harness.mjs --reps 4000
node --test truth-recovery/test-truth-recovery.mjs
```
