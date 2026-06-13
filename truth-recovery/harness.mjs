// ============================================================
// harness.mjs -- Truth-recovery yardstick for metashift.
//
// metashift detects "hidden regime shifts" in CUMULATIVE meta-analyses with
// CUSUM / binary-segmentation / PELT. The honest test: inject a cumulative MA
// with a KNOWN changepoint (the true study effect shifts at a known position) or
// NO changepoint, and measure detection POWER, FALSE-POSITIVE rate, and
// localization, using the app's OWN detectors (engine.mjs, verbatim).
//
// Subtlety measured here: the app runs PELT/CUSUM on the CUMULATIVE estimate
// trajectory, which SMOOTHS a per-study step into a gradual ramp. We therefore
// also run PELT on the PER-STUDY effect sequence to see how much detection power
// the cumulative smoothing costs.
//
// Truth-first: every number printed comes from seeded simulation here.
// Run:  node truth-recovery/harness.mjs --reps 3000
// ============================================================

import { cumulativeMeta, peltChangepoint, binarySegmentationCP, cusumChangepoint } from './engine.mjs';

const BASE_SEED = 20260613;
function makeRng(seed) { let a = seed >>> 0; return function () { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function randn(rng) { let u1 = rng(), u2 = rng(); if (u1 < 1e-12) u1 = 1e-12; return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2); }

// k studies; shift=true -> true effect jumps by delta at position cTrue.
function gen(rng, k, cTrue, delta, mu0 = 0.0, seLo = 0.1, seHi = 0.35) {
  const yi = [], sei = [];
  for (let t = 0; t < k; t++) {
    const se = Math.exp(Math.log(seLo) + (Math.log(seHi) - Math.log(seLo)) * rng());
    const mean = mu0 + (t >= cTrue ? delta : 0);
    yi.push(mean + se * randn(rng));
    sei.push(se);
  }
  return { yi, sei };
}

export function runCell(k, cTrue, delta, reps, rng) {
  let detCum = 0, detStudy = 0, detSig = 0, fpCum = 0, fpStudy = 0, fpSig = 0;
  let locCumSum = 0, locCumN = 0, locStudySum = 0, locStudyN = 0;
  const tol = 2;   // localization within +/-2 of the true changepoint
  for (let r = 0; r < reps; r++) {
    // --- shift present ---
    const s = gen(rng, k, cTrue, delta);
    const traj = cumulativeMeta(s.yi, s.sei);
    const cumEst = traj.map(x => x.est);
    const sigSeq = traj.map(x => (x.sig ? 1 : 0));
    const cpCum = peltChangepoint(cumEst);
    const cpStudy = peltChangepoint(s.yi);
    const cpSig = binarySegmentationCP(sigSeq);
    if (cpCum != null) { detCum++; locCumN++; if (Math.abs(cpCum - cTrue) <= tol) locCumSum++; }
    if (cpStudy != null) { detStudy++; locStudyN++; if (Math.abs(cpStudy - cTrue) <= tol) locStudySum++; }
    if (cpSig != null) detSig++;
    // --- no shift (null) ---
    const n = gen(rng, k, k + 1, 0);   // cTrue beyond k -> never shifts
    const trajN = cumulativeMeta(n.yi, n.sei);
    if (peltChangepoint(trajN.map(x => x.est)) != null) fpCum++;
    if (peltChangepoint(n.yi) != null) fpStudy++;
    if (binarySegmentationCP(trajN.map(x => (x.sig ? 1 : 0))) != null) fpSig++;
  }
  return {
    powerCum: +(detCum / reps).toFixed(3), powerStudy: +(detStudy / reps).toFixed(3), powerSig: +(detSig / reps).toFixed(3),
    fpCum: +(fpCum / reps).toFixed(3), fpStudy: +(fpStudy / reps).toFixed(3), fpSig: +(fpSig / reps).toFixed(3),
    locCum: locCumN ? +(locCumSum / locCumN).toFixed(3) : null, locStudy: locStudyN ? +(locStudySum / locStudyN).toFixed(3) : null,
  };
}

export function runGrid({ reps = 3000 } = {}) {
  const rng = makeRng(BASE_SEED);
  const cells = [[12, 6, 0.5], [12, 6, 1.0], [20, 10, 0.5], [20, 10, 1.0]];
  return cells.map(([k, c, d]) => ({ k, c, d, results: runCell(k, c, d, reps, rng) }));
}

const isMain = process.argv[1]?.endsWith('harness.mjs');
if (isMain) {
  const i = process.argv.indexOf('--reps');
  const reps = i >= 0 ? Number(process.argv[i + 1]) : 3000;
  const t0 = Date.now();
  const grid = runGrid({ reps });
  console.log(`\n# Truth-recovery yardstick -- metashift (changepoint detection)`);
  console.log(`reps=${reps}/cell  seed=${BASE_SEED}\n`);
  console.log('## PELT power (true shift detected) / false-positive (no shift) / localization\n');
  console.log('k  cTrue delta | PELT-cum: pow / FP / loc | PELT-study: pow / FP / loc | binSeg-sig: pow / FP');
  for (const g of grid) {
    const r = g.results;
    console.log(`${String(g.k).padStart(2)} ${String(g.c).padStart(4)} ${String(g.d).padStart(5)} | ` +
      `${String(r.powerCum).padStart(5)} / ${String(r.fpCum).padStart(5)} / ${String(r.locCum).padStart(5)} | ` +
      `${String(r.powerStudy).padStart(5)} / ${String(r.fpStudy).padStart(5)} / ${String(r.locStudy).padStart(5)} | ` +
      `${String(r.powerSig).padStart(5)} / ${String(r.fpSig).padStart(5)}`);
  }
  console.log(`\n(loc = fraction of detections within +/-2 of the true changepoint. ${((Date.now() - t0) / 1000).toFixed(1)}s)`);
}
