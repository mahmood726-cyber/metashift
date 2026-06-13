// node --test truth-recovery/test-truth-recovery.mjs
// Measured invariants for the metashift changepoint yardstick. Seeded; no
// hand-entered numbers.
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { peltChangepoint } from './engine.mjs';
import { runCell } from './harness.mjs';

function makeRng(seed) { let a = seed >>> 0; return function () { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

describe('engine', () => {
  it('PELT localizes a sharp step (and is eager: it splits even faintly-noisy flat data)', () => {
    assert.equal(peltChangepoint([0.2, 0.21, 0.19, 0.2, 0.9, 0.91, 0.89, 0.9]), 4);   // sharp step -> index 4
    assert.equal(peltChangepoint([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]), null);    // exactly constant -> guard
    // faint noise already triggers a split -> previews the false-positive finding
    assert.notEqual(peltChangepoint([0.2, 0.21, 0.19, 0.2, 0.2, 0.21, 0.2, 0.2]), null);
  });
});

describe('Truth-recovery (measured)', () => {
  it('CRITICAL: PELT on the CUMULATIVE trajectory has a catastrophic false-positive rate', () => {
    const r = runCell(20, 10, 1.0, 3000, makeRng(20260613));
    // a stable cumulative MA is flagged as having a "regime shift" almost always.
    assert.ok(r.fpCum > 0.9, `PELT-cumulative false-positive rate ${r.fpCum} (expected ~catastrophic)`);
  });

  it('IMPROVEMENT: running PELT on the PER-STUDY sequence cuts the false-positive rate and sharpens localization', () => {
    const r = runCell(20, 10, 1.0, 3000, makeRng(20260613));
    assert.ok(r.fpStudy < r.fpCum - 0.4, `per-study FP ${r.fpStudy} not much below cumulative ${r.fpCum}`);
    assert.ok(r.locStudy > r.locCum, `per-study localization ${r.locStudy} not better than cumulative ${r.locCum}`);
    assert.ok(r.powerStudy > 0.9, `per-study power ${r.powerStudy} too low`);
  });

  it('binary-segmentation on the significance sequence has near-zero power (not a useful detector here)', () => {
    const r = runCell(20, 10, 1.0, 3000, makeRng(20260613));
    assert.ok(r.powerSig < 0.25, `binSeg-significance power ${r.powerSig} unexpectedly high`);
  });
});
