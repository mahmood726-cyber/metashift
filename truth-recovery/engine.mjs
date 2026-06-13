// engine.mjs -- pure changepoint core EXTRACTED VERBATIM from metashift.html
// (571-682): normalCDF, dlMeta, cumulativeMeta, cusum/binarySeg/PELT detectors.

function normalCDF(x) {
  // Abramowitz & Stegun 26.2.17
  if (x < -8) return 0;
  if (x > 8) return 1;
  var t = 1.0 / (1.0 + 0.2316419 * Math.abs(x));
  var d = 0.3989422802 * Math.exp(-x * x / 2);
  var p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return (x > 0) ? (1 - p) : p;
}

function dlMeta(yi, sei) {
  var k = yi.length;
  if (k === 0) return null;
  if (k === 1) {
    var z0 = sei[0] > 0 ? Math.abs(yi[0] / sei[0]) : 0;
    return {
      est: yi[0], se: sei[0],
      ci_lo: yi[0] - 1.96 * sei[0], ci_hi: yi[0] + 1.96 * sei[0],
      pval: 2 * (1 - normalCDF(z0)),
      tau2: 0, I2: 0
    };
  }
  var vi = sei.map(function(s) { return s * s; });
  var wi = vi.map(function(v) { return 1.0 / v; });
  var sw = wi.reduce(function(a, b) { return a + b; }, 0);
  var mu_fe = wi.reduce(function(a, w, i) { return a + w * yi[i]; }, 0) / sw;
  var Q = wi.reduce(function(a, w, i) { return a + w * Math.pow(yi[i] - mu_fe, 2); }, 0);
  var sw2 = wi.reduce(function(a, w) { return a + w * w; }, 0);
  var C = sw - sw2 / sw;
  var tau2 = C > 0 ? Math.max(0, (Q - (k - 1)) / C) : 0;
  var wi_re = vi.map(function(v) { return 1.0 / (v + tau2); });
  var swr = wi_re.reduce(function(a, b) { return a + b; }, 0);
  var mu_re = wi_re.reduce(function(a, w, i) { return a + w * yi[i]; }, 0) / swr;
  var se_re = Math.sqrt(1.0 / swr);
  var ci_lo = mu_re - 1.96 * se_re;
  var ci_hi = mu_re + 1.96 * se_re;
  var z = se_re > 0 ? Math.abs(mu_re / se_re) : 0;
  var pval = 2 * (1 - normalCDF(z));
  var I2 = Q > 0 ? Math.max(0, (Q - (k - 1)) / Q) * 100 : 0;
  return { est: mu_re, se: se_re, ci_lo: ci_lo, ci_hi: ci_hi, pval: pval, tau2: tau2, I2: I2 };
}

function cumulativeMeta(yi, sei) {
  var traj = [];
  for (var t = 1; t <= yi.length; t++) {
    var r = dlMeta(yi.slice(0, t), sei.slice(0, t));
    r.step = t;
    r.sig = r.pval < 0.05;
    traj.push(r);
  }
  return traj;
}

function cusumChangepoint(values) {
  var n = values.length;
  if (n < 3) return null;
  var target = values[n - 1];
  var deviations = values.map(function(v) { return Math.abs(v - target); });
  var mean_d = deviations.reduce(function(a, b) { return a + b; }, 0) / n;
  var var_d = deviations.reduce(function(a, d) { return a + Math.pow(d - mean_d, 2); }, 0) / n;
  var sd = Math.sqrt(var_d) || 1e-10;
  var k_slack = 0.5 * sd;
  var h = 4 * sd;
  var S = 0;
  var last_exceed = null;
  for (var i = 0; i < n; i++) {
    S = Math.max(0, S + deviations[i] - k_slack);
    if (S > h) last_exceed = i;
  }
  return last_exceed;
}

function binarySegmentationCP(sigSeq) {
  var n = sigSeq.length;
  if (n < 4) return null;
  var total_p = sigSeq.reduce(function(a, b) { return a + b; }, 0) / n;
  if (total_p === 0 || total_p === 1) return null;
  var best_llr = 0;
  var best_c = null;
  for (var c = 2; c < n - 1; c++) {
    var n1 = c, n2 = n - c;
    var p1 = sigSeq.slice(0, c).reduce(function(a, b) { return a + b; }, 0) / n1;
    var p2 = sigSeq.slice(c).reduce(function(a, b) { return a + b; }, 0) / n2;
    if (p1 <= 0 || p1 >= 1 || p2 <= 0 || p2 >= 1) continue;
    var llr = n1 * (p1 * Math.log(p1 / total_p) + (1 - p1) * Math.log((1 - p1) / (1 - total_p))) +
              n2 * (p2 * Math.log(p2 / total_p) + (1 - p2) * Math.log((1 - p2) / (1 - total_p)));
    if (Math.abs(llr) > best_llr) { best_llr = Math.abs(llr); best_c = c; }
  }
  return best_llr > 3.84 ? best_c : null;
}

function peltChangepoint(values) {
  var n = values.length;
  if (n < 4) return null;
  var mean = values.reduce(function(a, b) { return a + b; }, 0) / n;
  var total_var = values.reduce(function(a, v) { return a + Math.pow(v - mean, 2); }, 0) / n;
  if (total_var < 1e-10) return null;
  var total_cost = n * Math.log(Math.max(total_var, 1e-10));
  var penalty = 3 * Math.log(n);
  var best_imp = 0, best_c = null;
  for (var c = 2; c < n - 1; c++) {
    var s1 = values.slice(0, c), s2 = values.slice(c);
    var m1 = s1.reduce(function(a, b) { return a + b; }, 0) / s1.length;
    var m2 = s2.reduce(function(a, b) { return a + b; }, 0) / s2.length;
    var v1 = s1.reduce(function(a, v) { return a + Math.pow(v - m1, 2); }, 0) / s1.length;
    var v2 = s2.reduce(function(a, v) { return a + Math.pow(v - m2, 2); }, 0) / s2.length;
    var cost_split = s1.length * Math.log(Math.max(v1, 1e-10)) + s2.length * Math.log(Math.max(v2, 1e-10));
    var imp = total_cost - cost_split;
    if (imp > best_imp) { best_imp = imp; best_c = c; }
  }
  return best_imp > penalty ? best_c : null;
}

export { dlMeta, cumulativeMeta, cusumChangepoint, binarySegmentationCP, peltChangepoint };
