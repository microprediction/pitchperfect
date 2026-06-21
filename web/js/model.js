// MettleNet value network -- JavaScript port (typed-array hot path).
//
// A faithful translation of pitchperfect/value_net.py (numpy), itself a port of
// the trained PyTorch model in the `value` repo. All three agree to ~1e-6 (see
// tests/test_parity.py).
//
// Architecture: per-player MLP encoder -> sum-pool each team; ball MLP encoder;
// concat[ball, blue_sum, red_sum] -> outcome head -> logit; V = 2σ(logit)-1.
//
// value() returns the ANTISYMMETRIZED value: V(s) = (raw(s) - raw(swap(s)))/2,
// where swap flips the pitch in x and swaps the teams. This guarantees
// V(s) = -V(swap(s)) exactly and removes any red/blue bias; a mirror-symmetric
// position reads 0. Inputs are sim coordinates (x in [-50,50], y in [-30,30]);
// blue attacks +x. +1 favors blue.

function erf(x) {
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741,
        a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const s = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1.0 / (1.0 + p * ax);
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax);
  return s * y;
}
const INV_SQRT2 = 1.0 / Math.sqrt(2.0);
function sigmoid(x) { return 1.0 / (1.0 + Math.exp(-x)); }

function geluInPlace(v) {
  for (let i = 0; i < v.length; i++) { const x = v[i]; v[i] = 0.5 * x * (1.0 + erf(x * INV_SQRT2)); }
  return v;
}

// y = x @ W^T + b. W flat (out*in), x Float64Array(in). Returns Float64Array(out).
function linear(x, Wd, bd, out, inn) {
  const y = new Float64Array(out);
  for (let o = 0; o < out; o++) {
    let s = bd[o]; const base = o * inn;
    for (let i = 0; i < inn; i++) s += x[i] * Wd[base + i];
    y[o] = s;
  }
  return y;
}

export class ValueNet {
  constructor(weights) {
    this.cfg = weights.config;
    this.t = {};
    for (const [k, v] of Object.entries(weights.tensors)) {
      this.t[k] = { shape: v.shape, data: Float64Array.from(v.data) };
    }
    this.hidden = this.cfg.hidden;
    const n = this.cfg.norm;
    this.L = n.half_length; this.W = n.half_width; this.velScale = n.vel_scale;
  }

  static async fromURL(url) {
    const r = await fetch(url);
    return new ValueNet(await r.json());
  }

  lin(x, name) {
    const W = this.t[name + ".weight"], b = this.t[name + ".bias"];
    return linear(x, W.data, b.data, W.shape[0], W.shape[1]);
  }

  mlp(x, prefix) {
    let h = geluInPlace(this.lin(x, prefix + ".0"));
    h = geluInPlace(this.lin(h, prefix + ".2"));
    return this.lin(h, prefix + ".4");
  }

  // ball: Float64Array(4); blue/red: array of Float64Array(4) -- normalized.
  forward(ball, blue, red) {
    const D = this.hidden;
    const ballEmb = this.mlp(ball, "ball_encoder");
    const blueSum = new Float64Array(D), redSum = new Float64Array(D);
    for (const p of blue) { const e = this.mlp(p, "player_encoder"); for (let c = 0; c < D; c++) blueSum[c] += e[c]; }
    for (const p of red) { const e = this.mlp(p, "player_encoder"); for (let c = 0; c < D; c++) redSum[c] += e[c]; }
    const merged = new Float64Array(3 * D);
    merged.set(ballEmb, 0); merged.set(blueSum, D); merged.set(redSum, 2 * D);
    return this.mlp(merged, "head_outcome")[0];
  }

  // --- public API (sim coordinates) -----------------------------------------
  norm(p) {
    const o = new Float64Array(4);
    o[0] = p[0] / this.L; o[1] = p[1] / this.W;
    o[2] = (p[2] || 0) / this.velScale; o[3] = (p[3] || 0) / this.velScale;
    return o;
  }

  logit(ball, blue, red) {
    return this.forward(this.norm(ball), blue.map((p) => this.norm(p)), red.map((p) => this.norm(p)));
  }

  valueRaw(ball, blue, red) {
    return 2.0 * sigmoid(this.logit(ball, blue, red)) - 1.0;
  }

  // swap(s): flip x (positions + x-velocity) and swap the two teams.
  static swap(ball, blue, red) {
    const fx = (p) => [-p[0], p[1], -(p[2] || 0), p[3] || 0];
    return { ball: fx(ball), blue: red.map(fx), red: blue.map(fx) };
  }

  // Antisymmetrized value in [-1, +1]; +1 favors blue. Mirror-symmetric -> 0.
  value(ball, blue, red) {
    const v = this.valueRaw(ball, blue, red);
    const s = ValueNet.swap(ball, blue, red);
    const vs = this.valueRaw(s.ball, s.blue, s.red);
    return 0.5 * (v - vs);
  }
}
