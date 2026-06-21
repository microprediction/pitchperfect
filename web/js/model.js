// SoccerNetV6 value network -- JavaScript port (typed-array hot path).
//
// A faithful translation of pitchperfect/value_net.py (numpy), itself a port of
// the trained PyTorch model in the `value` repo. All three agree to ~1e-6 (see
// tests/test_parity.py). Vectors are Float64Array; the math is identical to the
// numpy reference, so JS vs numpy agree to ~1e-15.
//
// Inputs are sim coordinates: x in [-50, 50], y in [-30, 30]; blue/team0 attacks
// +x. value() returns V in [-1, +1]; +1 favors blue.

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
  for (let i = 0; i < v.length; i++) {
    const x = v[i];
    v[i] = 0.5 * x * (1.0 + erf(x * INV_SQRT2));
  }
  return v;
}

// y = x @ W^T + b. W flat (out*in), x Float64Array(in). Returns Float64Array(out).
function linear(x, Wd, bd, out, inn) {
  const y = new Float64Array(out);
  for (let o = 0; o < out; o++) {
    let s = bd[o];
    const base = o * inn;
    for (let i = 0; i < inn; i++) s += x[i] * Wd[base + i];
    y[o] = s;
  }
  return y;
}

function layernormInPlace(x, w, b, eps) {
  const n = x.length;
  let mu = 0; for (let i = 0; i < n; i++) mu += x[i]; mu /= n;
  let v = 0; for (let i = 0; i < n; i++) { const d = x[i] - mu; v += d * d; } v /= n;
  const inv = 1.0 / Math.sqrt(v + eps);
  for (let i = 0; i < n; i++) x[i] = (x[i] - mu) * inv * w[i] + b[i];
  return x;
}

export class ValueNet {
  constructor(weights) {
    this.cfg = weights.config;
    this.t = {};
    for (const [k, v] of Object.entries(weights.tensors)) {
      this.t[k] = { shape: v.shape, data: Float64Array.from(v.data) };
    }
    this.hidden = this.cfg.hidden;
    this.H = this.cfg.n_heads;
    this.eps = this.cfg.layernorm_eps;
    this.sentinel = this.cfg.sentinel;
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

  // Self-attention over `rows` (array of Float64Array(D)). pad: bool[S] keys.
  mha(rows, prefix, pad) {
    const S = rows.length, D = this.hidden, H = this.H, hd = D / H;
    const Wi = this.t[prefix + ".in_proj_weight"].data;
    const bi = this.t[prefix + ".in_proj_bias"].data;
    const q = new Float64Array(S * D), k = new Float64Array(S * D), v = new Float64Array(S * D);
    for (let s = 0; s < S; s++) {
      const xs = rows[s], so = s * D;
      for (let o = 0; o < D; o++) {
        let sq = bi[o], sk = bi[D + o], sv = bi[2 * D + o];
        const bq = o * D, bk = (D + o) * D, bv = (2 * D + o) * D;
        for (let i = 0; i < D; i++) { const xv = xs[i]; sq += xv * Wi[bq + i]; sk += xv * Wi[bk + i]; sv += xv * Wi[bv + i]; }
        q[so + o] = sq; k[so + o] = sk; v[so + o] = sv;
      }
    }
    const scale = 1.0 / Math.sqrt(hd);
    const ctx = new Float64Array(S * D);
    const scores = new Float64Array(S);
    for (let h = 0; h < H; h++) {
      const off = h * hd;
      for (let i = 0; i < S; i++) {
        const qi = i * D + off;
        let mx = -Infinity;
        for (let j = 0; j < S; j++) {
          if (pad && pad[j]) { scores[j] = -Infinity; continue; }
          const kj = j * D + off;
          let sc = 0;
          for (let c = 0; c < hd; c++) sc += q[qi + c] * k[kj + c];
          sc *= scale; scores[j] = sc; if (sc > mx) mx = sc;
        }
        let sum = 0;
        for (let j = 0; j < S; j++) { const e = scores[j] === -Infinity ? 0 : Math.exp(scores[j] - mx); scores[j] = e; sum += e; }
        const ci = i * D + off;
        for (let j = 0; j < S; j++) {
          const a = scores[j] / sum;
          if (a === 0) continue;
          const vj = j * D + off;
          for (let c = 0; c < hd; c++) ctx[ci + c] += a * v[vj + c];
        }
      }
    }
    const Wo = this.t[prefix + ".out_proj.weight"], bo = this.t[prefix + ".out_proj.bias"];
    const out = new Array(S);
    for (let s = 0; s < S; s++) out[s] = linear(ctx.subarray(s * D, s * D + D), Wo.data, bo.data, D, D);
    return out;
  }

  // 8 relational features per player. players/opponents: array of Float64Array(4).
  spatial(players, opponents, ballX, ballY, pmask, omask, ownGoalX, oppGoalX) {
    const N = players.length, M = opponents.length, R = 0.3;
    const feats = new Array(N);
    let nValidOpp = 0; for (let m = 0; m < M; m++) if (!omask[m]) nValidOpp++;
    nValidOpp = Math.max(nValidOpp, 1);
    for (let i = 0; i < N; i++) {
      const f = new Float64Array(8);
      if (pmask[i]) { feats[i] = f; continue; }
      const pi = players[i], px = pi[0], py = pi[1], vx = pi[2], vy = pi[3];
      f[0] = Math.hypot(px - ballX, py - ballY);
      f[1] = Math.hypot(px - ownGoalX, py);
      f[2] = Math.hypot(px - oppGoalX, py);
      const tox = oppGoalX - px, toy = -py;
      f[3] = tox / (Math.hypot(tox, toy) + 1e-6);
      let nearestOpp = 1e6, density = 0;
      for (let m = 0; m < M; m++) {
        if (omask[m]) continue;
        const om = opponents[m];
        const dd = Math.hypot(px - om[0], py - om[1]);
        if (dd < nearestOpp) nearestOpp = dd;
        if (dd < R) density += 1;
      }
      f[4] = nearestOpp; f[6] = density / nValidOpp;
      let nearestTeam = 1e6;
      for (let j = 0; j < N; j++) {
        if (j === i || pmask[j]) continue;
        const pj = players[j];
        const dd = Math.hypot(px - pj[0], py - pj[1]);
        if (dd < nearestTeam) nearestTeam = dd;
      }
      f[5] = nearestTeam;
      const tbx = ballX - px, tby = ballY - py, tn = Math.hypot(tbx, tby) + 1e-6;
      f[7] = vx * (tbx / tn) + vy * (tby / tn);
      feats[i] = f;
    }
    return feats;
  }

  // ball: Float64Array(4); blue/red: array of Float64Array(4) -- normalized.
  forward(ball, blue, red) {
    const ballX = ball[0], ballY = ball[1];
    const bmask = blue.map((p) => p[0] === this.sentinel && p[1] === this.sentinel);
    const rmask = red.map((p) => p[0] === this.sentinel && p[1] === this.sentinel);

    const blueExtra = this.spatial(blue, red, ballX, ballY, bmask, rmask, -1.0, 1.0);
    const redExtra = this.spatial(red, blue, ballX, ballY, rmask, bmask, 1.0, -1.0);

    const D = this.hidden;
    const enc = (state, extra) => state.map((p, i) => {
      const inp = new Float64Array(12);
      inp.set(p); inp.set(extra[i], 4);
      return this.mlp(inp, "player_encoder");
    });
    const ballEmb = this.mlp(ball, "ball_encoder");
    const blueEnc = enc(blue, blueExtra);
    const redEnc = enc(red, redExtra);

    const cross = this.crossAttn(ballEmb, blueEnc, redEnc, bmask, rmask);
    const bluePool = this.maskedPool(cross.blue, bmask, "blue_pool");
    const redPool = this.maskedPool(cross.red, rmask, "red_pool");
    const merged = new Float64Array(3 * D);
    merged.set(cross.ball, 0); merged.set(bluePool, D); merged.set(redPool, 2 * D);
    return this.mlp(merged, "head_outcome")[0];
  }

  crossAttn(ballEmb, blueEnc, redEnc, bmask, rmask) {
    const N = blueEnc.length, D = this.hidden;
    const te = this.t["cross_attn.team_embed.weight"].data;
    const tokens = new Array(1 + 2 * N);
    const make = (vec, team, masked) => {
      const t = new Float64Array(D), to = team * D;
      if (!masked) for (let c = 0; c < D; c++) t[c] = vec[c] + te[to + c];
      else for (let c = 0; c < D; c++) t[c] = te[to + c];
      return t;
    };
    tokens[0] = make(ballEmb, 0, false);
    for (let i = 0; i < N; i++) tokens[1 + i] = make(blueEnc[i], 1, bmask[i]);
    for (let i = 0; i < N; i++) tokens[1 + N + i] = make(redEnc[i], 2, rmask[i]);
    const pad = [false]; for (let i = 0; i < N; i++) pad.push(bmask[i]); for (let i = 0; i < N; i++) pad.push(rmask[i]);
    const attn = this.mha(tokens, "cross_attn.attn", pad);
    const w = this.t["cross_attn.norm.weight"].data, b = this.t["cross_attn.norm.bias"].data;
    const normed = tokens.map((tok, i) => {
      const r = new Float64Array(D);
      for (let c = 0; c < D; c++) r[c] = tok[c] + attn[i][c];
      return layernormInPlace(r, w, b, this.eps);
    });
    return { ball: normed[0], blue: normed.slice(1, 1 + N), red: normed.slice(1 + N) };
  }

  maskedPool(rows, mask, prefix) {
    const D = this.hidden;
    const xz = rows.map((row, i) => {
      if (!mask[i]) return row;
      return new Float64Array(D);
    });
    const attn = this.mha(xz, prefix + ".attn", mask);
    const w = this.t[prefix + ".norm.weight"].data, b = this.t[prefix + ".norm.bias"].data;
    const pooled = new Float64Array(D);
    let nValid = 0;
    for (let i = 0; i < xz.length; i++) {
      if (mask[i]) continue;
      const r = new Float64Array(D);
      for (let c = 0; c < D; c++) r[c] = xz[i][c] + attn[i][c];
      layernormInPlace(r, w, b, this.eps);
      nValid++;
      for (let c = 0; c < D; c++) pooled[c] += r[c];
    }
    nValid = Math.max(nValid, 1);
    for (let c = 0; c < D; c++) pooled[c] /= nValid;
    return pooled;
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

  // Signed value in [-1, +1]; +1 favors blue (the +x attacker).
  value(ball, blue, red) {
    return 2.0 * sigmoid(this.logit(ball, blue, red)) - 1.0;
  }
}
