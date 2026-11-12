// Ring organ + tiny transformer inference. Mirrors organ.py and models.py (checked by parity.cjs).
(function (root) {
  const C = 8, D = 48, H = 2, DH = 24, W = 12, K = 3, FEAT = 18;
  const LAPS = Array.from({ length: 52 }, (_, i) => 13 + i), TAU = 32, THETA = 12;

  // ---------- the organ: a comb of delay rings, laps 13..64 ----------
  class Organ {
    constructor(gated = true) {
      this.gated = gated; this.t = 0;
      this.R = LAPS.map(L => new Uint8Array(L * C));
      this.filled = LAPS.map(L => new Uint8Array(L));
      this.Z = new Float64Array(LAPS.length);
      this.xorNow = LAPS.map(() => new Uint8Array(C));
    }
    // x: Uint8Array(8). Returns {tokens: Float64Array(K*FEAT), laps: [K]}
    step(x) {
      const t = this.t, n = LAPS.length;
      for (let j = 0; j < n; j++) {
        const L = LAPS[j], s = t % L, R = this.R[j], f = this.filled[j][s];
        let cnt = 0;
        for (let c = 0; c < C; c++) {
          const v = f ? (R[s * C + c] ^ x[c]) : x[c];
          this.xorNow[j][c] = v; cnt += v;
        }
        this.Z[j] = this.Z[j] * (1 - 1 / TAU) + cnt;
        if (!this.gated || this.Z[j] > THETA || !f) {
          for (let c = 0; c < C; c++) R[s * C + c] = x[c];
          this.filled[j][s] = 1;
        }
      }
      const order = Array.from({ length: n }, (_, j) => j).sort((a, b) => this.Z[a] - this.Z[b] || a - b).slice(0, K);
      const tok = new Float64Array(K * FEAT), laps = [];
      order.forEach((j, i) => {
        const L = LAPS[j], s = (t + 1) % L, o = i * FEAT;
        for (let c = 0; c < C; c++) { tok[o + c] = this.R[j][s * C + c]; tok[o + C + c] = this.xorNow[j][c]; }
        tok[o + 2 * C] = this.Z[j] / TAU; tok[o + 2 * C + 1] = L / 64; laps.push(L);
      });
      this.t++;
      return { tokens: tok, laps, Z: this.Z };
    }
  }

  // ---------- tensors ----------
  function b64f32(s) {
    const bin = typeof atob === 'function' ? atob(s) : Buffer.from(s, 'base64').toString('binary');
    const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    return new Float32Array(u.buffer);
  }
  function linear(Wt, b, x, out, nOut, nIn) {   // Wt row-major (nOut, nIn)
    for (let o = 0; o < nOut; o++) { let s = b[o]; const r = o * nIn; for (let i = 0; i < nIn; i++) s += Wt[r + i] * x[i]; out[o] = s; }
    return out;
  }
  function layerNorm(x, g, b) {
    let m = 0; for (const v of x) m += v; m /= x.length;
    let v2 = 0; for (const v of x) v2 += (v - m) ** 2; v2 /= x.length;
    const inv = 1 / Math.sqrt(v2 + 1e-5), o = new Float64Array(x.length);
    for (let i = 0; i < x.length; i++) o[i] = (x[i] - m) * inv * g[i] + b[i];
    return o;
  }
  function erf(x) {   // Abramowitz-Stegun 7.1.26
    const s = Math.sign(x); x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }
  const gelu = x => 0.5 * x * (1 + erf(x / Math.SQRT2));
  const FREQ = Array.from({ length: DH / 2 }, (_, i) => Math.pow(10000, -2 * i / DH));
  function ropeHead(v, off, pos) {   // rotate v[off..off+DH) in place
    for (let i = 0; i < DH / 2; i++) {
      const a = v[off + 2 * i], b = v[off + 2 * i + 1], ang = pos * FREQ[i], c = Math.cos(ang), s = Math.sin(ang);
      v[off + 2 * i] = a * c - b * s; v[off + 2 * i + 1] = a * s + b * c;
    }
  }

  // ---------- the tiny transformer, run one tick at a time ----------
  class Tiny {
    constructor(weights, organ) {
      this.w = {}; for (const [k, v] of Object.entries(weights)) this.w[k] = b64f32(v);
      this.organ = organ; this.t = 0;
      this.hist = [[], []];     // per layer: inputs to that layer for the last W ticks
      this.share = [0, 0];
    }
    step(x, org) {   // x: 8 values 0/1; org: {tokens, laps} or null. Returns 8 probabilities for the next tick.
      const w = this.w, t = this.t;
      let z = linear(w['inp.weight'], w['inp.bias'], Float64Array.from(x), new Float64Array(D), D, C);
      let oe = null;
      if (this.organ) {
        oe = [];
        for (let k = 0; k < K; k++) oe.push(linear(w['org.weight'], w['org.bias'], org.tokens.subarray(k * FEAT, (k + 1) * FEAT), new Float64Array(D), D, FEAT));
      }
      for (let l = 0; l < 2; l++) {
        const p = `layers.${l}.`, hs = this.hist[l];
        hs.push({ z, t }); if (hs.length > W) hs.shift();
        const ys = hs.map(h => layerNorm(h.z, w[p + 'ln1.weight'], w[p + 'ln1.bias']));
        const yq = ys[ys.length - 1];
        const q = linear(w[p + 'q.weight'], w[p + 'q.bias'], yq, new Float64Array(D), D, D);
        const ks = ys.map((y, i) => linear(w[p + 'k.weight'], w[p + 'k.bias'], y, new Float64Array(D), D, D));
        const vs = ys.map(y => linear(w[p + 'v.weight'], w[p + 'v.bias'], y, new Float64Array(D), D, D));
        for (let hh = 0; hh < H; hh++) { ropeHead(q, hh * DH, t); ks.forEach((k, i) => ropeHead(k, hh * DH, hs[i].t)); }
        let kos = [], vos = [];
        if (this.organ) {
          kos = oe.map((e, k) => { const v = linear(w[p + 'ko.weight'], w[p + 'ko.bias'], e, new Float64Array(D), D, D); for (let hh = 0; hh < H; hh++) ropeHead(v, hh * DH, t + 1 - org.laps[k]); return v; });
          vos = oe.map(e => linear(w[p + 'vo.weight'], w[p + 'vo.bias'], e, new Float64Array(D), D, D));
        }
        const att = new Float64Array(D); let shareSum = 0;
        for (let hh = 0; hh < H; hh++) {
          const o = hh * DH, sc = [];
          for (const k of ks.concat(kos)) { let s = 0; for (let i = 0; i < DH; i++) s += q[o + i] * k[o + i]; sc.push(s / Math.sqrt(DH)); }
          const mx = Math.max(...sc), ex = sc.map(s => Math.exp(s - mx)), sum = ex.reduce((a, b) => a + b, 0);
          const vals = vs.concat(vos);
          ex.forEach((e, i) => { const a = e / sum; for (let j = 0; j < DH; j++) att[o + j] += a * vals[i][o + j]; if (i >= ks.length) shareSum += a; });
        }
        this.share[l] = shareSum / H;
        const ao = linear(w[p + 'o.weight'], w[p + 'o.bias'], att, new Float64Array(D), D, D);
        const z2 = z.map((v, i) => v + ao[i]);
        const m1 = linear(w[p + 'mlp.0.weight'], w[p + 'mlp.0.bias'], layerNorm(z2, w[p + 'ln2.weight'], w[p + 'ln2.bias']), new Float64Array(2 * D), 2 * D, D).map(gelu);
        const m2 = linear(w[p + 'mlp.2.weight'], w[p + 'mlp.2.bias'], m1, new Float64Array(D), D, 2 * D);
        z = z2.map((v, i) => v + m2[i]);
      }
      const logits = linear(w['out.weight'], w['out.bias'], layerNorm(z, w['ln.weight'], w['ln.bias']), new Float64Array(C), C, D);
      this.t++;
      return Array.from(logits, v => 1 / (1 + Math.exp(-v)));
    }
  }

  // ---------- demo songs (same recipe as organ.song) ----------
  function mulberry(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function makeSong(seed, opts = {}) {
    const r = mulberry(seed), ri = n => Math.floor(r() * n);
    const P = opts.period || 20 + ri(41), pDrop = opts.drop ?? 0.1, pIns = opts.insert ?? 0.01, pFill = opts.fill ?? 0.04;
    const pattern = () => { const p = Array.from({ length: P }, () => Uint8Array.from({ length: C }, (_, c) => c < 7 && r() < 0.12 ? 1 : 0)); p[ri(P)][ri(7)] = 1; return p; };
    let pat = pattern(), next = pattern(), changeAt = opts.changeEvery ? opts.changeEvery : 0;
    let fill = [];
    return {
      P, get pattern() { return pat; },
      next(t) {
        if (changeAt && t > 0 && t % changeAt === 0) { pat = next; next = pattern(); }
        const clean = Uint8Array.from(pat[t % P]);
        const x = Uint8Array.from(clean);
        for (let c = 0; c < 7; c++) { if (x[c] && r() < pDrop) x[c] = 0; if (r() < pIns) x[c] = 1; }
        fill = fill.filter(f => f > t);
        if (r() < pFill) fill.push(t, t + 2, t + 4);
        if (fill.includes(t) || fill.length && false) {}
        return { x, clean };
      },
      fillAt(t) { return fill; }
    };
  }

  root.RingOrgan = { Organ, Tiny, makeSong, mulberry, LAPS, C, K, W, TAU };
})(typeof window !== 'undefined' ? window : globalThis);
