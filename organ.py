"""The ring organ and the data. numpy, so the page can copy it exactly.

Stream: 8 binary channels per tick (think 8 frequency bands).
Organ: a comb of delay rings, one per lap L = 13..64. Each tick, for every ring:
    comparator   m_L(t) = |x_t XOR ring_L[t mod L]|            bits that differ from one lap ago
    slow state   Z_L(t) = Z_L(t-1) (1 - 1/TAU) + m_L(t)          leaky surprise
    write        plain: ring_L[t mod L] = x_t   (copy, = toggle by the error)
                 gated: only if Z_L(t) > THETA                   (write when surprise persists)
    echo         e_L(t) = ring_L[(t+1) mod L]                    what this ring expects next tick
The organ hands attention K tokens: the K rings with the lowest Z (the best-tuned laps), each
[echo(8), comparator now(8), Z/TAU, L/64]. Selection by lowest Z is fixed, not learned
(a comb-filter tempo detector).
"""
import numpy as np

C = 8
LAPS = np.arange(13, 65)
TAU, THETA = 32.0, 12.0
K = 3
FEAT = 2 * C + 2


def organ(x, gated=True, laps=LAPS, k=K, scramble_rng=None):
    """x: (T, C) 0/1. Returns tokens (T, k, FEAT), chosen laps (T, k), Z (T, n_laps)."""
    T = x.shape[0]
    n = len(laps)
    R = np.zeros((n, int(laps.max()), C), np.uint8)
    filled = np.zeros((n, int(laps.max())), bool)
    Z = np.zeros(n)
    ar = np.arange(n)
    toks = np.zeros((T, k, FEAT), np.float32)
    chosen = np.zeros((T, k), np.int32)
    Zs = np.zeros((T, n), np.float32)
    for t in range(T):
        xt = x[t].astype(np.uint8)
        s = t % laps
        old, f = R[ar, s], filled[ar, s]
        xor = np.where(f[:, None], old ^ xt, xt)
        Z = Z * (1 - 1 / TAU) + xor.sum(1)
        w = np.ones(n, bool) if not gated else (Z > THETA) | ~f
        R[ar[w], s[w]] = xt
        filled[ar[w], s[w]] = True
        Zs[t] = Z
        order = np.argsort(Z, kind="stable")[:k] if scramble_rng is None else scramble_rng.choice(n, k, replace=False)
        nxt = (t + 1) % laps[order]
        toks[t, :, :C] = R[order, nxt]
        toks[t, :, C:2 * C] = xor[order]
        toks[t, :, 2 * C] = Z[order] / TAU
        toks[t, :, 2 * C + 1] = laps[order] / 64.0
        chosen[t] = laps[order]
    return toks, chosen, Zs


def song(rng, T=256, p_drop=0.1, p_insert=0.01, change=True, p_fill=0.04):
    """Bands 0..6: a rhythm with period P in 20..60 (always longer than a 12-tick window), with
    dropped beats (p_drop), stray beats (p_insert) and one change of pattern (same period).
    Band 7: 'fills', not periodic: a fill starts at random (p_fill per tick) and beats at t, t+2,
    t+4. The first beat of a fill cannot be predicted; the next two can, from the last few ticks.
    Returns observed x (T, C), clean expectation (T, C), period, change tick (or -1)."""
    P = int(rng.integers(20, 61))
    def pattern():
        pat = (rng.random((P, C)) < 0.12).astype(np.uint8)
        pat[:, 7] = 0
        pat[rng.integers(P), rng.integers(7)] = 1
        return pat
    a, b = pattern(), pattern()
    ch = int(rng.integers(T // 2, T - 2 * P)) if change and T - 2 * P > T // 2 else -1
    clean = np.array([(a if (ch < 0 or t < ch) else b)[t % P] for t in range(T)], np.uint8)
    x = clean.copy()
    x[(rng.random(x.shape) < p_drop) & (x == 1)] = 0
    ins = rng.random(x.shape) < p_insert
    ins[:, 7] = False
    x[ins] = 1
    for t in range(T):
        if rng.random() < p_fill:
            for d in (0, 2, 4):
                if t + d < T:
                    x[t + d, 7] = 1
                    if d: clean[t + d, 7] = 1          # continuations are expected; the start is not
    return x, clean, P, ch
