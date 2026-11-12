"""Train and score every condition. Writes results/results.json and weights/*.json for the page."""
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import organ as O
from models import TinyT, GRU, nparams

torch.set_num_threads(2)
T, NTR, NTE = 256, int(sys.argv[1]) if len(sys.argv) > 1 else 4000, 800
STEPS, BS = int(sys.argv[2]) if len(sys.argv) > 2 else 2500, 32
Path("results").mkdir(exist_ok=True); Path("weights").mkdir(exist_ok=True)


def make(n, seed, variants):
    rng = np.random.default_rng(seed)
    X, CL, P, CH = [], [], [], []
    org = {v: ([], []) for v in variants}
    for _ in range(n):
        x, cl, p, ch = O.song(rng, T=T)
        X.append(x); CL.append(cl); P.append(p); CH.append(ch)
        for v in variants:
            tok, lap, _ = O.organ(x, gated=(v != "plain"), scramble_rng=(np.random.default_rng(int(rng.integers(1 << 30))) if v == "scrambled" else None))
            org[v][0].append(tok); org[v][1].append(lap)
    t = lambda a, dt=torch.float32: torch.tensor(np.array(a), dtype=dt)
    return t(X), t(CL), np.array(P), np.array(CH), {v: (t(a), t(b, torch.long)) for v, (a, b) in org.items()}


def masks(P, CH):
    """Positions t (predicting t+1) scored as steady (after 2 periods, outside the change) and
    recovery (the 2 periods after a change)."""
    N = len(P)
    steady = np.zeros((N, T - 1), bool); rec = np.zeros((N, T - 1), bool)
    for n in range(N):
        for t in range(T - 1):
            u = t + 1
            if CH[n] >= 0 and CH[n] <= u < CH[n] + 2 * P[n]:
                rec[n, t] = True
            elif u >= 2 * P[n]:
                steady[n, t] = True
    return torch.tensor(steady), torch.tensor(rec)


def f1(pred, tgt, m, ch=slice(None)):
    p, y = pred[..., ch][m], tgt[..., ch][m]
    tp = (p * y).sum().item()
    return 2 * tp / max(1, p.sum().item() + y.sum().item())


VARIANTS = ["gated", "plain", "scrambled"]
t0 = time.time()
Xtr, CLtr, Ptr, CHtr, ORtr = make(NTR, 1, VARIANTS)
Xte, CLte, Pte, CHte, ORte = make(NTE, 2, VARIANTS)
steady, rec = masks(Pte, CHte)
print(f"data {time.time() - t0:.0f}s", flush=True)

ONLY = sys.argv[3].split(",") if len(sys.argv) > 3 else None
CONDS = {
    "transformer, window 12": dict(model=lambda: TinyT(W=12), org=None),
    "transformer, window 12 + ring organ (gated)": dict(model=lambda: TinyT(W=12, organ=True), org="gated"),
    "transformer, window 12 + ring organ (plain copy)": dict(model=lambda: TinyT(W=12, organ=True), org="plain"),
    "transformer, window 12 + organ with random laps": dict(model=lambda: TinyT(W=12, organ=True), org="scrambled"),
    "transformer, window 64": dict(model=lambda: TinyT(W=64), org=None),
    "GRU, hidden 96": dict(model=lambda: GRU(96), org=None),
}

res = {"setup": dict(T=T, train=NTR, test=NTE, steps=STEPS, batch=BS, periods="20..60", drop=0.1, insert=0.01,
                     laps=f"{O.LAPS[0]}..{O.LAPS[-1]}", K=O.K, tau=O.TAU, theta=O.THETA), "models": {}}

# no-learning references
for v in ("gated", "plain"):
    pred = ORte[v][0][:, :-1, 0, :8]
    res["models"][f"organ alone ({v}): best ring's echo, no model"] = {
        "steady_F1": f1(pred, Xte[:, 1:], steady), "steady_F1_vs_clean": f1(pred, CLte[:, 1:], steady),
        "rhythm_bands_F1_vs_clean": f1(pred, CLte[:, 1:], steady, slice(0, 7)),
        "fill_band_F1_vs_clean": f1(pred, CLte[:, 1:], steady, slice(7, 8)),
        "recovery_F1": f1(pred, Xte[:, 1:], rec), "params": 0}
oracle = torch.zeros_like(Xte[:, 1:])
for n in range(NTE):
    for t in range(T - 1):
        if t + 1 - Pte[n] >= 0: oracle[n, t] = CLte[n, t + 1 - Pte[n]] if (CHte[n] < 0 or not (CHte[n] <= t + 1 < CHte[n] + Pte[n])) else Xte[n, t + 1 - Pte[n]]
res["models"]["reference: knows the true period, copies the clean last lap"] = {
    "steady_F1": f1(oracle, Xte[:, 1:], steady), "steady_F1_vs_clean": f1(oracle, CLte[:, 1:], steady),
    "rhythm_bands_F1_vs_clean": f1(oracle, CLte[:, 1:], steady, slice(0, 7)), "params": 0}

for name, c in CONDS.items():
    if ONLY and not any(o in name for o in ONLY): continue
    torch.manual_seed(0)
    m = c["model"]()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=STEPS, pct_start=0.1)
    tt = time.time()
    g = torch.Generator().manual_seed(0)
    for step in range(STEPS):
        idx = torch.randint(0, NTR, (BS,), generator=g)
        o, lp = (ORtr[c["org"]][0][idx], ORtr[c["org"]][1][idx]) if c["org"] else (None, None)
        logit = m(Xtr[idx], o, lp)[:, :-1]
        loss = F.binary_cross_entropy_with_logits(logit, Xtr[idx][:, 1:])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 500 == 0: print(f"  {name[:50]:50s} step {step} loss {loss.item():.4f}", flush=True)
    m.eval()
    with torch.no_grad():
        preds = []
        for b in range(0, NTE, 100):
            o, lp = (ORte[c["org"]][0][b:b + 100], ORte[c["org"]][1][b:b + 100]) if c["org"] else (None, None)
            preds.append((torch.sigmoid(m(Xte[b:b + 100], o, lp)[:, :-1]) > 0.5).float())
        pred = torch.cat(preds)
        share = None
        if c["org"]:
            m(Xte[:100], ORte[c["org"]][0][:100], ORte[c["org"]][1][:100])
            share = [round(l.last_org_share, 3) for l in m.layers]
    r = {"steady_F1": f1(pred, Xte[:, 1:], steady), "steady_F1_vs_clean": f1(pred, CLte[:, 1:], steady),
         "rhythm_bands_F1_vs_clean": f1(pred, CLte[:, 1:], steady, slice(0, 7)),
         "fill_band_F1_vs_clean": f1(pred, CLte[:, 1:], steady, slice(7, 8)),
         "recovery_F1": f1(pred, Xte[:, 1:], rec), "params": nparams(m), "train_seconds": round(time.time() - tt),
         "final_loss": round(loss.item(), 4), "attention_share_on_organ_tokens_per_layer": share}
    res["models"][name] = r
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    if isinstance(m, TinyT) and m.W == 12:
        tag = "organ" if c["org"] == "gated" else ("plain" if c["org"] is None else None)
        if tag:
            json.dump({k: v.tolist() for k, v in m.state_dict().items()}, open(f"weights/{tag}.json", "w"))

    json.dump(res, open(os.environ.get("OUT", "results/main.json"), "w"), indent=1)
json.dump(res, open(os.environ.get("OUT", "results/main.json"), "w"), indent=1)
print(json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in res["models"].items()}, indent=1))
