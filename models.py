"""Tiny models. Same sizes as the page runs.

TinyT: 2 layers, 2 heads, d = 48, causal attention over the last W ticks, RoPE.
       With an organ, every query also attends to that tick's K organ tokens. Organ tokens are
       embedded once and read by every layer (they are memory, not updated by the model).
       An organ token for lap L at query t is rotated as position t + 1 - L, the time its echo
       comes from.
GRU:   one GRU layer, hidden 96, the conventional recurrent memory.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

C, FEAT = 8, 18


def rope(x, pos):
    """x (..., dh) with dh even; pos broadcastable to x[..., 0]."""
    dh = x.shape[-1]
    i = torch.arange(dh // 2, device=x.device, dtype=x.dtype)
    freq = 10000 ** (-2 * i / dh)
    ang = pos[..., None] * freq
    c, s = torch.cos(ang), torch.sin(ang)
    a, b = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = a * c - b * s
    out[..., 1::2] = a * s + b * c
    return out


class Layer(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.dh = h, d // h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.ko, self.vo = nn.Linear(d, d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x, mask, org=None, org_pos=None):
        B, T, d = x.shape
        h, dh = self.h, self.dh
        y = self.ln1(x)
        pos = torch.arange(T, device=x.device, dtype=x.dtype)
        q = rope(self.q(y).view(B, T, h, dh).transpose(1, 2), pos)
        k = rope(self.k(y).view(B, T, h, dh).transpose(1, 2), pos)
        v = self.v(y).view(B, T, h, dh).transpose(1, 2)
        s = q @ k.transpose(-1, -2) / math.sqrt(dh)
        s = s.masked_fill(~mask, float("-inf"))
        if org is not None:
            Kn = org.shape[2]
            ko = self.ko(org).view(B, T, Kn, h, dh).permute(0, 3, 1, 2, 4)      # B h T K dh
            ko = rope(ko, org_pos[:, None])                                   # org_pos B T K
            vo = self.vo(org).view(B, T, Kn, h, dh).permute(0, 3, 1, 2, 4)
            so = (q[:, :, :, None, :] * ko).sum(-1) / math.sqrt(dh)            # B h T K
            a = torch.softmax(torch.cat([s, so], -1), -1)
            out = a[..., :T] @ v + (a[..., T:, None] * vo).sum(-2)
            self.last_org_share = a[..., T:].sum(-1).mean().item()
        else:
            out = torch.softmax(s, -1) @ v
        x = x + self.o(out.transpose(1, 2).reshape(B, T, d))
        return x + self.mlp(self.ln2(x))


class TinyT(nn.Module):
    def __init__(self, W=12, organ=False, d=48, h=2, L=2):
        super().__init__()
        self.W, self.use_organ = W, organ
        self.inp = nn.Linear(C, d)
        self.org = nn.Linear(FEAT, d) if organ else None
        self.layers = nn.ModuleList(Layer(d, h) for _ in range(L))
        self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, C)

    def forward(self, x, org=None, org_lap=None):
        B, T, _ = x.shape
        i = torch.arange(T, device=x.device)
        mask = (i[None, :] <= i[:, None]) & (i[None, :] > i[:, None] - self.W)
        z = self.inp(x)
        oe = op = None
        if self.use_organ:
            oe = self.org(org)
            op = (i[None, :, None] + 1 - org_lap).to(x.dtype)
        for l in self.layers:
            z = l(z, mask, oe, op)
        return self.out(self.ln(z))


class GRU(nn.Module):
    def __init__(self, hdim=96):
        super().__init__()
        self.g = nn.GRU(C, hdim, batch_first=True)
        self.out = nn.Linear(hdim, C)

    def forward(self, x, org=None, org_lap=None):
        return self.out(self.g(x)[0])


def nparams(m):
    return sum(p.numel() for p in m.parameters())
