"""Pack the trained 12-tick models (weights/*.json from train.py) as base64 float32 for the page,
and write the fixtures tests/parity.cjs checks the browser engine against."""
import base64, json
import numpy as np, torch
import organ as O
from models import TinyT

out = {}
for tag, use_org in (("organ", True), ("plain", False)):
    sd = {k: torch.tensor(v) for k, v in json.load(open(f"weights/{tag}.json")).items()}
    out[tag] = {k: base64.b64encode(v.numpy().astype(np.float32).tobytes()).decode() for k, v in sd.items()}
    m = TinyT(W=12, organ=use_org); m.load_state_dict(sd); m.eval()
    rng = np.random.default_rng(99)
    x, cl, P, ch = O.song(rng, T=200)
    tok, lap, _ = O.organ(x)
    with torch.no_grad():
        p = torch.sigmoid(m(torch.tensor(x[None], dtype=torch.float32), torch.tensor(tok[None]), torch.tensor(lap[None], dtype=torch.long)))[0].numpy()
    json.dump({"x": x.tolist(), "p": p.round(6).tolist(), "laps": lap.tolist()}, open(f"tests/parity_{tag}.json", "w"))
json.dump(out, open("web/weights.json", "w"))
print({k: len(json.dumps(v)) for k, v in out.items()})
