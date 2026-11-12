"""Build index.html: web/page.html + web/engine.js + web/weights.json + merged results."""
import json

r = json.load(open("results/main.json"))
for k, v in json.load(open("results/long.json"))["models"].items():   # baselines retrained 4x longer
    if "window 64" in k or "GRU" in k:
        r["models"][k + " (10,000 steps)"] = v
p = open("web/page.html").read()
p = p.replace("/*ENGINE*/", open("web/engine.js").read())
p = p.replace("/*WEIGHTS*/", open("web/weights.json").read())
p = p.replace("/*RESULTS*/", json.dumps(r))
open("index.html", "w").write(open("web/skeleton_head.html").read() + p + "</body></html>")
print("index.html", len(p))
