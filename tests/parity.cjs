// The browser engine (web/engine.js) against PyTorch: organ laps identical, probabilities within 1e-3.
const fs = require('fs'), path = require('path'); globalThis.window = globalThis;
eval(fs.readFileSync(path.join(__dirname, '..', 'web', 'engine.js'), 'utf8'));
const W = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'web', 'weights.json')));
let worst = 0, lapsOk = true;
for (const tag of ['organ', 'plain']) {
  const fx = JSON.parse(fs.readFileSync(path.join(__dirname, `parity_${tag}.json`)));
  const org = new RingOrgan.Organ(true), m = new RingOrgan.Tiny(W[tag], tag === 'organ');
  fx.x.forEach((x, t) => {
    const o = org.step(Uint8Array.from(x));
    if (o.laps.join() !== fx.laps[t].join()) lapsOk = false;
    m.step(x, o).forEach((v, c) => { worst = Math.max(worst, Math.abs(v - fx.p[t][c])); });
  });
}
console.log('organ laps match:', lapsOk, ' max probability difference:', worst.toExponential(2));
if (!lapsOk || worst > 1e-3) { console.error('FAIL'); process.exit(1); }
console.log('PASS');
