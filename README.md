# pitchperfect ⚽

**A soccer value function you can touch.**

`pitchperfect` takes a trained neural-network *value function* for football — V(s),
an estimate of which team scores next — and makes it interactive. Drag 22 players
around a pitch in your browser and watch the model reason about danger in real time,
or call the same function from a dependency-light numpy package.

- **Interactive demo:** [`web/demo.html`](web/demo.html) — drag players/ball, watch
  V(s) update live, read off shape *probes* (compactness, width, block area, line
  height), and pick a player to see their **marginal value surface** sweep across the
  pitch.
- **Python package:** a numpy reimplementation of the network — `pip install`, no torch.
- **Provable fidelity:** the *same* network runs three ways — PyTorch (training), numpy
  (this package), and JavaScript (the demo) — verified identical to ~1e-6 by a parity
  test.

The World Cup is on, so here's something to play with at half-time.

## Install

```bash
pip install pitchperfect
```

```python
from pitchperfect import ValueNet

net = ValueNet.load()                 # loads bundled weights.json

# Sim coordinates: x in [-50, 50], y in [-30, 30]. Blue/team0 attacks +x.
v = net.value(
    ball=[0, 0],
    blue=[[-46, 0], [-30, -18], [-30, -6], [-30, 6], [-30, 18],
          [-8, -20], [-8, -7], [-8, 7], [-8, 20], [15, -8], [15, 8]],
    red=[[46, 0], [30, 18], [30, 6], [30, -6], [30, -18],
         [8, 20], [8, 7], [8, -7], [8, -20], [-15, 8], [-15, -8]],
)
print(v)   # signed value in [-1, +1]; +1 favors blue
```

## The model

The network is `SoccerNetV6` from the [`value`](https://github.com/score-technologies/value)
repo (the StatsBomb-trained checkpoint):

- **Deep Sets** per-player encoder — permutation invariant within a team.
- **Cross-team attention** — the ball and all 22 players attend to each other.
- **Masked pooling** per team, then a value head: `V = 2·σ(logit) − 1`.

It is fed normalized positions (and velocities, zero in the static demo) plus eight
relational features per player. Because it is set-based, it naturally generalizes to
other team sports and to any number of players.

## The probes

The demo also computes interpretable formation descriptors live (mirroring
`value/evaluation/shape.py`), so you can connect a *shape* change to a *value* change —
e.g. compress a defensive block and watch the attacker's value fall:

| probe | meaning |
|---|---|
| compactness (stretch) | mean distance of outfielders to their centroid |
| width / depth | lateral / longitudinal extent of the block |
| block area | convex-hull area of the outfield ten |
| line height | how far up-pitch the back unit holds |
| defenders goalside | defenders between the ball and their own goal |
| centroid → ball | distance from team centroid to the ball |

## How it fits together

```
value repo (PyTorch)  ──export──►  pitchperfect/data/weights.json
                                          │
                         ┌────────────────┴────────────────┐
                   pitchperfect/value_net.py          web/js/model.js
                        (numpy)                          (browser)
                                          │
                                  tests/test_parity.py
                          torch ≈ numpy ≈ JS  to ~1e-6
```

`weights.json` is the single source of truth shared by both ports.

## Development

Re-export the weights from a checkpoint in the sibling `value` repo:

```bash
cd ../value
.venv/bin/python ../pitchperfect/tools/export_from_value.py \
    --ckpt outputs/checkpoints/statsbomb_v6/latest.pt \
    --out  ../pitchperfect/pitchperfect/data
```

Run the parity + smoke tests (requires `node` for the JS checks):

```bash
pip install -e ".[test]"
pytest                              # numpy-vs-torch and JS-vs-numpy-vs-torch
node tests/dom_smoke.mjs            # headless boot of the interactive demo
```

Serve the demo locally:

```bash
python -m http.server -d web 8000   # then open http://localhost:8000/demo.html
```

## See also

- [The free-kick value study](https://humpday.microprediction.org/applications/free-kick.html)
- The [`value`](https://github.com/score-technologies/value) training repo

## License

MIT
