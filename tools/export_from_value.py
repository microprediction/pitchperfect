"""Export the trained SoccerNetV6 value network from the `value` repo into a
self-contained weights.json that pitchperfect (numpy + JS) can load.

Run this with the `value` repo's virtualenv, e.g.:

    cd ../value
    .venv/bin/python ../pitchperfect/tools/export_from_value.py \
        --ckpt outputs/checkpoints/statsbomb_v6/latest.pt \
        --out  ../pitchperfect/pitchperfect/data

It writes three files into --out:

  weights.json   all model tensors (flat float lists + shapes) + config
  refs.json      reference (state -> logit, V) cases computed by torch, used by
                 the parity test to prove the numpy/JS ports match the original
  presets.json   a few realistic 11v11 field states for the interactive demos

Velocities are exported/handled as zero throughout: the demos are static
"drag the players" scenes, which is exactly what a StatsBomb-trained freeze
frame model expects.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch


def find_value_repo() -> str:
    """Locate the sibling `value` repo so we can import its model module."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.environ.get("VALUE_REPO", ""),
        os.path.abspath(os.path.join(here, "..", "..", "value")),
        os.path.abspath(os.path.join(here, "..", "value")),
    ]
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, "value", "training", "model.py")):
            return c
    raise SystemExit(
        "Could not find the `value` repo. Set VALUE_REPO=/path/to/value."
    )


def tensor_to_json(t: torch.Tensor) -> dict:
    t = t.detach().cpu().float()
    return {"shape": list(t.shape), "data": t.reshape(-1).tolist()}


# ---------------------------------------------------------------------------
# Realistic field states (sim coordinates: x in [-50, 50], y in [-30, 30];
# blue/team0 attacks +x, red/team1 attacks -x). Velocities are zero.
# ---------------------------------------------------------------------------

def formation_442(attacking: bool) -> list[tuple[float, float]]:
    """A 4-4-2 for the +x-attacking (blue) team. `attacking` shifts it up."""
    shift = 18.0 if attacking else 0.0
    gk = (-46.0, 0.0)
    defs = [(-30 + shift, -18), (-30 + shift, -6), (-30 + shift, 6), (-30 + shift, 18)]
    mids = [(-8 + shift, -20), (-8 + shift, -7), (-8 + shift, 7), (-8 + shift, 20)]
    fwds = [(15 + shift, -8), (15 + shift, 8)]
    return [gk, *defs, *mids, *fwds]


def mirror(team: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Mirror a +x formation into a -x (red) defensive shape."""
    return [(-x, y) for (x, y) in team]


def build_presets() -> list[dict]:
    presets = []

    # 1. Kickoff-ish balanced 4-4-2 vs 4-4-2.
    presets.append({
        "name": "Balanced (4-4-2 vs 4-4-2)",
        "description": "Both teams in shape, ball at center.",
        "ball": [0.0, 0.0],
        "blue": formation_442(attacking=False),
        "red": mirror(formation_442(attacking=False)),
    })

    # 2. Blue attacking, red defending deep -- a final-third overload.
    blue = formation_442(attacking=True)
    red = mirror(formation_442(attacking=False))
    # pull red's back line deeper toward their own goal
    red = [(min(x + 6, 46) if i else x, y) for i, (x, y) in enumerate(red)]
    presets.append({
        "name": "Blue attacking third",
        "description": "Blue pushed up, red defending a deep block. Drag a striker into the box.",
        "ball": [22.0, 4.0],
        "blue": blue,
        "red": red,
    })

    # 3. Counter-attack: blue breaks with numbers, red caught high.
    blue = [(-46, 0), (-20, -16), (-18, 16), (-12, 0),
            (5, -10), (8, 10), (2, 0), (25, -6),
            (30, 6), (34, -2), (20, 18)]
    red = [(46, 0), (10, -12), (12, 12), (18, -4), (20, 6),
           (-2, -8), (0, 8), (-10, 0), (28, -16), (30, 16), (24, 0)]
    presets.append({
        "name": "Counter-attack",
        "description": "Blue breaking at speed with red caught upfield.",
        "ball": [24.0, -4.0],
        "blue": blue,
        "red": red,
    })

    return presets


def build_test_states(n: int = 11) -> list[dict]:
    """Deterministic pseudo-random states for the parity test (no RNG seed
    dependence across machines: we hand-roll a tiny LCG)."""
    states = []
    seed = 1234567
    def rnd():
        nonlocal seed
        seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
        return seed / 0x7FFFFFFF  # in [0,1)

    # A couple of the presets, plus random scatters.
    for p in build_presets():
        states.append({"ball": p["ball"], "blue": p["blue"], "red": p["red"]})
    for _ in range(6):
        ball = [(rnd() * 2 - 1) * 45, (rnd() * 2 - 1) * 25]
        blue = [[(rnd() * 2 - 1) * 48, (rnd() * 2 - 1) * 28] for _ in range(n)]
        red = [[(rnd() * 2 - 1) * 48, (rnd() * 2 - 1) * 28] for _ in range(n)]
        states.append({"ball": ball, "blue": blue, "red": red})
    return states


def state_to_inputs(state: dict, L: float, W: float, vel_scale: float):
    """Build normalized (ball_state, blue_state, red_state) torch tensors with
    zero velocity, matching value/control/smart._make_model_inputs."""
    def norm_pos(p):
        return [p[0] / L, p[1] / W]
    ball = norm_pos(state["ball"]) + [0.0, 0.0]
    blue = [norm_pos(p) + [0.0, 0.0] for p in state["blue"]]
    red = [norm_pos(p) + [0.0, 0.0] for p in state["red"]]
    bt = torch.tensor([ball], dtype=torch.float32)            # (1,4)
    blt = torch.tensor([blue], dtype=torch.float32)           # (1,N,4)
    rt = torch.tensor([red], dtype=torch.float32)             # (1,N,4)
    return bt, blt, rt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/checkpoints/statsbomb_v6/latest.pt")
    ap.add_argument("--out", required=True, help="output data directory")
    args = ap.parse_args()

    repo = find_value_repo()
    sys.path.insert(0, repo)
    from value.training.model import load_model, value_from_logit  # noqa: E402
    from value.sim.field import Field  # noqa: E402

    field = Field()
    L, W, vel_scale = field.half_length, field.half_width, field.ball_max_speed

    ckpt = args.ckpt
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(repo, ckpt)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model_version = blob.get("model_version", "v6")
    model = load_model(ckpt, device="cpu")
    model.eval()

    sd = model.state_dict()
    # The value head (head_outcome) is all we need for V; drop the future-ball
    # and auxiliary-task heads to keep the download small.
    skip = ("head_future.", "head_aux.")
    tensors = {name: tensor_to_json(t) for name, t in sd.items()
               if not name.startswith(skip)}

    config = {
        "model_version": model_version,
        "hidden": 96,
        "n_heads": 4,
        "n_horizons": 3,
        "n_aux_tasks": 3,
        "n_stack": 1,
        "n_spatial": 8,
        "sentinel": -2.0,
        "norm": {"half_length": L, "half_width": W, "vel_scale": vel_scale},
        "layernorm_eps": 1e-5,
        "value_formula": "V = 2*sigmoid(outcome_logit) - 1",
        "source_checkpoint": os.path.relpath(ckpt, repo),
        "step": int(blob.get("step", -1)),
    }

    # Canonical output (used by the python package + tests) and a copy for the
    # self-contained static site under web/data.
    web_out = os.path.abspath(os.path.join(args.out, "..", "..", "web", "data"))
    out_dirs = [args.out, web_out]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)

    def write_all(name, obj, **kw):
        for d in out_dirs:
            with open(os.path.join(d, name), "w") as f:
                json.dump(obj, f, **kw)

    write_all("weights.json", {"config": config, "tensors": tensors})
    print(f"wrote weights.json ({len(tensors)} tensors) -> {out_dirs}")

    # Reference cases for the parity test.
    refs = []
    with torch.no_grad():
        for st in build_test_states():
            bt, blt, rt = state_to_inputs(st, L, W, vel_scale)
            logit = model(bt, blt, rt)[0]
            v = value_from_logit(logit)
            refs.append({
                "state": st,
                "logit": float(logit.item()),
                "v": float(v.item()),
            })
    write_all("refs.json", {"config": config, "cases": refs}, indent=2)
    print(f"wrote refs.json ({len(refs)} cases)")

    # Presets (also stamped with the torch V so the demo can show a baseline).
    presets = build_presets()
    with torch.no_grad():
        for p in presets:
            bt, blt, rt = state_to_inputs(p, L, W, vel_scale)
            p["v_torch"] = float(value_from_logit(model(bt, blt, rt)[0]).item())
    write_all("presets.json", {"config": config, "presets": presets}, indent=2)
    print(f"wrote presets.json ({len(presets)} presets)")


if __name__ == "__main__":
    main()
