"""Diagnose which checkpoint/config gives sane interactive values.

Evaluates the demo presets (known expected signs) under several candidate
checkpoints, with zero velocity (as the demo uses), comparing:
  - raw V
  - de-biased V (current site approach: subtract V(balanced))
  - antisymmetrized V = (V(s) - V(mirror(s))) / 2   [mirror = flip-x + swap teams]

Run with the value repo's env:
    cd ../value && .venv/bin/python ../pitchperfect/tools/diagnose_models.py
"""
import os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
VALUE = os.path.abspath(os.path.join(HERE, "..", "..", "value"))
sys.path.insert(0, VALUE)
sys.path.insert(0, os.path.join(HERE))  # to import build_presets

from value.training.model import load_model, value_from_logit
from value.sim.field import Field
from export_from_value import build_presets

L, W = Field().half_length, Field().half_width   # 50, 30

CANDIDATES = {
    "v8 (DEPLOYED on site)": "outputs/checkpoints/v8/latest.pt",
    "v10 (sim, intended)": "outputs/checkpoints/v10/latest.pt",
    "statsbomb_v6":       "outputs/checkpoints/statsbomb_v6/latest.pt",
    "statsbomb_v6mt":     "outputs/checkpoints/statsbomb_v6mt/latest.pt",
}

EXPECT = {
    "Balanced (4-4-2 vs 4-4-2)": "~0",
    "Blue attacking third": "mild +",
    "Counter-attack": "+",
    "Blue open goal": "STRONG +",
    "Red open goal": "STRONG -",
    "Blue corner": "+",
}


def norm(p):
    return [p[0] / L, p[1] / W, 0.0, 0.0]


def to_inputs(st):
    ball = torch.tensor([norm(st["ball"][:2] if len(st["ball"]) >= 2 else st["ball"])],
                        dtype=torch.float32)
    blue = torch.tensor([[norm(p) for p in st["blue"]]], dtype=torch.float32)
    red = torch.tensor([[norm(p) for p in st["red"]]], dtype=torch.float32)
    return ball, blue, red


def mirror(st):
    """Flip x of every position and swap teams: a true pitch reflection."""
    fb = [-st["ball"][0], st["ball"][1]]
    fblue = [[-x, y] for x, y in st["red"]]   # red becomes flipped blue
    fred = [[-x, y] for x, y in st["blue"]]
    return {"ball": fb, "blue": fblue, "red": fred}


@torch.no_grad()
def raw_v(model, st):
    b, bl, r = to_inputs(st)
    return float(value_from_logit(model(b, bl, r)[0]).item())


def main():
    presets = build_presets()
    bias_cache = {}
    for label, ckpt in CANDIDATES.items():
        path = os.path.join(VALUE, ckpt)
        if not os.path.isfile(path):
            print(f"\n### {label}: MISSING {ckpt}")
            continue
        model = load_model(path, device="cpu"); model.eval()
        bias = raw_v(model, presets[0])     # V(balanced), the site's de-bias
        print(f"\n### {label}   (bias V(balanced)={bias:+.3f})")
        print(f"{'preset':<28}{'expect':>9}{'raw':>9}{'debiased':>10}{'antisym':>9}")
        for p in presets:
            r = raw_v(model, p)
            d = r - bias
            a = (r - raw_v(model, mirror(p))) / 2.0
            print(f"{p['name']:<28}{EXPECT.get(p['name'],''):>9}"
                  f"{r:>9.3f}{d:>10.3f}{a:>9.3f}")


if __name__ == "__main__":
    main()
