"""Exhaustive sanity study of the (antisymmetrized) value function.

Renders a battery of hand-built scenarios and several systematic position
sweeps as PNGs, each annotated with V, so the surface can be eyeballed for
clearly-wrong behaviour. Run with the value repo's venv:

    cd ../value
    .venv/bin/python ../pitchperfect/tools/study.py --out ../pitchperfect/outputs/study
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_value_repo():
    here = os.path.dirname(os.path.abspath(__file__))
    for c in [os.environ.get("VALUE_REPO", ""),
              os.path.abspath(os.path.join(here, "..", "..", "value")),
              os.path.abspath(os.path.join(here, "..", "value"))]:
        if c and os.path.isfile(os.path.join(c, "value", "training", "model.py")):
            return c
    raise SystemExit("value repo not found")


REPO = find_value_repo()
sys.path.insert(0, REPO)
from value.training.model import load_model, value_from_logit  # noqa: E402
from value.sim.field import Field  # noqa: E402

FIELD = Field()
L, W, VS = FIELD.half_length, FIELD.half_width, FIELD.ball_max_speed
CKPT = os.path.join(REPO, "outputs/checkpoints/v10/latest.pt")
MODEL = load_model(CKPT)
MODEL.eval()


def _nz(p):
    vx = p[2] if len(p) > 2 else 0.0
    vy = p[3] if len(p) > 3 else 0.0
    return [p[0] / L, p[1] / W, vx / VS, vy / VS]


@torch.no_grad()
def raw_batch(balls, blues, reds):
    bt = torch.tensor([[_nz(b)] for b in balls], dtype=torch.float32).squeeze(1)
    blt = torch.tensor([[[_nz(p) for p in bl]] for bl in blues], dtype=torch.float32).squeeze(1)
    rt = torch.tensor([[[_nz(p) for p in rd]] for rd in reds], dtype=torch.float32).squeeze(1)
    return value_from_logit(MODEL(bt, blt, rt)[0]).numpy()


def swap(ball, blue, red):
    fx = lambda p: [-p[0], p[1]]
    return fx(ball), [fx(p) for p in red], [fx(p) for p in blue]


def antisym_batch(balls, blues, reds):
    v = raw_batch(balls, blues, reds)
    sb, sbl, srd = [], [], []
    for b, bl, rd in zip(balls, blues, reds):
        s = swap(b, bl, rd); sb.append(s[0]); sbl.append(s[1]); srd.append(s[2])
    vs = raw_batch(sb, sbl, srd)
    return 0.5 * (v - vs)


def V(ball, blue, red):
    return float(antisym_batch([ball], [blue], [red])[0])


# --- formations -------------------------------------------------------------
def f442(shift=0):
    return [[-46, 0]] + [[-30 + shift, y] for y in (-18, -6, 6, 18)] + \
           [[-8 + shift, y] for y in (-20, -7, 7, 20)] + [[15 + shift, -8], [15 + shift, 8]]


def mir(t):
    return [[-x, y] for x, y in t]


# --- drawing ----------------------------------------------------------------
def draw_pitch(ax):
    ax.add_patch(plt.Rectangle((-L, -W), 2 * L, 2 * W, color="#2f8f4e", zorder=0))
    ax.plot([0, 0], [-W, W], color="white", lw=1)
    ax.add_patch(plt.Circle((0, 0), 9.15, fill=False, color="white", lw=1))
    for sx in (-1, 1):
        ax.add_patch(plt.Rectangle((sx * L - sx * 16.5 if sx > 0 else -L, -20),
                                   16.5, 40, fill=False, color="white", lw=1))
    ax.plot([-L, -L], [-13, 13], color="white", lw=3)
    ax.plot([L, L], [-13, 13], color="white", lw=3)
    ax.set_xlim(-L - 2, L + 2); ax.set_ylim(-W - 2, W + 2)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def draw_state(ax, ball, blue, red, title):
    draw_pitch(ax)
    for i, p in enumerate(red):
        ax.add_patch(plt.Circle((p[0], p[1]), 1.6, color="#e5484d", ec="white", lw=0.8, zorder=3))
    for i, p in enumerate(blue):
        ax.add_patch(plt.Circle((p[0], p[1]), 1.6, color="#1f6feb", ec="white", lw=0.8, zorder=3))
    ax.add_patch(plt.Circle((ball[0], ball[1]), 1.0, color="white", ec="black", lw=1, zorder=4))
    ax.set_title(title, fontsize=10)


# --- scenarios --------------------------------------------------------------
def scenarios():
    S = []
    S.append(("Balanced 4-4-2", [0, 0], f442(), mir(f442())))
    S.append(("Blue attacking third", [22, 4], f442(18), mir(f442())))
    # open goals
    S.append(("Blue open goal", [48, 1],
              [[-44, 0], [47, -1], [44, 3], [44, -4], [30, 10], [30, -10], [20, 0], [-10, 8], [-10, -8], [-20, 0], [-5, 0]],
              [[25, 0], [40, 8], [40, -8], [35, 0], [20, 12], [20, -12], [0, 6], [0, -6], [-15, 0], [-25, 5], [-25, -5]]))
    S.append(("Red open goal", [-48, 0],
              [[-25, 0], [-40, -8], [-40, 8], [-35, 0], [-20, -12], [-20, 12], [0, -6], [0, 6], [15, 0], [25, -5], [25, 5]],
              [[44, 0], [-47, 1], [-44, -3], [-44, 4], [-30, -10], [-30, 10], [-20, 0], [10, -8], [10, 8], [20, 0], [5, 0]]))
    # numerical overloads in the box
    b3_blue = [[-46, 0], [42, -4], [42, 4], [40, 0], [20, 0], [10, 10], [10, -10], [0, 0], [-10, 0], [-20, 5], [-20, -5]]
    b3_red = [[49, 0], [44, 0], [-10, 0], [-15, 8], [-15, -8], [-25, 0], [-30, 10], [-30, -10], [-35, 0], [-40, 5], [-40, -5]]
    S.append(("Blue 3v1 in red box", [40, 0], b3_blue, b3_red))
    # true swap (flip x + swap teams) -> should read as the exact negative
    S.append(("Red 3v1 in blue box", [-40, 0], mir(b3_red), mir(b3_blue)))
    # defensive shape: compact vs stretched red block, blue attacking
    base_blue = f442(15)
    compact_red = [[46, 0]] + [[20 + (i % 3) * 3, -4 + (i // 3) * 3] for i in range(10)]
    stretch_red = [[46, 0]] + [[8 + (i % 5) * 8, -24 + (i // 5) * 12] for i in range(10)]
    S.append(("Blue attack vs compact red", [18, 0], base_blue, compact_red))
    S.append(("Blue attack vs stretched red", [18, 0], base_blue, stretch_red))
    # high line caught
    S.append(("Red high line, blue in behind", [30, 0],
              [[-46, 0], [28, 0], [10, -10], [10, 10], [0, 0], [-10, 0], [-20, 8], [-20, -8], [-30, 0], [-35, 6], [-35, -6]],
              [[46, 0], [18, -6], [18, 6], [20, 0], [22, -12], [22, 12], [35, 0], [40, 8], [40, -8], [44, 4], [44, -4]]))
    # ball in own third (possession)
    S.append(("Blue keeper plays out (own third)", [-38, 6], f442(), mir(f442())))
    # central vs wide attack
    S.append(("Blue wide overload (right wing)", [30, 22],
              [[-46, 0], [32, 24], [30, 16], [28, 8], [20, 20], [10, 0], [0, 10], [-10, 0], [-20, 8], [-20, -8], [-30, 0]],
              mir(f442())))
    # 1v1 with keeper
    S.append(("Blue striker 1v1 keeper", [42, 0],
              [[-46, 0], [44, 0], [20, 0], [10, 8], [10, -8], [0, 0], [-10, 6], [-10, -6], [-20, 0], [-30, 5], [-30, -5]],
              [[49, 0], [25, -6], [25, 6], [15, 0], [5, 10], [5, -10], [-5, 0], [-15, 8], [-15, -8], [-25, 0], [-35, 0]]))
    # corner
    S.append(("Blue corner (red goal)", [48, 27],
              [[-46, 0], [48, 27], [44, 4], [45, -2], [43, 8], [46, 0], [40, -6], [25, 12], [15, 0], [0, 0], [-8, 0]],
              [[49, 0], [45, 3], [45, -3], [44, 6], [44, -6], [43, 0], [46, 9], [46, -9], [40, 0], [30, 0], [20, 0]]))
    # ball on the goal line dead centre (extreme)
    S.append(("Ball on blue goal line", [-49, 0], f442(), mir(f442())))
    S.append(("Ball on red goal line", [49, 0], f442(), mir(f442())))
    return S


def render_scenarios(out):
    S = scenarios()
    vs = antisym_batch([s[1] for s in S], [s[2] for s in S], [s[3] for s in S])
    n = len(S); cols = 4; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 2.8))
    axes = axes.ravel()
    data = []
    for k, (name, ball, blue, red) in enumerate(S):
        draw_state(axes[k], ball, blue, red, f"{name}\nV = {vs[k]:+.2f}")
        data.append((name, float(vs[k])))
    for k in range(n, len(axes)):
        axes[k].axis("off")
    fig.tight_layout()
    p = os.path.join(out, "scenarios.png")
    fig.savefig(p, dpi=110); plt.close(fig)
    return data


def render_sweep(out, who="blue", base_shift=0):
    """Sweep one attacker (+the ball) over the pitch; others fixed; plot V."""
    nx, ny = 50, 30
    blue0 = f442(base_shift); red0 = mir(f442())
    xs = np.linspace(-L + 3, L - 3, nx); ys = np.linspace(W - 2, -W + 2, ny)
    balls, blues, reds = [], [], []
    for yy in ys:
        for xx in xs:
            bl = [r[:] for r in blue0]; rd = [r[:] for r in red0]
            if who == "blue":
                bl[9] = [xx, yy]
            else:
                rd[9] = [xx, yy]
            balls.append([xx, yy]); blues.append(bl); reds.append(rd)
    v = antisym_batch(balls, blues, reds).reshape(ny, nx)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    draw_pitch(ax)
    m = np.abs(v).max()
    im = ax.imshow(v, extent=[-L, L, -W, W], origin="upper", cmap="RdBu",
                   vmin=-m, vmax=m, alpha=0.85, zorder=1)
    for p in (blue0 if who == "blue" else red0):
        ax.add_patch(plt.Circle((p[0], p[1]), 1.2,
                     color=("#1f6feb" if who == "blue" else "#e5484d"), ec="white", lw=0.5, zorder=3))
    for p in (red0 if who == "blue" else blue0):
        ax.add_patch(plt.Circle((p[0], p[1]), 1.2,
                     color=("#e5484d" if who == "blue" else "#1f6feb"), ec="white", lw=0.5, zorder=3))
    fig.colorbar(im, ax=ax, label="antisym V")
    ax.set_title(f"Marginal value: move {who} attacker + ball over pitch\n(red = favors red, blue = favors blue)")
    fig.tight_layout()
    fname = os.path.join(out, f"sweep_{who}.png")
    fig.savefig(fname, dpi=110); plt.close(fig)
    return v


def render_ball_only_sweep(out):
    nx, ny = 50, 30
    blue0 = f442(); red0 = mir(f442())
    xs = np.linspace(-L + 3, L - 3, nx); ys = np.linspace(W - 2, -W + 2, ny)
    balls = [[xx, yy] for yy in ys for xx in xs]
    blues = [blue0] * len(balls); reds = [red0] * len(balls)
    v = antisym_batch(balls, blues, reds).reshape(ny, nx)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    draw_pitch(ax)
    m = max(np.abs(v).max(), 1e-3)
    im = ax.imshow(v, extent=[-L, L, -W, W], origin="upper", cmap="RdBu", vmin=-m, vmax=m, alpha=0.85)
    for p in blue0: ax.add_patch(plt.Circle((p[0], p[1]), 1.2, color="#1f6feb", ec="white", lw=0.5, zorder=3))
    for p in red0: ax.add_patch(plt.Circle((p[0], p[1]), 1.2, color="#e5484d", ec="white", lw=0.5, zorder=3))
    fig.colorbar(im, ax=ax, label="antisym V")
    ax.set_title("Marginal value: move only the BALL (balanced 4-4-2)\n(red = favors red, blue = favors blue)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "sweep_ball.png"), dpi=110); plt.close(fig)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "outputs/study"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    data = render_scenarios(args.out)
    render_sweep(args.out, "blue")
    render_sweep(args.out, "red")
    render_ball_only_sweep(args.out)
    print("scenario values:")
    for name, v in data:
        print(f"  {name:34s} {v:+.3f}")
    print(f"\nwrote images to {args.out}")


if __name__ == "__main__":
    main()
