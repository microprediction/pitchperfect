"""MettleNet -- a numpy reimplementation of the trained soccer value network.

The network (a permutation-invariant Deep Sets model: per-player MLP encoder +
sum-pooling per team + a value head) is trained in PyTorch in the sibling
``value`` repo. We export its weights to ``data/weights.json`` (see
``tools/export_from_value.py``) and reimplement the forward pass here with numpy
alone, so pitchperfect installs without torch. ``web/js/model.js`` is a matching
JavaScript port; ``tests/test_parity.py`` proves all three agree.

Architecture:

    per-player MLP encoder  ->  sum-pool each team
    ball MLP encoder
    concat[ball, blue_sum, red_sum]  ->  outcome head  ->  logit
    V = 2*sigmoid(logit) - 1

``value()`` returns the **antisymmetrized** value, which removes any residual
red/blue bias in the trained weights and guarantees V(s) = -V(swap(s)) exactly,
where swap flips the pitch in x and swaps the two teams. +1 favors blue (the
team attacking +x); a perfectly balanced, mirror-symmetric position reads 0.

All inputs are in sim coordinates: x in [-half_length, half_length],
y in [-half_width, half_width]; blue attacks +x. Velocities default to 0.
"""
from __future__ import annotations

import json
import os
from typing import Sequence

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# --- erf / GELU -------------------------------------------------------------
# Abramowitz & Stegun 7.1.26 (max abs error ~1.5e-7). The JS port uses the
# identical formula so numpy and JS agree to ~1e-12; both sit ~1e-7 from torch's
# exact erf, which the parity test accounts for with a generous torch tolerance.
def _erf(x: np.ndarray) -> np.ndarray:
    a1, a2, a3, a4, a5 = (
        0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429,
    )
    p = 0.3275911
    s = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + p * ax)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-ax * ax)
    return s * y


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class ValueNet:
    """Numpy MettleNet. Load with ``ValueNet.load()``."""

    def __init__(self, weights: dict):
        self.cfg = weights["config"]
        self.t = {k: np.array(v["data"], dtype=np.float64).reshape(v["shape"])
                  for k, v in weights["tensors"].items()}
        self.hidden = self.cfg["hidden"]
        n = self.cfg["norm"]
        self.L = n["half_length"]
        self.W = n["half_width"]
        self.vel_scale = n["vel_scale"]

    @classmethod
    def load(cls, path: str | None = None) -> "ValueNet":
        path = path or os.path.join(DATA_DIR, "weights.json")
        with open(path) as f:
            return cls(json.load(f))

    # --- linear / mlp -------------------------------------------------------
    def _linear(self, x, name):
        return x @ self.t[name + ".weight"].T + self.t[name + ".bias"]

    def _mlp(self, x, prefix):
        x = _gelu(self._linear(x, prefix + ".0"))
        x = _gelu(self._linear(x, prefix + ".2"))
        return self._linear(x, prefix + ".4")

    # --- forward ------------------------------------------------------------
    def forward(self, ball_state, blue_state, red_state):
        """ball_state (B,4); blue_state/red_state (B,N,4) normalized. Returns
        outcome logits (B,)."""
        ball_emb = self._mlp(ball_state, "ball_encoder")            # (B,D)
        blue_emb = self._mlp(blue_state, "player_encoder").sum(axis=1)
        red_emb = self._mlp(red_state, "player_encoder").sum(axis=1)
        merged = np.concatenate([ball_emb, blue_emb, red_emb], axis=-1)
        return self._mlp(merged, "head_outcome")[:, 0]

    # --- input building -----------------------------------------------------
    def _state_to_inputs(self, ball, blue, red):
        def norm(p):
            vx = p[2] if len(p) > 2 else 0.0
            vy = p[3] if len(p) > 3 else 0.0
            return [p[0] / self.L, p[1] / self.W, vx / self.vel_scale, vy / self.vel_scale]
        bt = np.array([norm(ball)], dtype=np.float64)
        blt = np.array([[norm(p) for p in blue]], dtype=np.float64)
        rt = np.array([[norm(p) for p in red]], dtype=np.float64)
        return bt, blt, rt

    def logit(self, ball, blue, red) -> float:
        return float(self.forward(*self._state_to_inputs(ball, blue, red))[0])

    def value_raw(self, ball, blue, red) -> float:
        """Raw signed value in [-1, +1] (no symmetrization)."""
        return float(2.0 * _sigmoid(self.forward(*self._state_to_inputs(ball, blue, red))[0]) - 1.0)

    @staticmethod
    def _swap(ball, blue, red):
        """swap(s): flip x (positions + x-velocity) and swap the two teams."""
        def fx(p):
            q = list(p) + [0.0, 0.0]
            return [-q[0], q[1], -q[2], q[3]]
        return fx(ball), [fx(p) for p in red], [fx(p) for p in blue]

    def value(self, ball: Sequence, blue: Sequence, red: Sequence) -> float:
        """Antisymmetrized signed value in [-1, +1]; +1 favors blue (the +x
        attacker). V(s) = (raw(s) - raw(swap s))/2, so V(s) = -V(swap s) exactly
        and a mirror-balanced position reads 0. This corrects the model's
        position-dependent red/blue asymmetry at every point (a single additive
        bias cannot)."""
        v = self.value_raw(ball, blue, red)
        vs = self.value_raw(*self._swap(ball, blue, red))
        return 0.5 * (v - vs)
