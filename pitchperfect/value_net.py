"""A numpy reimplementation of the trained SoccerNetV6 soccer value network.

The original model is a PyTorch module living in the sibling ``value`` repo. We
export its weights to ``data/weights.json`` (see ``tools/export_from_value.py``)
and reimplement the forward pass here with numpy alone, so pitchperfect installs
without torch. ``web/js/model.js`` is a line-for-line JavaScript port of this
same forward pass; ``tests/test_parity.py`` proves all three agree.

Architecture (see value/value/training/model.py::SoccerNetV6):

    per-player spatial features (8) -> shared player MLP encoder
    ball MLP encoder
    cross-team self-attention over [ball, 11 blue, 11 red] tokens
    masked self-attention pooling per team -> team embeddings
    concat[ball, blue, red] -> outcome head -> logit
    V = 2*sigmoid(logit) - 1   in [-1, +1]; +1 favors blue (the +x attacker)

All inputs are in *sim coordinates*: x in [-half_length, half_length],
y in [-half_width, half_width]; blue/team0 attacks +x. Velocities default to 0.
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


def _layernorm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


class ValueNet:
    """Numpy SoccerNetV6. Load with ``ValueNet.load()``."""

    def __init__(self, weights: dict):
        self.cfg = weights["config"]
        self.t = {k: np.array(v["data"], dtype=np.float64).reshape(v["shape"])
                  for k, v in weights["tensors"].items()}
        self.hidden = self.cfg["hidden"]
        self.n_heads = self.cfg["n_heads"]
        self.eps = self.cfg["layernorm_eps"]
        self.sentinel = self.cfg["sentinel"]
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

    # --- attention ----------------------------------------------------------
    def _mha(self, x, prefix, key_padding_mask=None):
        """PyTorch nn.MultiheadAttention self-attention (batch_first).

        x: (B, S, D). key_padding_mask: (B, S) bool, True = ignore key.
        Returns (B, S, D).
        """
        B, S, D = x.shape
        H, hd = self.n_heads, D // self.n_heads
        Wi = self.t[prefix + ".in_proj_weight"]      # (3D, D)
        bi = self.t[prefix + ".in_proj_bias"]        # (3D,)
        q = x @ Wi[0:D].T + bi[0:D]
        k = x @ Wi[D:2 * D].T + bi[D:2 * D]
        v = x @ Wi[2 * D:3 * D].T + bi[2 * D:3 * D]
        # (B, H, S, hd)
        def split(z):
            return z.reshape(B, S, H, hd).transpose(0, 2, 1, 3)
        q, k, v = split(q), split(k), split(v)
        scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(hd)   # (B,H,S,S)
        if key_padding_mask is not None:
            m = key_padding_mask[:, None, None, :]             # (B,1,1,S)
            scores = np.where(m, -np.inf, scores)
        scores = scores - scores.max(axis=-1, keepdims=True)
        e = np.exp(scores)
        attn = e / e.sum(axis=-1, keepdims=True)
        out = attn @ v                                         # (B,H,S,hd)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, D)
        return out @ self.t[prefix + ".out_proj.weight"].T + self.t[prefix + ".out_proj.bias"]

    # --- spatial features ---------------------------------------------------
    def _spatial(self, players, opponents, ball_pos, pmask, omask,
                 own_goal_x, opp_goal_x, density_radius=0.3):
        """8 relational features per player; (B, N, 8). Mirrors
        value/training/model.py::_compute_spatial_features_masked."""
        B, N, _ = players.shape
        pos = players[..., :2]
        vel = players[..., 2:4]
        opp = opponents[..., :2]
        ball = ball_pos[:, None, :]                            # (B,1,2)

        d_ball = np.linalg.norm(pos - ball, axis=-1)
        own_goal = np.array([own_goal_x, 0.0])
        opp_goal = np.array([opp_goal_x, 0.0])
        d_own = np.linalg.norm(pos - own_goal, axis=-1)
        d_opp = np.linalg.norm(pos - opp_goal, axis=-1)
        to_opp = opp_goal - pos
        cos_angle = to_opp[..., 0] / (np.linalg.norm(to_opp, axis=-1) + 1e-6)

        d_to_opp = np.linalg.norm(pos[:, :, None, :] - opp[:, None, :, :], axis=-1)  # (B,N,M)
        d_to_opp = np.where(omask[:, None, :], 1e6, d_to_opp)
        nearest_opp = d_to_opp.min(axis=-1)

        d_to_team = np.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)  # (B,N,N)
        eye = np.eye(N, dtype=bool)[None]
        d_to_team = np.where(eye | pmask[:, None, :], 1e6, d_to_team)
        nearest_team = d_to_team.min(axis=-1)

        n_valid_opp = np.clip((~omask).sum(axis=-1, keepdims=True), 1, None)
        density = (d_to_opp < density_radius).sum(axis=-1) / n_valid_opp

        to_ball = ball - pos
        unit = to_ball / (np.linalg.norm(to_ball, axis=-1, keepdims=True) + 1e-6)
        vel_to_ball = (vel * unit).sum(axis=-1)

        feats = np.stack([d_ball, d_own, d_opp, cos_angle, nearest_opp,
                          nearest_team, density, vel_to_ball], axis=-1)
        feats = np.where(pmask[..., None], 0.0, feats)
        return feats

    # --- forward ------------------------------------------------------------
    def forward(self, ball_state, blue_state, red_state):
        """ball_state (B,4); blue_state/red_state (B,N,4) -- normalized. Returns
        outcome logits (B,)."""
        ball_pos = ball_state[:, :2]
        bmask = np.all(blue_state[..., :2] == self.sentinel, axis=-1)
        rmask = np.all(red_state[..., :2] == self.sentinel, axis=-1)

        blue_extra = self._spatial(blue_state, red_state, ball_pos, bmask, rmask, -1.0, 1.0)
        red_extra = self._spatial(red_state, blue_state, ball_pos, rmask, bmask, 1.0, -1.0)
        blue_in = np.concatenate([blue_state, blue_extra], axis=-1)
        red_in = np.concatenate([red_state, red_extra], axis=-1)

        ball_emb = self._mlp(ball_state, "ball_encoder")           # (B,D)
        blue_enc = self._mlp(blue_in, "player_encoder")            # (B,N,D)
        red_enc = self._mlp(red_in, "player_encoder")

        ball_emb, blue_enc, red_enc = self._cross_attn(ball_emb, blue_enc, red_enc, bmask, rmask)

        blue_pool = self._masked_pool(blue_enc, bmask, "blue_pool")
        red_pool = self._masked_pool(red_enc, rmask, "red_pool")
        merged = np.concatenate([ball_emb, blue_pool, red_pool], axis=-1)   # (B,3D)
        return self._mlp(merged, "head_outcome")[:, 0]

    def _cross_attn(self, ball_emb, blue_enc, red_enc, bmask, rmask):
        B, N, D = blue_enc.shape
        blue_enc = np.where(bmask[..., None], 0.0, blue_enc)
        red_enc = np.where(rmask[..., None], 0.0, red_enc)
        tokens = np.concatenate([ball_emb[:, None, :], blue_enc, red_enc], axis=1)  # (B,1+2N,D)
        te = self.t["cross_attn.team_embed.weight"]                # (3,D)
        team = np.concatenate([
            np.zeros((B, 1), dtype=int),
            np.ones((B, N), dtype=int),
            np.full((B, N), 2, dtype=int),
        ], axis=1)
        tokens = tokens + te[team]
        pad = np.concatenate([
            np.zeros((B, 1), dtype=bool), bmask, rmask], axis=1)
        attn = self._mha(tokens, "cross_attn.attn", key_padding_mask=pad)
        tokens = _layernorm(tokens + attn, self.t["cross_attn.norm.weight"],
                            self.t["cross_attn.norm.bias"], self.eps)
        ball_out = tokens[:, 0, :]
        blue_out = tokens[:, 1:1 + N, :]
        red_out = tokens[:, 1 + N:, :]
        return ball_out, blue_out, red_out

    def _masked_pool(self, x, mask, prefix):
        x = np.where(mask[..., None], 0.0, x)
        attn = self._mha(x, prefix + ".attn", key_padding_mask=mask)
        x = _layernorm(x + attn, self.t[prefix + ".norm.weight"],
                       self.t[prefix + ".norm.bias"], self.eps)
        valid = (~mask)[..., None].astype(np.float64)
        n_valid = np.clip(valid.sum(axis=1), 1, None)
        return (x * valid).sum(axis=1) / n_valid

    # --- convenience --------------------------------------------------------
    def _state_to_inputs(self, ball, blue, red):
        """Build normalized (B=1) inputs from sim coords. ball=[x,y] or
        [x,y,vx,vy]; blue/red = lists of [x,y] (or with velocity)."""
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

    def value(self, ball: Sequence, blue: Sequence, red: Sequence) -> float:
        """Signed value V in [-1, +1] for a state in sim coordinates.
        +1 strongly favors blue (the +x-attacking team)."""
        return float(2.0 * _sigmoid(self.forward(*self._state_to_inputs(ball, blue, red))[0]) - 1.0)
