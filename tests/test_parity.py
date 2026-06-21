"""Parity tests: the JavaScript port, the numpy port, and the original torch
model must all agree on V(s).

  - refs.json carries the torch-computed (logit, V) for each reference state.
  - The numpy port is run here directly.
  - The JS port is run via node (tests/run_js_refs.mjs) and compared.

torch vs numpy/JS differ only by the erf approximation (~1e-6). numpy vs JS use
the identical erf formula and agree far tighter.
"""
import json
import os
import shutil
import subprocess

try:
    import pytest
except ImportError:                      # allow the __main__ self-check to run
    pytest = None

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "pitchperfect", "data")

TORCH_TOL = 1e-4   # numpy/JS A&S-erf vs torch exact-erf
JS_TOL = 1e-6      # JS vs numpy (same formula, both float64)


def _load_refs():
    with open(os.path.join(DATA, "refs.json")) as f:
        return json.load(f)["cases"]


def _numpy_outputs(refs):
    from pitchperfect import ValueNet
    net = ValueNet.load()
    out = []
    for c in refs:
        s = c["state"]
        out.append({"logit": net.logit(s["ball"], s["blue"], s["red"]),
                    "v": net.value(s["ball"], s["blue"], s["red"])})
    return out


def _js_outputs():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed; skipping JS parity")
    res = subprocess.run(
        [node, os.path.join(HERE, "run_js_refs.mjs")],
        capture_output=True, text=True, cwd=ROOT,
    )
    if res.returncode != 0:
        raise AssertionError(f"node runner failed:\n{res.stderr}")
    return json.loads(res.stdout)


def test_numpy_matches_torch():
    refs = _load_refs()
    npy = _numpy_outputs(refs)
    for ref, got in zip(refs, npy):
        assert abs(got["logit"] - ref["logit"]) < TORCH_TOL, (got, ref)
        assert abs(got["v"] - ref["v"]) < TORCH_TOL


def test_js_matches_numpy_and_torch():
    refs = _load_refs()
    npy = _numpy_outputs(refs)
    js = _js_outputs()
    assert len(js) == len(refs)
    for ref, n, j in zip(refs, npy, js):
        assert abs(j["logit"] - n["logit"]) < JS_TOL, ("js vs numpy", j, n)
        assert abs(j["logit"] - ref["logit"]) < TORCH_TOL, ("js vs torch", j, ref)
        assert abs(j["v"] - ref["v"]) < TORCH_TOL


if __name__ == "__main__":
    refs = _load_refs()
    npy = _numpy_outputs(refs)
    js = _js_outputs()
    dmax_jn = max(abs(j["logit"] - n["logit"]) for j, n in zip(js, npy))
    dmax_jt = max(abs(j["logit"] - r["logit"]) for j, r in zip(js, refs))
    print(f"max |JS - numpy| logit = {dmax_jn:.2e}")
    print(f"max |JS - torch| logit = {dmax_jt:.2e}")
    for r, j in zip(refs, js):
        print(f"  torch {r['v']:+.4f}   js {j['v']:+.4f}")
