// pitchperfect interactive demo: drag players, watch V(s) and shape probes,
// or pick a player and sweep them to see the marginal value surface.
import { ValueNet } from "./model.js";
import { teamProbes } from "./probes.js";

const SIM = { L: 50, W: 30 };            // half-length, half-width
const CW = 1000, CH = 600;               // canvas internal resolution (5:3)
const PLAYER_R = 13, BALL_R = 8;

const canvas = document.getElementById("pitch");
const ctx = canvas.getContext("2d");
canvas.width = CW; canvas.height = CH;

const el = (id) => document.getElementById(id);
const state = {
  net: null,
  ball: [0, 0],
  blue: [], red: [],
  mode: "value",                          // "value" | "marginal"
  selected: null,                         // {team, idx}
  heat: null,                             // {grid, min, max, best}
  dragging: null,
};

// --- coordinate transforms --------------------------------------------------
const toCanvas = (x, y) => [ (x + SIM.L) / (2 * SIM.L) * CW, (SIM.W - y) / (2 * SIM.W) * CH ];
const toSim = (cx, cy) => [ cx / CW * 2 * SIM.L - SIM.L, SIM.W - cy / CH * 2 * SIM.W ];

// --- pitch drawing ----------------------------------------------------------
function drawPitch() {
  // striped turf
  const stripes = 10, sw = CW / stripes;
  for (let i = 0; i < stripes; i++) {
    ctx.fillStyle = i % 2 ? "#2a8347" : "#2f8f4e";
    ctx.fillRect(i * sw, 0, sw, CH);
  }
  ctx.strokeStyle = "rgba(255,255,255,0.85)";
  ctx.lineWidth = 3;
  ctx.strokeRect(12, 12, CW - 24, CH - 24);
  // halfway line + center circle
  ctx.beginPath(); ctx.moveTo(CW / 2, 12); ctx.lineTo(CW / 2, CH - 12); ctx.stroke();
  ctx.beginPath(); ctx.arc(CW / 2, CH / 2, 70, 0, 2 * Math.PI); ctx.stroke();
  ctx.beginPath(); ctx.arc(CW / 2, CH / 2, 4, 0, 2 * Math.PI); ctx.fillStyle = "#fff"; ctx.fill();
  // penalty boxes
  const boxH = CH * (40 / 60), boxY = (CH - boxH) / 2, boxW = CW * (16.5 / 100);
  ctx.strokeRect(12, boxY, boxW, boxH);
  ctx.strokeRect(CW - 12 - boxW, boxY, boxW, boxH);
  const sixH = CH * (18 / 60), sixY = (CH - sixH) / 2, sixW = CW * (5.5 / 100);
  ctx.strokeRect(12, sixY, sixW, sixH);
  ctx.strokeRect(CW - 12 - sixW, sixY, sixW, sixH);
  // goals
  const gH = CH * (26 / 60), gY = (CH - gH) / 2;
  ctx.lineWidth = 5;
  ctx.strokeStyle = "rgba(255,255,255,0.95)";
  ctx.beginPath(); ctx.moveTo(12, gY); ctx.lineTo(12, gY + gH); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(CW - 12, gY); ctx.lineTo(CW - 12, gY + gH); ctx.stroke();
}

// --- heatmap (marginal value surface) ---------------------------------------
// Computed asynchronously in time-budgeted chunks so the tab never freezes;
// the surface fills in progressively and is cancelable if the user grabs a
// different player.
const HEAT_NX = 48, HEAT_NY = 29;
let heatToken = 0;
const raf = () => new Promise((r) => requestAnimationFrame(r));

async function computeHeatAsync() {
  if (!state.selected) { state.heat = null; return; }
  const token = ++heatToken;
  const { team, idx } = state.selected;
  const arr = team === "blue" ? state.blue : state.red;
  const orig = arr[idx].slice();
  const sign = team === "blue" ? 1 : -1;       // value from this team's view
  const nx = HEAT_NX, ny = HEAT_NY, total = nx * ny;
  const vals = new Float32Array(total);
  state.heat = { nx, ny, vals, count: 0, min: Infinity, max: -Infinity, best: null, done: false };
  let last = performance.now();
  for (let c = 0; c < total; c++) {
    const i = c % nx, j = (c / nx) | 0;
    const sx = (i + 0.5) / nx * 2 * SIM.L - SIM.L;
    const sy = SIM.W - (j + 0.5) / ny * 2 * SIM.W;
    arr[idx] = [sx, sy];
    const v = sign * state.net.value(state.ball, state.blue, state.red);
    vals[c] = v;
    const h = state.heat;
    if (v < h.min) h.min = v;
    if (v > h.max) { h.max = v; h.best = [sx, sy]; }
    h.count = c + 1;
    if (performance.now() - last > 12) {       // yield to keep UI responsive
      arr[idx] = orig;                          // restore while painting
      render();
      el("selinfo").textContent = `Sweeping ${team} ${idx === 0 ? "GK" : "#" + idx} … ${Math.round(100 * h.count / total)}%`;
      await raf();
      if (token !== heatToken) return;          // superseded -> abandon
      arr[idx] = [sx, sy];
      last = performance.now();
    }
  }
  arr[idx] = orig;
  state.heat.done = true;
  el("selinfo").textContent = `Sweeping ${team} ${idx === 0 ? "GK" : "#" + idx} — ✚ = best spot`;
  render();
}

function heatColor(t) {                          // t in [0,1] -> blue->green->yellow->red
  const stops = [[43,58,143],[47,143,78],[244,224,77],[229,72,77]];
  const x = Math.max(0, Math.min(1, t)) * (stops.length - 1);
  const i = Math.floor(x), f = x - i;
  const a = stops[i], b = stops[Math.min(i + 1, stops.length - 1)];
  return [a[0]+(b[0]-a[0])*f, a[1]+(b[1]-a[1])*f, a[2]+(b[2]-a[2])*f];
}

function drawHeat() {
  if (!state.heat) return;
  const { nx, ny, vals, count, min, max, best, done } = state.heat;
  const span = max - min || 1;
  const cw = CW / nx, ch = CH / ny;
  ctx.globalAlpha = 0.6;
  for (let c = 0; c < count; c++) {
    const i = c % nx, j = (c / nx) | 0;
    const [r, g, b] = heatColor((vals[c] - min) / span);
    ctx.fillStyle = `rgb(${r|0},${g|0},${b|0})`;
    ctx.fillRect(i * cw, j * ch, cw + 1, ch + 1);
  }
  ctx.globalAlpha = 1;
  if (done && best) {
    const [bx, by] = toCanvas(best[0], best[1]);
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(bx, by, 16, 0, 2*Math.PI); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(bx-22, by); ctx.lineTo(bx+22, by);
    ctx.moveTo(bx, by-22); ctx.lineTo(bx, by+22); ctx.stroke();
  }
}

// --- entities ---------------------------------------------------------------
function drawPlayer(p, color, selected) {
  const [cx, cy] = toCanvas(p[0], p[1]);
  ctx.beginPath(); ctx.arc(cx, cy, PLAYER_R, 0, 2 * Math.PI);
  ctx.fillStyle = color; ctx.fill();
  ctx.lineWidth = selected ? 4 : 2;
  ctx.strokeStyle = selected ? "#ffd400" : "rgba(255,255,255,0.9)";
  ctx.stroke();
}
function drawBall() {
  const [cx, cy] = toCanvas(state.ball[0], state.ball[1]);
  ctx.beginPath(); ctx.arc(cx, cy, BALL_R, 0, 2 * Math.PI);
  ctx.fillStyle = "#fff"; ctx.fill();
  ctx.lineWidth = 2; ctx.strokeStyle = "#222"; ctx.stroke();
}

function render() {
  drawPitch();
  if (state.mode === "marginal") drawHeat();
  const sel = state.selected;
  state.red.forEach((p, i) => drawPlayer(p, "#e5484d", sel && sel.team === "red" && sel.idx === i));
  state.blue.forEach((p, i) => drawPlayer(p, "#1f6feb", sel && sel.team === "blue" && sel.idx === i));
  drawBall();
}

// --- value + probes readout -------------------------------------------------
function updateReadout() {
  const v = state.net.value(state.ball, state.blue, state.red);
  el("vbig").innerHTML = `${v >= 0 ? "+" : ""}${v.toFixed(3)}<small>blue next-goal prob ${(50*(v+1)).toFixed(1)}%</small>`;
  el("needle").style.left = `${50 * (v + 1)}%`;

  const pb = teamProbes(state.blue, 1, state.ball);
  const pr = teamProbes(state.red, -1, state.ball);
  const fmt = (x, d = 1) => x.toFixed(d);
  el("probes").innerHTML = `
    <table class="probe-table">
      <tr><th></th><th class="bl">Blue</th><th class="rd">Red</th></tr>
      <tr><td>Compactness (stretch)</td><td>${fmt(pb.stretch)}</td><td>${fmt(pr.stretch)}</td></tr>
      <tr><td>Width</td><td>${fmt(pb.width)}</td><td>${fmt(pr.width)}</td></tr>
      <tr><td>Depth</td><td>${fmt(pb.depth)}</td><td>${fmt(pr.depth)}</td></tr>
      <tr><td>Block area</td><td>${fmt(pb.area,0)}</td><td>${fmt(pr.area,0)}</td></tr>
      <tr><td>Line height</td><td>${fmt(pb.lineHeight)}</td><td>${fmt(pr.lineHeight)}</td></tr>
      <tr><td>Defenders goalside</td><td>${pb.goalside}</td><td>${pr.goalside}</td></tr>
      <tr><td>Centroid → ball</td><td>${fmt(pb.centroidToBall)}</td><td>${fmt(pr.centroidToBall)}</td></tr>
    </table>`;
}

// --- interaction ------------------------------------------------------------
function evtSim(e) {
  const r = canvas.getBoundingClientRect();
  const t = e.touches ? e.touches[0] : e;
  const cx = (t.clientX - r.left) / r.width * CW;
  const cy = (t.clientY - r.top) / r.height * CH;
  return { cx, cy, sim: toSim(cx, cy) };
}
function pick(cx, cy) {
  // ball first, then nearest player within radius
  const [bx, by] = toCanvas(state.ball[0], state.ball[1]);
  if (Math.hypot(cx - bx, cy - by) <= BALL_R + 6) return { kind: "ball" };
  for (const team of ["blue", "red"]) {
    const arr = state[team];
    for (let i = 0; i < arr.length; i++) {
      const [px, py] = toCanvas(arr[i][0], arr[i][1]);
      if (Math.hypot(cx - px, cy - py) <= PLAYER_R + 4) return { kind: "player", team, idx: i };
    }
  }
  return null;
}

function onDown(e) {
  e.preventDefault();
  const { cx, cy, sim } = evtSim(e);
  const hit = pick(cx, cy);
  if (!hit) return;
  canvas.classList.add("dragging");
  if (hit.kind === "ball") { state.dragging = { kind: "ball" }; return; }
  state.dragging = { kind: "player", team: hit.team, idx: hit.idx };
  if (state.mode === "marginal") {
    state.selected = { team: hit.team, idx: hit.idx };
    heatToken++;                        // cancel any in-flight sweep
    state.heat = null;                  // recompute on move-end
  }
}
function clampSim([x, y]) {
  return [Math.max(-SIM.L + 0.8, Math.min(SIM.L - 0.8, x)),
          Math.max(-SIM.W + 0.8, Math.min(SIM.W - 0.8, y))];
}
function onMove(e) {
  if (!state.dragging) return;
  e.preventDefault();
  const { sim } = evtSim(e);
  const p = clampSim(sim);
  if (state.dragging.kind === "ball") state.ball = p;
  else state[state.dragging.team][state.dragging.idx] = p;
  // live value is cheap; defer the (heavier) heatmap to drag-end
  render(); updateReadout();
}
function onUp() {
  if (!state.dragging) return;
  state.dragging = null;
  canvas.classList.remove("dragging");
  render(); updateReadout();
  if (state.mode === "marginal" && state.selected) { computeHeatAsync(); }
}

canvas.addEventListener("mousedown", onDown);
window.addEventListener("mousemove", onMove);
window.addEventListener("mouseup", onUp);
canvas.addEventListener("touchstart", onDown, { passive: false });
canvas.addEventListener("touchmove", onMove, { passive: false });
window.addEventListener("touchend", onUp);

// --- controls ---------------------------------------------------------------
function setMode(m) {
  state.mode = m;
  el("mode-value").classList.toggle("active", m === "value");
  el("mode-marginal").classList.toggle("active", m === "marginal");
  el("marginal-help").style.display = m === "marginal" ? "block" : "none";
  if (m === "value") { state.selected = null; state.heat = null; }
  render(); updateReadout();
}
el("mode-value").onclick = () => setMode("value");
el("mode-marginal").onclick = () => setMode("marginal");

function loadPreset(p) {
  state.ball = p.ball.slice();
  state.blue = p.blue.map((q) => q.slice());
  state.red = p.red.map((q) => q.slice());
  state.selected = null; state.heat = null;
  render(); updateReadout();
}

async function boot() {
  // Standalone single-file builds inject PP_WEIGHTS / PP_PRESETS so the demo
  // works from file:// (no fetch, no module loading). Otherwise fetch them.
  state.net = (typeof PP_WEIGHTS !== "undefined")
    ? new ValueNet(PP_WEIGHTS)
    : await ValueNet.fromURL("./data/weights.json");
  const presets = (typeof PP_PRESETS !== "undefined")
    ? PP_PRESETS.presets
    : (await (await fetch("./data/presets.json")).json()).presets;
  const sel = el("preset");
  presets.forEach((p, i) => {
    const o = document.createElement("option");
    o.value = i; o.textContent = p.name; sel.appendChild(o);
  });
  sel.onchange = () => loadPreset(presets[sel.value]);
  el("reset").onclick = () => loadPreset(presets[sel.value]);
  loadPreset(presets[0]);
  setMode("value");
}

boot().catch((err) => {
  el("vbig").textContent = "load error";
  console.error(err);
});
