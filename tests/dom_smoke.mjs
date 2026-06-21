// Headless smoke test for web/js/demo.js: stub just enough DOM + canvas + fetch
// to boot the demo, read a value, and run a full async marginal sweep. Verifies
// the wiring (model load, readout, controls, progressive heatmap) end-to-end.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const web = join(here, "..", "web");

const noop = () => {};
const ctxProxy = new Proxy({}, { get: () => noop, set: () => true });

const listeners = {};            // type -> handler (last wins, fine for test)
function mkEl(extra = {}) {
  return {
    style: {}, textContent: "", innerHTML: "",
    classList: { add: noop, remove: noop, toggle: noop, contains: () => false },
    addEventListener: (t, h) => { listeners[t] = h; },
    appendChild: noop, onclick: null, onchange: null, value: "0",
    getContext: () => ctxProxy,
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 1000, height: 600 }),
    width: 0, height: 0, ...extra,
  };
}
const elements = {};
const ids = ["pitch", "vbig", "needle", "probes", "mode-value", "mode-marginal",
             "marginal-help", "selinfo", "preset", "reset"];
for (const id of ids) elements[id] = mkEl();

globalThis.document = {
  getElementById: (id) => elements[id],
  createElement: () => ({ value: "", textContent: "" }),
};
globalThis.window = { addEventListener: (t, h) => { listeners[t] = h; } };
globalThis.performance = { now: () => Date.now() };
globalThis.requestAnimationFrame = (cb) => setTimeout(() => cb(Date.now()), 0);
globalThis.fetch = async (url) => {
  const file = join(web, url.replace(/^\.\//, ""));
  const text = readFileSync(file, "utf8");
  return { json: async () => JSON.parse(text), ok: true };
};

await import("../web/js/demo.js");

// wait for boot() to finish loading weights + presets and render once
async function waitFor(pred, ms = 8000) {
  const t0 = Date.now();
  while (!pred()) {
    if (Date.now() - t0 > ms) throw new Error("timeout waiting for condition");
    await new Promise((r) => setTimeout(r, 25));
  }
}
await waitFor(() => elements.vbig.innerHTML.includes("blue next-goal"));
if (elements.vbig.textContent === "load error") throw new Error("demo failed to boot");
console.log("boot ok    vbig:", elements.vbig.innerHTML.replace(/<[^>]+>/g, " ").trim());
console.log("probes ok :", elements.probes.innerHTML.includes("Stretch"));

// switch to marginal mode and simulate grabbing + releasing a player
elements["mode-marginal"].onclick();
const down = listeners["mousedown"], up = listeners["mouseup"];
// blue[0] (GK) of preset 0 is at sim (-46,0) -> canvas; rect is 1000x600 == CW/CH
const cx = (-46 + 50) / 100 * 1000, cy = (30 - 0) / 60 * 600;
down({ preventDefault: noop, clientX: cx, clientY: cy });
up({ preventDefault: noop });
await waitFor(() => elements.selinfo.textContent.includes("best spot"), 20000);
console.log("marginal ok:", elements.selinfo.textContent);
console.log("ALL GOOD");
