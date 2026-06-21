// Run the JS ValueNet over the reference states and print JSON: [{logit, v}].
// Used by tests/test_parity.py to compare JS against numpy and torch.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { ValueNet } from "../web/js/model.js";

const here = dirname(fileURLToPath(import.meta.url));
const data = join(here, "..", "pitchperfect", "data");
const weights = JSON.parse(readFileSync(join(data, "weights.json"), "utf8"));
const refs = JSON.parse(readFileSync(join(data, "refs.json"), "utf8"));

const net = new ValueNet(weights);
const out = refs.cases.map((c) => {
  const s = c.state;
  return { logit: net.logit(s.ball, s.blue, s.red), v: net.value(s.ball, s.blue, s.red) };
});
process.stdout.write(JSON.stringify(out));
