"""Build a single, self-contained web/standalone.html that runs from file://
(just double-click it) -- no server, no module loading, no fetch.

It inlines the CSS, concatenates the three JS modules (stripping import/export),
and embeds weights.json + presets.json as plain JS objects. Regenerate after
changing the demo or re-exporting weights:

    python tools/build_standalone.py
"""
from __future__ import annotations

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")


def read(*parts):
    with open(os.path.join(WEB, *parts)) as f:
        return f.read()


def strip_module_syntax(src: str, drop_imports=False) -> str:
    if drop_imports:
        src = "\n".join(l for l in src.splitlines() if not l.startswith("import "))
    src = re.sub(r"^export\s+", "", src, flags=re.MULTILINE)
    return src


def main():
    css = read("css", "style.css")
    model = strip_module_syntax(read("js", "model.js"))
    probes = strip_module_syntax(read("js", "probes.js"))
    demo = strip_module_syntax(read("js", "demo.js"), drop_imports=True)
    weights = read("data", "weights.json")
    presets = read("data", "presets.json")

    script = (
        f"var PP_WEIGHTS = {weights};\n"
        f"var PP_PRESETS = {presets};\n"
        f"{model}\n{probes}\n{demo}\n"
    )

    html = read("demo.html")
    html = html.replace(
        '<link rel="stylesheet" href="./css/style.css" />',
        f"<style>\n{css}\n</style>",
    )
    # neutralize nav links that only exist on the served site
    html = html.replace('href="./index.html"', 'href="https://github.com/microprediction/pitchperfect"')
    html = html.replace(
        '<script type="module" src="./js/demo.js"></script>',
        f"<script>\n{script}\n</script>",
    )
    html = html.replace(
        "<title>pitchperfect — interactive value function</title>",
        "<title>pitchperfect — interactive value function (standalone)</title>",
    )

    out = os.path.join(WEB, "standalone.html")
    with open(out, "w") as f:
        f.write(html)
    kb = os.path.getsize(out) / 1024
    print(f"wrote {out} ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
