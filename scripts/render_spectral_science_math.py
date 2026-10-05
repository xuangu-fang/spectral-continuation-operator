#!/usr/bin/env python3
"""Render the scientific memo's equations without browser-side LaTeX support.

Edit docs/assets/spectral_science_math/equations.json, then run this script.
Use --check to validate links and detect stale or missing SVGs without writing.
Only requires the project's existing matplotlib dependency; no TeX or network.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
from matplotlib import mathtext, rc_context
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets/spectral_science_math"
SOURCE = ASSETS / "equations.json"


def render(equation: dict) -> bytes:
    """Make an accessible, font-independent, white-backed vector equation."""
    lines = equation.get("lines", [equation["latex"]])
    style = {
        "svg.hashsalt": "spectral-science-math-v1",
        "svg.fonttype": "path",
        "mathtext.fontset": "stix",
        "font.family": "DejaVu Sans",
        "font.size": 16,
    }
    with rc_context(style):
        prop = FontProperties(size=16)
        parser = mathtext.MathTextParser("path")
        metrics = [parser.parse(f"${line}$", dpi=72, prop=prop) for line in lines]
        pad, gap = 12, 12
        width = max(float(m.width) for m in metrics) + 2 * pad
        height = sum(float(m.height) for m in metrics) + 2 * pad + gap * (len(lines) - 1)
        fig = Figure(figsize=(width / 72, height / 72), dpi=72, facecolor="white")
        top = height - pad
        for line, measure in zip(lines, metrics):
            baseline = top - float(measure.height) + float(measure.depth)
            fig.text(
                pad / width, baseline / height, f"${line}$",
                fontsize=16, color="#14213d", ha="left", va="baseline",
            )
            top -= float(measure.height) + gap
        out = io.BytesIO()
        fig.savefig(
            out, format="svg", facecolor="white", edgecolor="none",
            metadata={
                "Date": None,
                "Title": equation["alt"],
                "Description": equation["latex"],
                "Creator": "render_spectral_science_math.py",
            },
        )
        # Matplotlib leaves spaces at the end of SVG path lines. Newlines still
        # separate path commands after normalization, with identical geometry.
        data = b"\n".join(line.rstrip() for line in out.getvalue().splitlines()) + b"\n"
    root = ET.fromstring(data)
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert not root.findall(".//{http://www.w3.org/2000/svg}script")
    return data


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--check", action="store_true", help="Validate without changing any files")
    args = cli.parse_args()
    spec = json.loads(SOURCE.read_text(encoding="utf-8"))
    assert spec["schema_version"] == 1
    document = (ROOT / spec["document"]).read_text(encoding="utf-8")
    equations = spec["equations"]
    ids = [equation["id"] for equation in equations]
    assert len(ids) == len(set(ids)), "Duplicate equation IDs"
    assert all(re.fullmatch(r"equation_\d{2}", ident) for ident in ids)
    assert "$" not in document, "Raw dollar-delimited math was reintroduced in the memo"
    assert not re.search(r"\\(?:operatorname|mathcal|mathbb|frac|begin)\b", document)
    linked = re.findall(r"assets/spectral_science_math/(equation_\d{2})\.svg", document)
    assert sorted(linked) == sorted(ids), "Equation links and source manifest differ"
    problems = []
    for equation in equations:
        path = ASSETS / f"{equation['id']}.svg"
        data = render(equation)
        if args.check:
            if not path.exists() or path.read_bytes() != data:
                problems.append(str(path.relative_to(ROOT)))
        else:
            path.write_bytes(data)
    if problems:
        print("Stale or missing formula images:\n" + "\n".join(problems), file=sys.stderr)
        print(f"Re-render with matplotlib {matplotlib.__version__} using this script.", file=sys.stderr)
        return 1
    print(f"{'Checked' if args.check else 'Rendered'} {len(equations)} SVG equations; "
          f"document links and math delimiters valid (matplotlib {matplotlib.__version__}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
