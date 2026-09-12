#!/usr/bin/env python3
import csv
import statistics
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "results" / "performance.csv"
with path.open(newline="") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 112 or len({row["model_id"] for row in rows}) != 112:
    raise SystemExit("performance.csv must contain 112 unique models")

measurements = [
    ("sampling", "sample_evaluator_over_bridgestan"),
    ("density evaluation", "native_evaluator_over_bridgestan"),
]
for label, ratio_column in measurements:
    ratios = [float(row[ratio_column]) for row in rows]
    speedups = [1 / ratio for ratio in ratios]
    print(
        f"{label}: mean speedup={statistics.fmean(speedups):.3f}x; "
        f"median speedup={statistics.median(speedups):.3f}x; "
        f"density evaluator faster={sum(ratio < 1 for ratio in ratios)}/{len(ratios)}"
    )
