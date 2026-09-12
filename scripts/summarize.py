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
    (
        "sampling",
        "sample_kernel_over_bridgestan",
        "sample_bridgestan_median_s",
        "sample_kernel_median_s",
        1,
        "s",
    ),
    (
        "native evaluator",
        "native_kernel_over_bridgestan",
        "native_bridgestan_median_ns_per_eval",
        "native_kernel_median_ns_per_eval",
        1 / 1_000,
        "us/eval",
    ),
]
for label, ratio_column, baseline_column, kernel_column, scale, unit in measurements:
    ratios = [float(row[ratio_column]) for row in rows]
    speedups = [1 / ratio for ratio in ratios]
    baseline = statistics.fmean(float(row[baseline_column]) for row in rows) * scale
    kernel = statistics.fmean(float(row[kernel_column]) for row in rows) * scale
    print(
        f"{label}: mean time={baseline:.6g}->{kernel:.6g} {unit}; "
        f"mean speedup={statistics.fmean(speedups):.3f}x; "
        f"median speedup={statistics.median(speedups):.3f}x; "
        f"kernel faster={sum(ratio < 1 for ratio in ratios)}/{len(ratios)}"
    )
