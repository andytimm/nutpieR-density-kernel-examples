#!/usr/bin/env python3
import csv
import statistics
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "results" / "performance.csv"
with path.open(newline="") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 112 or len({row["model_id"] for row in rows}) != 112:
    raise SystemExit("performance.csv must contain 112 unique models")

for label, column in [
    ("sampling", "sample_kernel_over_bridgestan"),
    ("native evaluator", "native_kernel_over_bridgestan"),
]:
    ratios = [float(row[column]) for row in rows]
    median = statistics.median(ratios)
    print(
        f"{label}: median kernel/BridgeStan={median:.6f}; "
        f"BridgeStan/kernel={1 / median:.3f}x; "
        f"kernel faster={sum(value < 1 for value in ratios)}/{len(ratios)}"
    )
