#!/usr/bin/env python3
import csv
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / "results" / "source"

def read(name):
    with (source / name).open(newline="") as handle:
        return list(csv.DictReader(handle))

def number(value):
    return float(value)

rows = []
for row in read("frozen-102.csv"):
    sample_ratio = number(row["sample_kernel_over_bridgestan"])
    native_ratio = number(row["native_kernel_over_bridgestan"])
    rows.append({
        "model_id": row["model_id"],
        "measurement_source": row["measurement_source"],
        "sample_bridgestan_median_s": row["bridgestan_median_sample_s"],
        "sample_kernel_median_s": row["kernel_median_sample_s"],
        "sample_kernel_over_bridgestan": row["sample_kernel_over_bridgestan"],
        "sample_bridgestan_over_kernel": 1 / sample_ratio,
        "sample_valid_pairs": int(float(row["sample_valid_pairs"])),
        "native_bridgestan_median_ns_per_eval": row["bridgestan_median_ns_per_eval"],
        "native_kernel_median_ns_per_eval": row["kernel_median_ns_per_eval"],
        "native_kernel_over_bridgestan": row["native_kernel_over_bridgestan"],
        "native_bridgestan_over_kernel": 1 / native_ratio,
        "native_blocks_per_method": int(float(row["native_blocks"])),
    })

sampling = {row["model_id"]: row for row in read("phase02-sampling.csv")}
native = {row["model_id"]: row for row in read("phase02-native.csv")}
if sampling.keys() != native.keys() or len(sampling) != 10:
    raise SystemExit("phase02 source tables must contain the same 10 models")
for model_id in sampling:
    sample = sampling[model_id]
    native_row = native[model_id]
    sample_ratio = number(sample["sample_kernel_over_bridgestan"])
    native_ratio = number(native_row["native_kernel_over_bridgestan"])
    rows.append({
        "model_id": model_id,
        "measurement_source": "phase02",
        "sample_bridgestan_median_s": sample["sample_elapsed_s_bridgestan"],
        "sample_kernel_median_s": sample["sample_elapsed_s_kernel"],
        "sample_kernel_over_bridgestan": sample["sample_kernel_over_bridgestan"],
        "sample_bridgestan_over_kernel": 1 / sample_ratio,
        "sample_valid_pairs": 3,
        "native_bridgestan_median_ns_per_eval": native_row["bridgestan"],
        "native_kernel_median_ns_per_eval": native_row["kernel"],
        "native_kernel_over_bridgestan": native_row["native_kernel_over_bridgestan"],
        "native_bridgestan_over_kernel": 1 / native_ratio,
        "native_blocks_per_method": 3,
    })

rows.sort(key=lambda row: row["model_id"])
if len(rows) != 112 or len({row["model_id"] for row in rows}) != 112:
    raise SystemExit("harmonized output must contain 112 unique models")

output = root / "results" / "performance.csv"
with output.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: f"{value:.12g}" if isinstance(value, float) else value for key, value in row.items()})
print(f"Wrote {len(rows)} rows to {output}")
