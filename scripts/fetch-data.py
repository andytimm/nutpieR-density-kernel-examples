#!/usr/bin/env python3
import argparse
import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

REVISION = "5545a1dd07ae297c36edecbcd82aa49097b4c385"
URL = "https://raw.githubusercontent.com/stan-dev/posteriordb/{revision}/posterior_database/data/data/{name}.json.zip"

root = Path(__file__).resolve().parents[1]
with (root / "models" / "SOURCES.csv").open(newline="") as handle:
    sources = {row["model_id"]: row for row in csv.DictReader(handle)}

def fetch(model_dir):
    model_id = model_dir.name
    if model_id not in sources:
        raise SystemExit(f"Unknown model directory: {model_dir}")
    posterior = json.loads((model_dir / "metadata" / "posterior.json").read_text())
    data_name = posterior["data_name"]
    url = URL.format(revision=REVISION, name=data_name)
    print(f"Fetching {model_id} data from PosteriorDB")
    with urllib.request.urlopen(url) as response:
        archive = response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        members = [name for name in zipped.namelist() if name.endswith(".json")]
        if len(members) != 1:
            raise SystemExit(f"Expected one JSON file in {url}")
        value = json.loads(zipped.read(members[0]))
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    actual = hashlib.sha256(canonical).hexdigest()
    expected = sources[model_id]["data_json_canonical_sha256"]
    if actual != expected:
        raise SystemExit(f"Data hash mismatch for {model_id}: {actual} != {expected}")
    output = model_dir / "data.json"
    output.write_text(json.dumps(value, separators=(",", ":"), ensure_ascii=False))
    print(f"Wrote {output}")

parser = argparse.ArgumentParser(description="Fetch pinned PosteriorDB data for these examples")
parser.add_argument("model", nargs="?", help="model directory, for example models/mesquite-mesquite")
parser.add_argument("--all", action="store_true", help="fetch data for all 112 models")
args = parser.parse_args()
if args.all == bool(args.model):
    parser.error("supply one model directory or --all")
if args.all:
    for model_id in sorted(sources):
        fetch(root / "models" / model_id)
else:
    fetch(Path(args.model).resolve())
