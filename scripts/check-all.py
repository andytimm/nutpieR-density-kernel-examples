#!/usr/bin/env python3
import os
import subprocess
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
models = sorted(path for path in (root / "models").iterdir() if path.is_dir())
with tempfile.TemporaryDirectory(prefix="nutpier-byold-cargo-check-") as target:
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = target
    for number, model in enumerate(models, 1):
        manifest = model / "evaluator" / "Cargo.toml"
        print(f"[{number:3d}/{len(models)}] {model.name}", flush=True)
        result = subprocess.run(
            ["cargo", "check", "--release", "--locked", "--manifest-path", str(manifest)],
            env=env,
        )
        if result.returncode:
            raise SystemExit(result.returncode)
print(f"Checked {len(models)} density evaluator crates.")
