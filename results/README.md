# Results

`performance.csv` is a saved-only harmonization of two completed measurement
sets:

- 102 rows from the frozen published-114 result;
- 10 later phase02 rows for the remaining non-ODE models.

No model was rerun to create this table. The original 102-model result remains
unchanged in the evaluation archive.

## Ratios

For each model, the sampling ratio is:

```text
median kernel sample elapsed / median BridgeStan sample elapsed
```

Each method has three matched repeats with fixed seeds. Runs used four chains,
400 warmup draws, and 1,000 retained draws per chain. Compile, attach, and the
pre-run kernel check are outside the sample timer. This differs from the
1,000-warmup protocol in the Fisher HMC paper.

The native evaluator ratio is:

```text
median kernel nanoseconds/evaluation / median BridgeStan nanoseconds/evaluation
```

Each method has three alternating blocks over saved checker points. This timer
covers log density plus the full gradient. It excludes process startup, binding,
and R conversion. Native and sampling ratios are not interchangeable.

A ratio below one favors the kernel. The median ratio in the main README is the
unweighted median of the 112 model-level ratios. Mean time averages the stored
model-level method medians. Mean speedup averages the 112 per-model
BridgeStan/kernel ratios, so a few large wins pull it upward.

## Provenance

The sanitized source tables used for the harmonization are in `source/`:

- `frozen-102.csv`
- `phase02-sampling.csv`
- `phase02-native.csv`

The frozen ratio columns are authoritative. Some displayed elapsed columns were
rounded, so dividing them does not reproduce every stored ratio exactly. The
phase02 ratios reproduce directly from their displayed method medians.

The corrected phase02 native run changed only stale manifest paths to already
staged data and points. Its ten earlier `no points` errors remain preserved in
the private evaluation archive. All corrected calls completed, and all 40
post-run reference-library, kernel-library, data, and point hashes were
unchanged.

## Limits

These are descriptive measurements from one machine and three repeats. They do
not certify convergence or explain why a model was faster or slower. Several
models retain matched diagnostic caveats. In particular, Basketball HMM0 and
ThreeMen2 showed poor behavior on both paths; Traffic, Diamonds, Butterfly, and
Survey also have saved caveats. Unfavorable results remain in the table.
