# nutpieR BYOK examples

Seyboldt, Carlson, and Carpenter's paper, ["Preconditioning Hamiltonian Monte Carlo by minimizing Fisher Divergence"](https://arxiv.org/abs/2603.18845), retained 114 models from [PosteriorDB](https://github.com/stan-dev/posteriordb). This repository contains 112 handwritten Rust BYOK kernels for those models. Lotka-Volterra and SIR are omitted because matching their ODE solvers and sensitivities exactly was more work than we wanted. We were a bit lazy.

| Measurement | Mean time across models (BridgeStan → kernel) | Median kernel / BridgeStan ratio (lower is better) | Mean speedup | Kernel faster |
| --- | --- | ---: | ---: | ---: |
| End-to-end NUTS sampling | `5.25 s → 1.88 s` | `0.208` | `7.38x` | `100/112` |
| Log density + full gradient | `118 µs → 21.5 µs per evaluation` | `0.114` | `10.15x` | `99/112` |

Mean time and mean speedup are arithmetic averages across models. A few very large wins pull mean speedup upward, so the median ratio is the steadier typical-model summary. See [`results/performance.csv`](results/performance.csv) and [`results/README.md`](results/README.md) for the measurements, methods, and limits.

## Repository layout

Each directory in [`models/`](models/) contains:

- `model.stan`, the exact PosteriorDB Stan source;
- `metadata/`, the corresponding PosteriorDB metadata;
- `kernel/`, the Rust source for the handwritten kernel.

PosteriorDB's data metadata does not specify licenses, so the scripts fetch data from a pinned PosteriorDB revision instead of redistributing it. The scripts are in [`scripts/`](scripts/). Generated binaries, Cargo targets, agent logs, and scratch work are not tracked. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for source and license details.

## Try one

To try the `mesquite-mesquite` example, make [`nutpieR`](https://github.com/andytimm/nutpieR) available in R and run:

```sh
python3 scripts/fetch-data.py models/mesquite-mesquite
Rscript scripts/check.R models/mesquite-mesquite --reference
```

The first command fetches the pinned data, verifies its hash, and writes the gitignored `data.json`. The second builds the example kernel, compiles the Stan reference, and checks the kernel against it.
