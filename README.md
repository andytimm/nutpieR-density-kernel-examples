# nutpieR Bring Your Own Density Kernel examples

Seyboldt, Carlson, and Carpenter's paper, ["Preconditioning Hamiltonian Monte Carlo by minimizing Fisher Divergence"](https://arxiv.org/abs/2603.18845), retained 114 models from [PosteriorDB](https://github.com/stan-dev/posteriordb). This repository tests Bring Your Own Density Kernel in nutpieR with 112 handwritten Rust custom density kernels. Each kernel returns the unconstrained log density and full gradient rather than using BridgeStan; both paths use the same nuts-rs NUTS sampler and settings.

Lotka-Volterra and SIR are omitted because exact matching of their ODE solvers and sensitivities was outside this experiment.

| Measurement | Mean speedup (BridgeStan / custom kernel) | Median speedup (BridgeStan / custom kernel) | Custom kernel faster |
| --- | ---: | ---: | ---: |
| NUTS sampling wall time | `7.38x` | `4.80x` | `100/112` |
| Log density + full gradient | `10.15x` | `8.74x` | `99/112` |

Speedup is calculated for each model before taking the mean or median. A few very large wins pull the mean upward; the median is less affected by those outliers. See [`results/performance.csv`](results/performance.csv) and [`results/README.md`](results/README.md) for methods and limits.

## Repository layout

Each directory in [`models/`](models/) contains:

- `model.stan`, the exact PosteriorDB Stan source;
- `metadata/`, the corresponding PosteriorDB metadata;
- `kernel/`, the Rust source for the handwritten density kernel.

PosteriorDB's data metadata does not specify licenses, so the scripts fetch data from a pinned PosteriorDB revision instead of redistributing it. The scripts are in [`scripts/`](scripts/). Generated binaries, Cargo targets, agent logs, and scratch work are not tracked. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for source and license details.

## Try one

To try the `mesquite-mesquite` example, make [`nutpieR`](https://github.com/andytimm/nutpieR) available in R and run:

```sh
python3 scripts/fetch-data.py models/mesquite-mesquite
Rscript scripts/check.R models/mesquite-mesquite --reference
```

The first command fetches the pinned data, verifies its hash, and writes the gitignored `data.json`. The second builds the custom density kernel, compiles the Stan reference, and checks the kernel against it.
