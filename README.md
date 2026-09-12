# nutpieR BYOK examples

Seyboldt, Carlson, and Carpenter's paper, ["Preconditioning Hamiltonian Monte Carlo by minimizing Fisher Divergence"](https://arxiv.org/abs/2603.18845), retained 114 models from [PosteriorDB](https://github.com/stan-dev/posteriordb). This repository has 112 handwritten Rust BYOK kernels for those models. Lotka-Volterra and SIR are omitted: matching their ODE solvers and sensitivities exactly was more work than we wanted for this experiment. We were a bit lazy.

| Measurement | Models | Median kernel / BridgeStan ratio (lower is better) | BridgeStan / kernel | Kernel faster |
| --- | ---: | ---: | ---: | ---: |
| End-to-end NUTS sampling | 112 | 0.208 | 4.80x | 100/112 |
| Log density + full gradient | 112 | 0.114 | 8.74x | 99/112 |

The reciprocal is the corresponding speedup. These are unweighted medians of model-level ratios from 3 repeats, not a universal speed claim. See [`results/performance.csv`](results/performance.csv) and [`results/README.md`](results/README.md) for the measurements and their limits.

## Repository layout

Each directory under [`models/`](models/) contains:

- `model.stan`, the exact PosteriorDB Stan source;
- `metadata/`, the corresponding PosteriorDB metadata;
- `kernel/`, the Rust source for the handwritten kernel.

PosteriorDB's data metadata does not specify licenses, so data is fetched from the pinned PosteriorDB revision rather than redistributed. The scripts live in [`scripts/`](scripts/). Generated binaries and Cargo targets, agent logs, and failed scratch work are left out of the repository. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the source and license details.

## Try one

The examples use the BYOK API in [`nutpieR`](https://github.com/andytimm/nutpieR). With it available in your R library, run:

```sh
python3 scripts/fetch-data.py models/mesquite-mesquite
Rscript scripts/check.R models/mesquite-mesquite --reference
```

The first command fetches the pinned data, verifies its hash, and writes the gitignored `data.json`. The second builds the example kernel, compiles the Stan reference, and checks the kernel against it.
