# Third-party material

The Stan models and metadata under `models/` come from
[PosteriorDB](https://github.com/stan-dev/posteriordb), pinned at commit
`5545a1dd07ae297c36edecbcd82aa49097b4c385`.

PosteriorDB and 111 of the included Stan models use the BSD 3-Clause license.
See `licenses/POSTERIORDB-BSD-3-Clause.txt`. The Prophet Stan model used by
`rstan_downloads-prophet` is marked MIT in its PosteriorDB metadata. See
`licenses/PROPHET-MIT.txt`.

The included PosteriorDB data metadata does not state a license. Data files are
therefore not redistributed here. `scripts/fetch-data.py` downloads them from
the pinned upstream revision and checks their content hashes. Each model
folder retains the upstream posterior, model, and data metadata, including its
titles, references, URLs, and available license fields.

The custom density kernels written in Rust and the repository scripts are available under the root
MIT license.
