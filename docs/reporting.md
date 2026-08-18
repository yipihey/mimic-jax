# Self-Documenting Run Reports

Mimic-jax run reports turn existing scientific results and diagnostics into durable, shareable artifacts. The reporting layer summarizes canonical outputs; it does not recompute physics, own numerical arrays, or enter a JAX kernel.

## Design

The data flow is:

```text
physics and tree evolution
  -> canonical result objects and saved scientific arrays
  -> typed report manifest
  -> Markdown + JSON
  -> optional Quarto static site
```

Existing objects remain authoritative. `PartitionEvolutionResult` owns evolved catalog records, `ParameterResponseMatrix` and `HistoricalProcessResponse` own differentiable responses, `TimestepRefinementResult` owns convergence products, MIMIC's run-local `metadata/` and HDF5 `RunProperties` own upstream provenance, benchmark JSON owns timing measurements, and the model-local plot registry owns familiar SAGE figures. A report records compact summaries and relative references to those products rather than introducing a second catalogue or array format.

The report manifest has five stable concepts:

- **Run identity**: model, tree data, parameter set, integration method, and a concise scientific summary.
- **Health checks**: explicitly `passed`, `warning`, `failed`, or `not_evaluated`. A missing test is never rendered as a pass.
- **Sections and diagnostics**: extensible scientific summaries with scalar metrics, physical interpretations, and links to existing figures or array products.
- **Artifacts**: relative, checksum-capable references to figures, configurations, logs, JSON, and compressed numerical arrays.
- **Provenance**: git state, command, configuration and input checksums, software versions, backend/hardware, seeds, and any upstream MIMIC version record.

Comparison reports use a separate manifest with explicit baseline and candidate runs. Scalar comparisons retain both values, the absolute difference, and a fractional difference only when the baseline supplies a meaningful nonzero scale. A derivative prediction can be recorded alongside the measured finite change without conflating the two.

## Durable outputs

Every published report directory contains at least:

```text
index.md       human-readable canonical report
report.json    compact machine-readable manifest
assets/        selected figures and referenced small data products
```

The Markdown uses ordinary headings, tables, image links, fenced code, and YAML front matter. It remains readable on GitHub and in Obsidian without Quarto. Paths are relative to the report directory, so a report can be moved or shared as one directory.

Large catalogues and response tensors do not belong in `report.json`. They stay in HDF5, NPZ, or another appropriate scientific format and appear as checksummed artifact references. Canonical reports worth publishing live under `reports/`; temporary reports and large raw run products belong outside git unless deliberately promoted. Git LFS, a release asset, or external scientific storage can be added later when a selected artifact is too large for ordinary git.

## Static web presentation

Quarto renders the committed Markdown into a lightweight site. Scientific computation is never part of the site build: the GitHub Pages workflow checks out already generated report artifacts, installs Quarto, renders static HTML, and deploys the result. This follows the separation recommended in the [Quarto GitHub Pages guide](https://quarto.org/docs/publishing/github-pages.html) and GitHub's [custom Pages workflow](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

The site landing page shows only a small run overview, health summary, and selected headline figures. Detailed science, numerical diagnostics, sensitivities, and provenance remain in linked report sections. Existing pages for cooling, feedback, sensitivity, conservation, and numerical integration remain the conceptual source of truth and can be linked rather than duplicated.

## Scientific language

Reports lead with familiar SAGE observables, then numerical trust diagnostics, then differentiable diagnostics. A logarithmic parameter response is described as “percentage change in the observable per 1% parameter change.” A finite-epoch process response is described as “percentage change in today's observable caused by a 1% change in this process during this epoch.” The rigorous normalization and derivative method remain visible in metadata.

No population-equivalence, conservation, convergence, gradient, or performance status is inferred merely because a corresponding API exists. Each status must cite an evaluated diagnostic and its tolerance or remain `not_evaluated`.

## Extension boundary

New diagnostics integrate by producing a diagnostic summary and, where necessary, a referenced scientific artifact. Markdown, JSON, and site-rendering logic consume the report manifest generically. Physics functions never import the reporting package, and plotting functions continue to return their established `(plot_path, skip_message)` result.

## Practitioner workflow

The initial Mini-Millennium report deliberately separates expensive computation from presentation. First create machine-readable benchmark and equivalence products in the gitignored `benchmarks/` workspace:

```bash
mkdir -p benchmarks

JAX_ENABLE_X64=1 mimic_venv/bin/python \
    scripts/check_mini_millennium_partition_equivalence.py \
    --tree-start 1500 --tree-count 100 --member-binning=power_of_two \
    --output benchmarks/mini-millennium-equivalence.json

JAX_ENABLE_X64=1 mimic_venv/bin/python \
    scripts/benchmark_mini_millennium_partition.py \
    --tree-start 1500 --tree-count 100 --repeats 2 \
    --output benchmarks/mini-millennium-benchmark.json
```

Then build the report. The builder stages the two JSON products, saves its controlled response and refinement arrays, asks the existing model-local plotting registry for familiar figures, and writes Markdown plus JSON:

```bash
JAX_ENABLE_X64=1 mimic_venv/bin/python \
    examples/build_mini_millennium_report.py \
    --equivalence-json benchmarks/mini-millennium-equivalence.json \
    --benchmark-json benchmarks/mini-millennium-benchmark.json

mimic_venv/bin/python scripts/check_reports.py
```

The resulting `reports/mini-millennium-sage16-initial/index.md` opens directly in GitHub or Obsidian. `report.json` exposes stable status, diagnostic, observable, parameter, artifact, and provenance fields for programmatic queries. The `.npz` response and refinement products retain the larger arrays and their scientific metadata.

For ordinary Python use, construct `RunReport` or `ComparisonReport` from canonical result summaries and call `write_report(report, directory)`. `parameter_response_diagnostic`, `timestep_refinement_diagnostic`, `conservation_diagnostic`, and `benchmark_diagnostic` are adapters: they summarize existing objects and never rerun the science. `capture_provenance` records the repository state, explicit configurations and input checksums, software, hardware/backend, command, and upstream MIMIC run record.

Comparison metrics should normally be constructed with `ComparisonMetric.from_values(...)`. It records baseline, candidate, and absolute difference, and defines a fractional difference only for a meaningful nonzero baseline. `derivative_prediction` is a separate optional fractional prediction, so a local elasticity is never confused with the measured finite run-to-run change.

## Publishing

Install Quarto locally only if a browser preview is useful, then render the already generated artifacts:

```bash
quarto render
```

The static site is written to the gitignored `_site/` directory. Quarto executes no scientific code. The GitHub Pages workflow performs the same lightweight validation and render, uploads `_site`, and deploys it with GitHub's Pages actions. A repository administrator must select **GitHub Actions** as the Pages source once under **Settings → Pages**.

Publishing a report is deliberate: review its health states, scientific scope, provenance, and asset sizes, then commit the selected report directory. Temporary reports remain outside `reports/`; raw catalogues stay under the gitignored run-output area. If a future canonical array is too large for ordinary git, publish it as a release or in scientific storage and keep a checksummed reference in the report rather than turning the repository into a binary run database.
