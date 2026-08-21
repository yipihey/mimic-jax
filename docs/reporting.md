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

`ReportSection.body` optionally carries ordinary Markdown for equation-rich pedagogical sections. It is stored in the same machine-readable manifest and rendered before the section artifacts; it does not create a parallel report source or let presentation enter the physics kernels.

Pairwise comparison reports use a manifest with explicit baseline and candidate runs. Scalar comparisons retain both values, the absolute difference, and a fractional difference only when the baseline supplies a meaningful nonzero scale. A derivative prediction can be recorded alongside the measured finite change without conflating the two. `MultiModelComparisonReport` generalizes the same durable report boundary to three or more named models without choosing an arbitrary baseline; every metric must provide exactly one explicit value or availability state for every participating model.

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

The Mini-Millennium report deliberately separates expensive computation from presentation. The current reference uses two complementary gates: a zero-failure 1,000-tree control and a complete input-partition comparison that exposes rarer residuals. Put temporary products and the persistent compilation cache in a gitignored workspace:

```bash
mkdir -p archive/jax-cache

JAX_ENABLE_X64=1 JAX_COMPILATION_CACHE_DIR=archive/jax-cache \
    mimic_venv/bin/python scripts/check_mini_millennium_partition_equivalence.py \
    --tree-start 1500 --tree-count 1000 --member-binning power_of_two \
    --output archive/mini-millennium-equivalence-1000.json

JAX_ENABLE_X64=1 mimic_venv/bin/python \
    scripts/benchmark_mini_millennium_partition.py \
    --tree-start 1500 --tree-count 1000 --repeats 2 \
    --member-binning power_of_two --compilation-cache-dir archive/jax-cache \
    --output archive/mini-millennium-benchmark-1000.json
```

The science sample is every tree in input partition 1. `global-tree-offset=3432` preserves the run-wide `UniqueGalaxyID` encoding because partition 0 contains 3,432 trees. The field comparison retains every configured output snapshot; the population analysis retains z=0 summaries and uses one eighth of the Mini-Millennium volume, matching the plotting registry's file-fraction convention:

```bash
JAX_ENABLE_X64=1 JAX_COMPILATION_CACHE_DIR=archive/jax-cache \
    mimic_venv/bin/python scripts/check_mini_millennium_partition_equivalence.py \
    --trees simulations/mini-millennium/snapshots/trees_063.1 \
    --upstream output/sage16-mini-millennium/model_001.hdf5 \
    --tree-start 0 --tree-count 2864 --global-tree-offset 3432 \
    --member-binning power_of_two \
    --output archive/mini-millennium-equivalence-partition-1.json

JAX_ENABLE_X64=1 mimic_venv/bin/python \
    scripts/analyze_mini_millennium_partition.py \
    --compilation-cache-dir archive/jax-cache \
    --equivalence-json archive/mini-millennium-equivalence-partition-1.json \
    --output-json archive/mini-millennium-partition-1-science.json \
    --output-arrays archive/mini-millennium-partition-1-science.npz
```

Then build the report. The builder stages the durable products, renders the SMF and baryon-inventory figures from the NPZ arrays, saves its controlled response and refinement arrays, asks the existing model-local plotting registry for familiar figures, and writes Markdown plus JSON. It does not rerun Mini-Millennium:

```bash
JAX_ENABLE_X64=1 mimic_venv/bin/python \
    examples/build_mini_millennium_report.py \
    --equivalence-json archive/mini-millennium-equivalence-1000.json \
    --partition-equivalence-json archive/mini-millennium-equivalence-partition-1.json \
    --benchmark-json archive/mini-millennium-benchmark-1000.json \
    --science-json archive/mini-millennium-partition-1-science.json \
    --science-arrays archive/mini-millennium-partition-1-science.npz

mimic_venv/bin/python scripts/check_reports.py
```

On the measured Apple-arm64 CPU run, the 1,000-tree benchmark completed in 93.8 seconds for the first call and 4.77 seconds for the same-process warm call. The complete 2,864-tree artifact run completed in 147 seconds with a persistent compilation cache and used 7.54 GiB peak resident memory. These are scoped measurements, not universal performance claims.

The resulting `reports/mini-millennium-sage16-initial/index.md` opens directly in GitHub or Obsidian. `report.json` exposes stable status, diagnostic, observable, parameter, artifact, and provenance fields for programmatic queries. The population, response, and refinement NPZ products retain larger arrays and their scientific metadata.

The science-program report is built in independent stages so each expensive calculation releases its JAX compilation state and peak memory before the next begins. The complete-partition parameter response, 1,000-tree finite-difference audit, selected-history response, and 500-tree timestep study are durable inputs to a presentation-only builder:

```bash
JAX_ENABLE_X64=1 mimic_venv/bin/python \
    scripts/analyze_mini_millennium_science_program.py \
    --skip-history --skip-convergence --skip-finite-difference \
    --output-json archive/mini-millennium-sage16-parameter-responses.json \
    --output-arrays archive/mini-millennium-sage16-parameter-responses.npz

JAX_ENABLE_X64=1 mimic_venv/bin/python \
    scripts/analyze_mini_millennium_science_program.py \
    --tree-start 1500 --tree-count 1000 \
    --skip-history --skip-convergence \
    --output-json archive/mini-millennium-sage16-response-validation-1000.json \
    --output-arrays archive/mini-millennium-sage16-response-validation-1000.npz

JAX_ENABLE_X64=1 mimic_venv/bin/python \
    scripts/analyze_mini_millennium_history.py \
    --output-json archive/mini-millennium-sage16-history-responses.json \
    --output-arrays archive/mini-millennium-sage16-history-responses.npz

JAX_ENABLE_X64=1 mimic_venv/bin/python \
    scripts/validate_mini_millennium_history_responses.py \
    --output-json archive/mini-millennium-sage16-history-validation.json \
    --output-arrays archive/mini-millennium-sage16-history-validation.npz

JAX_ENABLE_X64=1 mimic_venv/bin/python \
    scripts/analyze_mini_millennium_convergence.py \
    --output-json archive/mini-millennium-sage16-convergence-500.json \
    --output-arrays archive/mini-millennium-sage16-convergence-500.npz

JAX_ENABLE_X64=1 mimic_venv/bin/python \
    scripts/analyze_mini_millennium_adaptive.py \
    --output-json archive/mini-millennium-sage16-adaptive-continuous.json \
    --output-arrays archive/mini-millennium-sage16-adaptive-continuous.npz

JAX_ENABLE_X64=1 mimic_venv/bin/python \
    examples/build_mini_millennium_science_report.py
```

The result is `reports/mini-millennium-sage16-science-program/index.md`. The hard-bin stellar mass function remains the upstream-equivalence observable. A labeled Gaussian-CDF finite-volume estimator is used only for population derivatives because the pathwise derivative of fixed hard-bin membership is zero almost everywhere. Historical responses use finite bins in `ln(a)` with redshift labels, and all report claims retain sample sizes and derivative-validation caveats.

The standalone galaxy-memory report is another two-stage product. The science command samples real fiducial trajectories, batches JAX Jacobians, validates a finite cooling pulse against the full nonlinear flow, and stores response/mode/map arrays. The presentation command reads only those durable products:

```bash
JAX_ENABLE_X64=1 mimic_venv/bin/python \
    scripts/analyze_sage16_linear_response.py \
    --tree-count 96 \
    --output-json archive/mini-millennium-sage16-linear-response.json \
    --output-arrays archive/mini-millennium-sage16-linear-response.npz

JAX_ENABLE_X64=1 mimic_venv/bin/python \
    examples/build_sage16_linear_response_report.py
```

The result is `reports/sage16-linear-response/index.md`. Its public narrative is organized around SAGE questions—galaxy memory, gas-supply filtering, reservoir participation, and AGN regulation—while the local frozen-coefficient assumptions, transfer function, poles, and hybrid-map limitations remain explicit underneath.

The differentiable-calibration report likewise separates an expensive, restartable science product from presentation. It uses the exact differentiable tree map, the real Baldry et al. SMF table, a fixed two-parameter emulator design, reserved exact SAGE validation points, and a familiar MCMC reference:

```bash
JAX_ENABLE_X64=1 JAX_COMPILATION_CACHE_DIR=archive/jax-cache \
    mimic_venv/bin/python scripts/analyze_sage16_differentiable_calibration.py \
    --output-json archive/mini-millennium-sage16-differentiable-calibration.json \
    --output-arrays archive/mini-millennium-sage16-differentiable-calibration.npz

JAX_ENABLE_X64=1 mimic_venv/bin/python \
    examples/build_sage16_differentiable_calibration_report.py
```

The result is `reports/sage16-differentiable-calibration/index.md`. Its failed surrogate gate and unavailable final parameter intervals are intentional report outputs, not build failures. The exact evaluated SAGE improvement, local curvature forecast, emulator training/validation arrays, and MCMC diagnostic remain available for inspection.

For ordinary Python use, construct `RunReport`, `ComparisonReport`, or `MultiModelComparisonReport` from canonical result summaries and call `write_report(report, directory)`. `parameter_response_diagnostic`, `timestep_refinement_diagnostic`, `conservation_diagnostic`, and `benchmark_diagnostic` are adapters: they summarize existing objects and never rerun the science. `capture_provenance` records the repository state, explicit configurations and input checksums, software, hardware/backend, command, and upstream run record.

Comparison metrics should normally be constructed with `ComparisonMetric.from_values(...)`. It records baseline, candidate, and absolute difference, and defines a fractional difference only for a meaningful nonzero baseline. `derivative_prediction` is a separate optional fractional prediction, so a local elasticity is never confused with the measured finite run-to-run change.

## Publishing

Install Quarto locally only if a browser preview is useful, then render the already generated artifacts:

```bash
quarto render
```

The static site is written to the gitignored `_site/` directory. Quarto executes no scientific code. The GitHub Pages workflow performs the same lightweight validation and render, uploads `_site`, and deploys it with GitHub's Pages actions. A repository administrator must select **GitHub Actions** as the Pages source once under **Settings → Pages**.

Publishing a report is deliberate: review its health states, scientific scope, provenance, and asset sizes, then commit the selected report directory. Temporary reports remain outside `reports/`; raw catalogues stay under the gitignored run-output area. If a future canonical array is too large for ordinary git, publish it as a release or in scientific storage and keep a checksummed reference in the report rather than turning the repository into a binary run database.
