# Differentiable SAGE16 calibration

The first calibration application asks a familiar question: can SAGE16 move closer to the observed low-redshift stellar mass function, and what extra information does a differentiated model provide compared with perturb-and-rerun fitting?

The current answer is deliberately split into what has been demonstrated and what remains unavailable.

## Current demonstrated workflow

1. Evolve the exact upstream-sequential SAGE16 tree map at the fiducial parameters.
2. Carry named parameter tangents through the same tree to obtain fractional stellar-mass-function responses.
3. Build a local Gaussian working likelihood in `ln(phi)` using the Baldry, Glazebrook & Driver (2008) range plus fixed Mini-Millennium Poisson counting variance.
4. Use the response to propose a parameter move in `ln(theta)`.
5. Rerun exact SAGE at the proposed point; never accept the derivative as a finite-change prediction without this check.
6. Build a small response-surface emulator only when repeated likelihood calls justify it.
7. Reserve exact SAGE points for validation and withhold error bars if the emulator misses its declared tolerance.

The durable scientific result and its limitations are in [Fit SAGE with gradients](../reports/sage16-differentiable-calibration/index.md).

## Why logarithmic parameters and observables?

For positive SAGE parameters and observables, the natural local quantity is

\[
E_{\alpha i}
=
\frac{\partial\ln O_\alpha}{\partial\ln\theta_i}.
\]

It has the practitioner-facing interpretation “percent change in this observable per 1% parameter change.” A local response emulator therefore uses

\[
\ln O(q)
\simeq
\ln O_0 + E q,
\qquad
q_i=\ln(\theta_i/\theta_{i,0}).
\]

This guarantees positive predictions and gives every coefficient a direct physical meaning. It does **not** make finite parameter moves linear; SAGE thresholds, event branches, and hard population bins must still be checked with full reruns.

The reusable implementation lives in [`mimic_jax/inference.py`](../mimic_jax/inference.py). It supplies:

- an explicit Gaussian likelihood in log-observable space;
- a local elasticity emulator;
- a maximum-likelihood/local-Laplace solve that refuses rank-deficient fits rather than inventing a prior;
- finite-difference validation of the likelihood gradient; and
- a deliberately simple random-walk Metropolis reference for low-dimensional checks.

## Reproduce the application

The expensive science calculation is separate from report rendering:

```bash
JAX_COMPILATION_CACHE_DIR=archive/jax-cache \
  .venv/bin/python scripts/analyze_sage16_differentiable_calibration.py \
  --output-json archive/mini-millennium-sage16-differentiable-calibration.json \
  --output-arrays archive/mini-millennium-sage16-differentiable-calibration.npz
```

Then build the durable Markdown/JSON/SVG report without rerunning SAGE:

```bash
.venv/bin/python examples/build_sage16_differentiable_calibration_report.py
```

Quarto renders the generated Markdown as part of the existing static site. Obsidian and GitHub can read the same `index.md` directly.

## What the first run established

The exact tested SAGE point improves the stated working chi-square substantially. The initial derivative-only move also points in the useful direction. However:

- the local response does not predict every finite hard-bin change within 0.05 dex;
- the first 3×3 quadratic emulator also fails that held-out hard-bin tolerance;
- its optimum reaches the edge of the measured reincorporation domain; and
- the observational covariance is only a diagonal working approximation.

Consequently the report publishes a local curvature forecast but marks final observational parameter intervals unavailable. The failure products remain in the archive and report rather than being discarded.

## MCMC and gradients are complementary

Automatic differentiation does not replace posterior sampling. It provides information that conventional black-box MCMC leaves unused:

- a direction to the mode;
- local curvature and degeneracy directions;
- efficient gradient-informed proposals;
- a derivative constraint for a sample-efficient emulator; and
- an immediate test of whether the selected observations identify the chosen parameters.

MCMC remains appropriate for non-Gaussian, bounded, multimodal, or threshold-crossing posteriors—after the forward model or emulator has been validated over the sampled region.

## Next scientific extension

The stellar mass function alone should not be asked to constrain every SAGE parameter. The next joint likelihood should add real SAGE16 calibration observables with complementary physics, beginning with the baryonic Tully–Fisher relation, mass–metallicity relation, and black-hole–bulge relation. Each observable needs an explicit selection, unit conversion, covariance treatment, and differentiable population estimator.

The complete adaptive continuous/hybrid tree driver is also not yet validated through every event and projection. Until it is, observational fitting uses the exact differentiable upstream-sequential map so numerical reformulation is not confused with parameter calibration.
