---
title: "mimic-jax scientific run reports"
description: "Self-describing SAM runs, numerical diagnostics, and physically interpretable responses"
toc: false
---

# Understand the run before interpreting the science

Mimic-jax reports connect familiar semi-analytic-model results to the evidence needed to trust and interrogate them. Each published run records what was evolved, which diagnostics were actually evaluated, where its scientific arrays live, and how to reproduce it.

## Published reference reports

- [Can SAGE16 and SHARK be compared without hidden conventions?](reports/sage16-shark-interoperability-audit/index.md) — a model-neutral catalogue and observable contract, native-output comparison diagnostics, an explicit observation registry, and a measured tree-portability matrix that separates field coverage from topology-driver readiness.
- [SHARK Lagos23 on the same testable footing as SAGE16](reports/shark-continuous-foundation/index.md) — a pinned native population, exact 19-state disk/starburst assembly, augmented BH/AGN memory, explicit hybrid events, controlled interval and burst oracles, common SAGE-facing and SHARK-only observables, convergence, fractional responses, and an explicit record of the population-equivalence work that remains.
- [Fit SAGE with gradients: what one stellar mass function can—and cannot—constrain](reports/sage16-differentiable-calibration/index.md) — a real Baldry et al. stellar-mass-function fit showing the useful direction and local curvature from JAX responses, exact SAGE validation, a familiar MCMC comparison, and a deliberately rejected first emulator/error-bar claim.
- [How much of SAGE can we remove?](reports/sage16-minimal-model/index.md) — a held-out teacher--student experiment showing that four evolving states recover the broad z=0 stellar-mass prediction within a predeclared 30% contract, while cold gas, SFR, fine-bin structure, and quenching expose exactly where the reduction fails.
- [How long does SAGE remember?](reports/sage16-linear-response/index.md) — a science-led introduction to local galaxy response times, gas-supply filtering, coupled baryon-cycle modes, reservoir participation, mass/redshift memory, and the dynamical effect of stored AGN heating.
- [What controls galaxies in SAGE16?](reports/mini-millennium-sage16-science-program/index.md) — the complete-partition stellar mass function, its seven-parameter fractional response, an observable–parameter response matrix, parameter similarities, finite-epoch baryon-cycle responses, a cooling/AGN comparison, halo–stellar growth histories, population timestep refinement, and Jacobian-aware adaptive convergence of the separated continuous flows.
- [SAGE16 Mini-Millennium: from equivalence to baryon-cycle insight](reports/mini-millennium-sage16-initial/index.md) — a complete input-partition stellar mass function, explicit FoF baryon inventory, quantified object-level residuals, larger-sample performance, controlled derivative evidence, and complete provenance.

## How to read a report

The opening health table distinguishes passed, warning, failed, and not evaluated checks. Familiar model plots come first. Numerical diagnostics and physically interpreted derivatives follow, with array products linked rather than embedded in the manifest. The final provenance section records the code, command, configurations, inputs, software, and hardware.

Every report is also available as ordinary Markdown for GitHub or Obsidian and as a compact `report.json` manifest for agents and future MCP tools. See the [report architecture](docs/reporting.md) for the stable contract.
