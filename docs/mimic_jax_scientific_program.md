# First mimic-jax Scientific Application Program

The first application program asks familiar SAGE questions with direct fractional responses. Its audience is MIMIC and SAM practitioners, and its opening result must be numerical equivalence on familiar Mini-Millennium outputs rather than unfamiliar mathematical terminology.

## Readiness gate

Applications begin only after the complete fiducial SAGE16 pipeline, tree inheritance, shared-central ordering, and merger event maps reproduce the relevant upstream Mini-Millennium catalog fields and histories at recorded tolerances. Process-level C oracle agreement for the initial quiescent slice is necessary evidence but does not discharge this gate.

Figure 1 reproduces upstream selections, units, bins, and definitions for the existing Mini-Millennium example plots and reports quantitative residuals.

## First application: Does the numerical timestep matter?

Before population sensitivities, the first application after equivalence quantifies the reference SAGE16 sequential update itself. It refines internal substeps at fixed tree forcing; reports reservoir, SFR, metallicity, BH, and quenched-state convergence; measures baryon/metal residuals and positivity; and maps the finite-step-to-fastest-timescale ratio over halo mass and redshift. Tree-forcing interpolation and baryonic integration resolution are varied separately.

Only prescriptions with an explicit continuous-rate interpretation are compared with a higher-order fixed-step method. Accuracy is compared at matched right-hand-side evaluations, wall-clock cost, and target error, with compilation separated from warmed execution. Numerical shifts in familiar Mini-Millennium statistics are compared directly with fractional parameter responses so numerical error is not confused with parameter or model uncertainty. See [`numerical_integration.md`](numerical_integration.md).

## Application A: Which observations constrain which SAGE physics?

The first completed application augments the familiar z=0 stellar mass function with `% abundance change per 1% parameter change`. Selected automatic responses are validated with symmetric finite differences across several step sizes. Standard SAGE calibration observables are then assembled into an observable/bin-by-parameter fractional response matrix. A companion response-similarity matrix separates parameter influence from observational identifiability.

Hard mass bins are not naively differentiable with respect to galaxy masses. The implementation must establish a population-response estimator whose relation to the upstream hard-bin stellar mass function is quantitatively validated, including empty-bin handling, bin-edge behavior, resolution dependence, and finite-difference agreement. Any smoothing belongs to the summary-statistic estimator, never to the faithful SAGE physics, and must be labeled as such.

## Application B: When does each baryonic process matter?

For galaxies grouped by present-day stellar or halo mass, finite epochs in `ln(a)` receive named log-rate perturbations. Signed maps of `d ln O(z=0) / d epsilon_i,k` are shown with redshift labels for cooling, SN reheating/ejection, reincorporation, and AGN heating after each channel has a faithful implementation. The plain-language question is: which process matters, for which galaxies, and during which epoch?

Normalized magnitude weights may summarize intervals containing a chosen fraction of total sensitivity, but they are always shown alongside the signed response.

## Application C: Where does AGN regulation create the massive-galaxy cutoff?

Paired cooling and AGN historical-response panels quantify the transition between cooling-dominated and AGN-regulated behavior as a function of final mass and redshift. Candidate outputs include final stellar mass, SFR, quenched fraction, and massive-galaxy abundance. The transition is measured from SAGE16 rather than assumed.

## Intended figure sequence

1. Familiar Mini-Millennium universe: upstream MIMIC versus mimic-jax with numerical residuals.
2. Timestep convergence for representative histories and final observables.
3. Conservation and positivity residuals versus timestep and method.
4. Accuracy versus cost for the upstream reference and legitimate alternative integrators.
5. Mass/redshift map of timestep resolution and convergence error.
6. Numerical impact on familiar Mini-Millennium statistics.
7. Familiar stellar mass function plus fractional parameter responses.
8. Observable/bin-by-parameter fractional response matrix.
9. Parameter response-similarity matrix.
10. Cooling, SN, reincorporation, and AGN historical-response maps.
11. Direct cooling-versus-AGN transition map.

The later mathematical interpretation in terms of tangent models, reverse-mode propagation, and response operators belongs after these practitioner-facing results, not before them.
