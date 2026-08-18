# Fiducial SAGE16 Disk Radius and Merger Clock

These two pre-timestep modules prepare structural and event state before any baryonic substep runs. Neither is a reservoir transfer, so mimic-jax exposes each as an immutable state update with branch diagnostics rather than pretending it is an ODE rate.

## Disk scale radius

The upstream source is [`sage_set_disk_scale_radius.c`](../models/sage16/modules/sage_set_disk_scale_radius/sage_set_disk_scale_radius.c). Only a Type-0 FoF central is updated. Type-1 and Type-2 satellites retain the radius inherited from the last time they were central, so recomputing every galaxy from its current halo would alter its star-formation threshold.

For a valid virial velocity and radius, upstream calculates

`|J| = sqrt(Jx^2 + Jy^2 + Jz^2)`, `lambda = |J| / (1.414 Vvir Rvir)`, and `Rd = (lambda / 1.414) Rvir`.

The truncated `1.414` literal is intentional. Upstream also narrows `Vvir` and `Rvir` from their double halo storage to float arguments before this calculation, computes the expression in double precision, and writes the result to the float `DiskScaleRadius` field. mimic-jax preserves all three details. If either narrowed virial quantity is at most `1e-10`, SAGE uses the float calculation `0.1f * Rvir`.

This update moves no baryons or metals and therefore leaves both conservation ledgers unchanged. Within the active Type-0, positive-radius branch it is smooth in the halo spin components and virial quantities, except at zero spin. Halo type and the virial fallback are discrete/thresholded choices.

## Merger-clock initialization

The upstream source is [`sage_initialise_merger_clock.c`](../models/sage16/modules/sage_initialise_merger_clock/sage_initialise_merger_clock.c), with target rules in [`central_link.h`](../models/sage16/shared/central_link.h). The float `MergTime` field follows an explicit sentinel protocol:

- `999.9f` means unset;
- a Type-0 promotion resets any value below `999.0` to that unset sentinel;
- an unset Type-2 orphan is assigned `0.0f` for immediate resolution;
- an already initialized Type-1/2 clock is retained;
- a computed time at or above `999.0` is capped to `998.0`;
- an under-resolved or zero-mass Type-1 satellite receives `-1.0`.

For a resolved Type-1 satellite with at least ten particles, SAGE calculates

`t_merge = 2 * 1.17 * Rvir,c^2 * Vvir,c / [ln(1 + N_c/N_s) G (Mvir,s + Mstar,s + Mcold,s)]`.

The central virial quantities and satellite virial mass are double precision; stellar and cold masses are promoted from their float reservoir storage; the final clock is narrowed to float. `initialise_merger_clocks` accepts a batched FoF group, derives the first Type-0 central exactly as the upstream scan does, and returns the updated immutable group plus masks identifying resets, initialized clocks, immediate orphans, and resolved target indices. A group with no Type-0 central is unchanged.

The clock calculation moves no mass or metals. Its formula is smooth on a fixed resolved branch, but particle-count resolution, type/sentinel handling, target identity, and the later zero-crossing that triggers a merger are discrete. A derivative of the calculated clock can describe local timescale dependence; it must not be presented as a derivative through merger identity or event occurrence.

## Executable evidence and current boundary

The compiled C oracle exercises a Type-0 disk radius and a four-member group containing a reset central, initialized Type-1 satellite, immediate Type-2 orphan, and skipped Type-3 entry. All five controlled fields match exactly. Python tests additionally cover the float fallback, frozen satellite radius, JIT/VMAP execution, a halo-spin gradient, under-resolved immediate clocks, the `998.0` ceiling, and the no-central behavior.

This is pre-timestep process equivalence, not yet merger-event, full-tree, or Mini-Millennium equivalence. Clock decrement, disruption/merger selection, event payloads, immediate quasar/starburst consumers, and descendant inheritance remain separate ordered maps.

Current code: [`disk_radius.py`](../mimic_jax/sage16/processes/disk_radius.py) and [`merger_clock.py`](../mimic_jax/sage16/processes/merger_clock.py). Tests: [`test_disk_radius_merger_clock.py`](../tests/mimic_jax/test_disk_radius_merger_clock.py).
