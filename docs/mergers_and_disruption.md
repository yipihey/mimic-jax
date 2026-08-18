# Fiducial SAGE16 Mergers and Disruption

Mergers and disruption are discrete ordered state maps, not continuous rates. The JAX implementation uses `lax.scan` over the live FoF workspace because an event can change the state or identity seen by every later satellite in the same pass.

## Upstream sequence

The source is [`sage_resolve_mergers_and_disruption.c`](../models/sage16/modules/sage_resolve_mergers_and_disruption/sage_resolve_mergers_and_disruption.c), with exact ownership arithmetic in [`sage_merger_ops.h`](../models/sage16/modules/sage_resolve_mergers_and_disruption/sage_merger_ops.h). For each live Type-1/2 satellite, upstream:

1. derives the source-local substep duration `dT_source / N` and decrements the float `MergTime` field;
2. evaluates the live halo mass `Mvir - deltaMvir [1 - (k + 1)/N]`, floored at zero;
3. marks the source eligible when its stellar-plus-cold baryonic mass is zero or its live virial-to-baryonic ratio is at most `ThresholdSatDisruption`;
4. resolves the live target, including one redirect through the `CentralHalo` link when a Type-2 target was already consumed earlier in this pass;
5. disrupts if the decremented clock remains positive, otherwise merges;
6. for a merger, transfers ownership, runs quasar and starburst consumers immediately, applies the same-step minor-merger disk-instability follow-up, writes major/minor event history, marks the source Type 3, and only then advances to the next source.

The baryonic merger ratio is computed from the live pre-transfer states:

`mu = min(Mstar,s + Mcold,s, Mstar,t + Mcold,t) / max(Mstar,s + Mcold,s, Mstar,t + Mcold,t)`,

with `mu = 1` when both totals vanish. A minor-merger time is recorded for `mu > 0.1`; a major merger is the strict condition `mu > ThresholdMajorMerger` and makes all remnant stars and stellar metals bulge components. Equality with the major threshold remains on the minor branch.

## Explicit ownership maps

A merger sends cold to cold, hot to hot, ejected to ejected, stars to stars and the bulge component, ICS to ICS, and BH mass to BH mass. Pre-existing tracked metals follow their corresponding reservoirs.

A disruption instead heats both cold and hot source gas into the target hot atmosphere, retains ejected gas as ejected, and sends bound source stars plus source ICS into target ICS. Upstream does not transfer the disrupted source's black-hole mass. `MergerOwnershipTransfer.black_hole_sink` exposes that loss explicitly; it is not hidden by weakening the conservation test.

Consumed source records are retained in the workspace but marked Type 3. Consequently, a physical group ledger must sum only live non-Type-3 owners. `active_group_baryonic_mass` and `active_group_metal_mass` implement that boundary. Mergers conserve both ledgers within upstream float-write precision. Disruption conserves tracked metals and conserves baryons after subtracting the declared BH sink.

## Immediate event consumers

The fiducial `satellite_mergers` phase is

`resolver -> merger quasar mode -> merger starburst -> optional post-minor disk instability -> optional quasar follow-up -> disk-instability starburst`.

The first quasar and starburst use the live post-transfer target. Merger starbursts use `0.56 mu^0.7` and accumulate rates using the target's full `dT`, while the event payload's source substep duration validates event timing. For `mu < ThresholdMajorMerger`, SAGE rechecks disk instability after the merger burst. The fiducial configuration then applies its quasar growth/wind before the disk-instability burst. mimic-jax returns all five consumer transfer records separately.

The named `quasar_mode`, `starburst`, `sn_reheating`, `sn_ejection`, and `disk_instability` fractional perturbations also act on these event consumers. Such a derivative is conditional on the same merger event, target, and classification. It says how a faithful event's physical transfer would respond locally; it does not differentiate whether the discrete event occurs.

## Timing, failures, and numerical meaning

The event time is the source object's substep midpoint, `(time + dT_source) - (k + 1/2) dT_source/N`, and is narrowed when written to event-history fields. The result reports per-source action, target, clock timing, live halo ratio, event transfers, and explicit error/status codes. Like upstream, an invalid object timestep, unset clock, or invalid target halts later processing; an initial-boundary object is skipped and a non-finite decremented clock is a nonfatal skip.

Changing the substep count can change both the clock-crossing epoch and the order in which thresholded events interact with ordinary galaxy physics. A continuous ODE integrator must never absorb this map. Numerical convergence studies must preserve it as a jump map and report event-sequence changes separately from continuous truncation error.

## Executable evidence and current boundary

The compiled oracle covers a disruption followed by a minor merger into the same central, including ownership transfers, quasar growth/wind, merger burst, post-minor instability recheck, second quasar/burst chain, event times, clocks, and source Type changes. All 23 controlled event fields match compiled SAGE16 exactly. Python tests cover the disruption BH sink, major history updates, live one-hop redirects, object-local timing and halo-mass interpolation, fail-fast behavior, JIT, VMAP over equal-sized groups, and derivative-level ownership conservation.

This establishes the isolated fiducial event phase. Full-tree equivalence still requires the tree inheritance map and a complete group driver that interleaves ordinary galaxy-major physics with shared-central writes before invoking this phase.

Current code: [`mergers.py`](../mimic_jax/sage16/processes/mergers.py) and [`transfers.py`](../mimic_jax/sage16/transfers.py). Tests: [`test_mergers.py`](../tests/mimic_jax/test_mergers.py).
