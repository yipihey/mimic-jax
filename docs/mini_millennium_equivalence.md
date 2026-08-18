# Mini-Millennium Tree and Catalogue Equivalence

`evolve_lhalo_tree` connects the complete JAX FoF kernel to real legacy L-Halo merger trees. This is the first end-to-end SAGE16 path in mimic-jax: binary tree record to inherited galaxy workspaces to a catalogue-ready final record. It preserves the upstream tree driver rather than replacing the discrete topology with a differentiable surrogate.

## Driver boundary

The ordinary-Python driver owns operations that are intrinsically ragged or discrete:

- reading the 104-byte aligned `RawHalo` binary records;
- following progenitor and FoF linked lists;
- selecting upstream's first occupied progenitor;
- allocating persistent `UniqueGalaxyID` values;
- omitting consumed Type-3 records from descendant/output segments; and
- scheduling FoF groups after their progenitors.

For each fixed-shape FoF interval, it stacks immutable `GalaxyState` and `HaloForcing` PyTrees and calls the JIT-compiled `evolve_upstream_sequential_group_interval` kernel. This boundary keeps tree connectivity discrete while preserving differentiation through the baryonic evolution on a fixed tree schedule. A later padded/static tree representation can move more orchestration onto device without changing either the physics kernel or catalogue contract.

The driver reproduces the upstream adaptive-Simpson lookback-time table, virial mass fallback, 200-critical radius and velocity, object-local `dT`, central identity, `CentralMvir`, and ten-substep context. Halo forcing is piecewise constant over a tree interval, exactly as in the reference run.

## Catalogue boundary

`record_to_catalogue` explicitly performs the same non-physical output transformations as `src/include/generated/copy_to_output.inc`: code time to Myr/h, internal mass rate to solar masses per year, cooling/heating power to cgs `log10`, merger time to Gyr/h, central infall sentinels to zero, and current Type-0/1 virial-property recalculation. Reservoirs and metals remain in SAGE's `1e10 Msun/h` units.

Keeping this conversion outside the differentiable core prevents output units from contaminating physical transfer calculations and makes every comparison field traceable.

## Executable evidence

The permanent regression evolves partition 0, tree 1575: a six-node linear history spanning snapshots 58–63. All 42 z=0 public catalogue fields match an upstream MIMIC HDF5 record. Integers are exact; float32 fields use `rtol=atol=2e-6`; ordinary float64 fields use `rtol=atol=2e-12`. The only unresolved upstream value is a `-1.6e-24` float32 hot-metal roundoff where mimic-jax produces exact zero, covered by the stated absolute tolerance.

The reproducible checker additionally evolves tree 61, a 67-node history with two multi-progenitor joins and FoF groups of size two. Across all eight configured output snapshots it compares 13 records and 546 public fields by `UniqueGalaxyID`. Integers match exactly. Float32 reservoirs and the mixed-precision `log10` cooling/heating diagnostics use `rtol=atol=2e-6`; other float64 fields use `rtol=atol=2e-12`. The largest resolved relative difference is `1.056e-6`, in accumulated float32 metal reservoirs. The largest measured luminosity relative difference on this tree is `2.1e-10`; the broader tolerance is declared because other real histories can accumulate float-level differences before the output transform.

Run the direct comparison after producing the upstream fiducial catalogue:

```bash
JAX_ENABLE_X64=1 mimic_venv/bin/python \
  scripts/check_mini_millennium_tree_equivalence.py \
  --tree 61 --all-output-snapshots
```

This establishes complete process, inheritance, schedule, and selected real-tree catalogue equivalence. It does **not** yet claim all-tree Mini-Millennium population equivalence, a stellar-mass-function match, or a performance result. Those require a partition/catalogue runner and quantitative aggregate comparisons rather than extrapolation from two trees.

## Partition gate and current limitation

`evolve_lhalo_partition` batches independent same-snapshot FoF groups while the host retains ragged tree ownership. The exact mode specializes on live member count. The optional power-of-two mode appends inactive Type-3 slots and accepts the central index as batched data, reducing the number of compiled array shapes without reordering live galaxies. Infall totals use the same sequential member accumulation as upstream, so trailing zero/inactive slots do not change the floating-point reduction tree. Unit tests compare every live state and halo leaf bitwise between exact and padded group kernels.

On partition-0 trees 1500--1599, the power-of-two path evolved 2,932 input halos and 2,929 FoF intervals and passed all 9,408 public-field comparisons over the eight configured output snapshots. Its catalogue digest also matched the exact-member path. This is a useful multi-tree gate, not full-population evidence.

A deliberately complex tree-0 check currently has one failure among 33,306 comparisons: snapshot-32 `SupernovaOutflowRate` differs by `1.26e-5` relatively. The underlying SFR differs by about `1e-6` in absolute output units near a threshold, then the fiducial reheating factor of three amplifies the diagnostic-rate difference. The same field failed in the earlier exact-member full-partition diagnostic, so member padding is not responsible. A cold exact-member pass over partition 0 found 26 comparisons outside the declared tolerances among 890,274 fields, with the largest relative reservoir difference about `4.4e-4`. These discrepancies remain under investigation; the tolerances are not silently broadened and all-tree equivalence is not claimed.

Run the batched gate with:

```bash
JAX_ENABLE_X64=1 mimic_venv/bin/python \
  scripts/check_mini_millennium_partition_equivalence.py \
  --tree-start 1500 --tree-count 100 \
  --member-binning power_of_two --max-batch-members 512
```

See [`performance.md`](performance.md) for cold/warm timing and memory measurements from the same path.

Code: [`mimic_jax/io/lhalo.py`](../mimic_jax/io/lhalo.py), [`mimic_jax/sage16/tree_evolve.py`](../mimic_jax/sage16/tree_evolve.py), and [`mimic_jax/sage16/catalogue.py`](../mimic_jax/sage16/catalogue.py). Tests: [`tests/mimic_jax/test_lhalo_io.py`](../tests/mimic_jax/test_lhalo_io.py) and [`tests/mimic_jax/test_tree_evolve.py`](../tests/mimic_jax/test_tree_evolve.py).
