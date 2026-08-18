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

The reproducible checker additionally evolves tree 61, a 67-node history with two multi-progenitor joins and FoF groups of size two. Across all eight configured output snapshots it compares 13 records and 546 public fields by `UniqueGalaxyID`. Integers match exactly. Float32 reservoirs use the same `2e-6` tolerance. The largest resolved relative difference is `1.056e-6`, in accumulated float32 metal reservoirs. Output `Cooling` and `Heating` use `rtol=3e-10, atol=1e-8` because a tiny internal power difference is subsequently transformed by `log10`; the largest measured luminosity relative difference is `2.1e-10`.

Run the direct comparison after producing the upstream fiducial catalogue:

```bash
JAX_ENABLE_X64=1 mimic_venv/bin/python \
  scripts/check_mini_millennium_tree_equivalence.py \
  --tree 61 --all-output-snapshots
```

This establishes complete process, inheritance, schedule, and selected real-tree catalogue equivalence. It does **not** yet claim all-tree Mini-Millennium population equivalence, a stellar-mass-function match, or a performance result. Those require a partition/catalogue runner and quantitative aggregate comparisons rather than extrapolation from two trees.

Code: [`mimic_jax/io/lhalo.py`](../mimic_jax/io/lhalo.py), [`mimic_jax/sage16/tree_evolve.py`](../mimic_jax/sage16/tree_evolve.py), and [`mimic_jax/sage16/catalogue.py`](../mimic_jax/sage16/catalogue.py). Tests: [`tests/mimic_jax/test_lhalo_io.py`](../tests/mimic_jax/test_lhalo_io.py) and [`tests/mimic_jax/test_tree_evolve.py`](../tests/mimic_jax/test_tree_evolve.py).
