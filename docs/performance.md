# mimic-jax Performance Method

Performance claims must compare equivalent scientific work and separate compilation from execution. The complete SAGE16 FoF kernel is now connected to Mini-Millennium trees, but the current result is a diagnostic, not a speedup claim: cold JAX execution is dramatically slower than upstream MIMIC, while warmed batches are much faster than their first invocation.

## Reproducible partition benchmark

[`scripts/benchmark_mini_millennium_partition.py`](../scripts/benchmark_mini_millennium_partition.py) reports four separately timed components:

- first-process evolution, including tracing, compilation, execution, host tree assembly, and transfers back to the host;
- repeated evolution in the same process, reusing compiled executables;
- time spent inside padded/stacked JAX batches versus the remaining host tree driver;
- ordinary-Python catalogue conversion after evolution.

It also records JAX/backend/hardware metadata, the number of executable array shapes, batch calls, input halos, FoF updates, output records, peak process resident memory, and a SHA-256 digest of every retained catalogue field. Repeated runs must produce the same digest. `member_binning=exact` preserves each live workspace size. `power_of_two` appends trailing Type-3 records, which all physical and event modules skip, to reduce compilation shapes. Tests require the live leaves to be bitwise unchanged by that padding. `max_batch_members` bounds the product of member slots and VMAP groups.

Reproduce a representative CPU measurement with:

```bash
JAX_ENABLE_X64=1 mimic_venv/bin/python \
  scripts/benchmark_mini_millennium_partition.py \
  --tree-start 1500 --tree-count 100 --repeats 2 \
  --member-binning power_of_two --max-batch-members 512
```

An optional `--compilation-cache-dir` measures persistent-cache behavior across separate processes. A cache hit is not labeled a warm run: Python tracing/lowering and executable deserialization remain visible.

## Current CPU evidence

These development measurements were made on an Apple arm64 CPU under macOS 26.6.1, Python 3.9.6, JAX 0.4.30, the CPU backend, x64 enabled, batch size 128, and no GPU. They are committed to make the current problem quantitative, not to advertise universal timings.

| Workload | Shapes | First evolution | Same-process warm evolution | Warm JAX batches | Warm host driver | Catalogue | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| trees 1500--1599: 2,932 input halos, 2,929 FoF updates | 2 | 45.6 s | 0.53 s | 0.20 s | 0.33 s | 0.48 s | 2.9 GB |
| tree 0: 4,569 input halos, 3,397 FoF updates, power-of-two members | 7 | 172.8 s | 3.01 s | 2.54 s | 0.47 s | 0.018 s | 7.5 GB |

Exact and power-of-two member binning produced identical catalogue digests on the 100-tree control. Reducing the member budget from 4,096 to 512 left the tree-0 cold time essentially unchanged but reduced its warm evolution from 8.11 s to 3.01 s. An earlier 20-tree diagnostic compiled 11 padded executable shapes, took 268 s cold, and reached about 11.3 GB peak RSS. After removing batch-size specialization, tree 0 by itself requires 7 shapes, takes 173 s cold, and reaches about 7.5 GB; because the workloads differ, this is evidence about specialization count rather than a direct runtime or memory speedup ratio.

A two-shape 100-tree run took 50.6 s in a fresh process while populating a persistent cache, 28.4 s in another fresh process using the 3.9 MB cache, and about 0.53 s when reused inside one process. Persistent compilation therefore helps but does not remove the tracing/lowering boundary.

The upstream executable evolved all eight Mini-Millennium partitions in about 1.9--2.1 s on this machine. A prior exact-shape cold JAX run of partition 0 alone took 975 s. Those workloads and output scopes are not yet a fair speedup ratio, but they establish the practical conclusion unambiguously: the current cold catalogue path is much slower than upstream and must be improved. No end-to-end JAX speedup is claimed.

## Scientific gate

Performance is accepted only for paths that pass equivalence checks. The optimized path passed 9,408 public-field comparisons for trees 1500--1599 across all configured output snapshots. A complex tree-0 check retained one failure among 33,306 comparisons: `SupernovaOutflowRate` at snapshot 32 differed by `1.26e-5` relatively after a near-threshold SFR difference was multiplied by the reheating factor. The same failure occurs on the exact-size path, so inactive padding is not its cause. Full-population equivalence remains open; tolerances have not been widened to turn the performance run green.

## Earlier process benchmark

[`scripts/benchmark_mimic_jax.py`](../scripts/benchmark_mimic_jax.py) remains useful for isolating the controlled quiescent process subset. It records eager scalar time, first JIT call time including compilation, warmed scalar JIT time, first batched `vmap` plus JIT time, and warmed batch time per galaxy. Reproduce it with:

```bash
source mimic_venv/bin/activate
JAX_ENABLE_X64=1 python scripts/benchmark_mimic_jax.py --batch-size 4096 --repeats 100
```

Future matched comparisons will use the same Mini-Millennium input, parameters, output scope, precision, and process count; report compilation, first execution, warm execution, catalogue I/O, peak memory, and CPU/GPU backend separately; and compare at matched scientific accuracy. No committed number is a universal performance claim.
