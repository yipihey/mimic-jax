# mimic-jax Performance Method

Performance claims must compare equivalent scientific work and separate compilation from execution. The initial benchmark therefore reports only the implemented quiescent SAGE16 subset and does not call it an end-to-end speedup over MIMIC.

[`scripts/benchmark_mimic_jax.py`](../scripts/benchmark_mimic_jax.py) records eager scalar time, first JIT call time including compilation, warmed scalar JIT time, first batched `vmap` plus JIT time, and warmed batch time per galaxy. It synchronizes JAX results before stopping each timer. Command-line arguments control batch size, warm-up calls, repeats, and optional JSON output.

A publishable comparison will use the complete SAGE16 workload and the same Mini-Millennium input, model parameters, output scope, hardware, precision, and process count. It will report C upstream wall time, JAX compilation time, JAX warmed execution, CPU batch scaling, GPU execution when available, and peak memory. Compilation may be amortized in a calibration workflow, but it must not disappear from the report.

Reproduce the current subset measurement with:

```bash
source mimic_venv/bin/activate
JAX_ENABLE_X64=1 python scripts/benchmark_mimic_jax.py --batch-size 4096 --repeats 100
```

No committed number is a universal performance claim. Hardware, JAX/JAXLIB versions, backend, thread configuration, batch size, and whether compilation is included must accompany any reported result.
