"""Precision helpers that reproduce SAGE16's float-storage boundaries."""

import jax
import jax.numpy as jnp


def require_x64() -> None:
    """Fail before silently narrowing SAGE16's double transport calculations."""

    if not jax.config.x64_enabled:
        raise RuntimeError(
            "SAGE16 parity requires JAX 64-bit mode. Set JAX_ENABLE_X64=1 before importing JAX."
        )


def as_float32(value):
    """Round a persistent reservoir write exactly once at its C float boundary."""

    return jnp.asarray(value, dtype=jnp.float32)


def as_float64(value):
    """Promote a stored value for SAGE16's double-precision local arithmetic."""

    require_x64()
    return jnp.asarray(value, dtype=jnp.float64)
