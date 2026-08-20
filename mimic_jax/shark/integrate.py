"""JAX-native integration entry points for the SHARK flow system."""

from mimic_jax.numerics import integrate_adaptive, integrate_fixed_step
from mimic_jax.shark.flows import (
    shark_augmented_continuous_rhs,
    shark_continuous_rhs,
    shark_rhs,
)


def _flow_rhs(formulation, rate_law, parameters, reincorporation_rate_law):
    if formulation == "reference":
        if reincorporation_rate_law is not None:
            raise ValueError(
                "Reference ODE excludes upstream's finite reincorporation map; "
                "use formulation='continuous' for a reincorporation rate law."
            )
        return lambda time, state: shark_rhs(time, state, rate_law, parameters)
    if formulation == "continuous":
        return lambda time, state: shark_continuous_rhs(
            time,
            state,
            rate_law,
            parameters,
            reincorporation_rate_law,
        )
    raise ValueError("formulation must be 'reference' or 'continuous'")


def integrate_shark_flow(
    initial_state,
    rate_law,
    parameters,
    *,
    duration,
    num_steps,
    method="rk4",
    start_time=0.0,
    formulation="reference",
    reincorporation_rate_law=None,
):
    """Integrate reference or explicit-continuous SHARK flows at fixed step."""

    rhs = _flow_rhs(formulation, rate_law, parameters, reincorporation_rate_law)
    return integrate_fixed_step(
        rhs,
        initial_state,
        start_time=start_time,
        duration=duration,
        num_steps=num_steps,
        method=method,
    )


def integrate_shark_flow_adaptive(
    initial_state,
    rate_law,
    parameters,
    *,
    duration,
    start_time=0.0,
    formulation="reference",
    reincorporation_rate_law=None,
    **solver_options,
):
    """Adaptively integrate reference or explicit-continuous SHARK flows."""

    rhs = _flow_rhs(formulation, rate_law, parameters, reincorporation_rate_law)
    return integrate_adaptive(
        rhs,
        initial_state,
        start_time=start_time,
        duration=duration,
        **solver_options,
    )


def integrate_shark_augmented_flow(
    initial_state,
    rate_law,
    parameters,
    *,
    duration,
    num_steps,
    method="rk4",
    start_time=0.0,
):
    """Integrate the augmented continuous reservoir/BH state at fixed step."""

    rhs = lambda time, state: shark_augmented_continuous_rhs(time, state, rate_law, parameters)
    return integrate_fixed_step(
        rhs,
        initial_state,
        start_time=start_time,
        duration=duration,
        num_steps=num_steps,
        method=method,
    )


def integrate_shark_augmented_flow_adaptive(
    initial_state,
    rate_law,
    parameters,
    *,
    duration,
    start_time=0.0,
    **solver_options,
):
    """Adaptively integrate the augmented continuous reservoir/BH state."""

    rhs = lambda time, state: shark_augmented_continuous_rhs(time, state, rate_law, parameters)
    return integrate_adaptive(
        rhs,
        initial_state,
        start_time=start_time,
        duration=duration,
        **solver_options,
    )
