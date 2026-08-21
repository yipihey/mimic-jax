"""Minimal common protocol for configured differentiable SAM formulations.

The protocol standardizes mathematical and scientific questions, not model
physics. SAGE16 and SHARK retain their native states, parameters, forcing,
rates, finite maps, and upstream-equivalence entry points.
"""

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from mimic_jax.linear_response import (
    AnnotatedStateSpace,
    LinearizationPoint,
    ResponseCoordinate,
    linearize_annotated,
)
from mimic_jax.sensitivity import parameter_response_matrix

Array = Any

_VARIABLE_ROLES = {
    "reservoir",
    "metal_reservoir",
    "angular_momentum",
    "memory",
    "tracker",
    "forcing",
    "parameter",
    "observable",
}
_OPERATOR_TYPES = {"flow", "forcing", "event", "constraint", "projection"}
_CAPABILITY_STATES = {
    "supported",
    "partial",
    "model_specific",
    "unavailable",
    "not_evaluated",
    "not_applicable",
}


@dataclass(frozen=True)
class VariableMetadata:
    """Physical meaning and units of one model-owned coordinate."""

    name: str
    label: str
    unit: str
    role: str
    description: str

    def __post_init__(self) -> None:
        if self.role not in _VARIABLE_ROLES:
            raise ValueError(f"Unknown SAM variable role {self.role!r}")
        if not self.name or not self.label or not self.unit or not self.description:
            raise ValueError("SAM variable metadata fields cannot be empty")

    def response_coordinate(self) -> ResponseCoordinate:
        """Convert to the coordinate metadata used by response calculations."""

        return ResponseCoordinate(self.name, self.label, self.unit, self.description)


@dataclass(frozen=True)
class ProcessMetadata:
    """One model process and its faithful mathematical classification."""

    name: str
    label: str
    operator_type: str
    description: str
    differentiability: str
    source_reference: str
    source_reservoirs: Tuple[str, ...] = ()
    target_reservoirs: Tuple[str, ...] = ()
    perturbable: bool = False

    def __post_init__(self) -> None:
        if self.operator_type not in _OPERATOR_TYPES:
            raise ValueError(f"Unknown SAM operator type {self.operator_type!r}")
        if not self.name or not self.description or not self.source_reference:
            raise ValueError("SAM process metadata requires name, description, and source")


@dataclass(frozen=True)
class Capability:
    """Honest availability state for one configured model capability."""

    name: str
    status: str
    detail: str

    def __post_init__(self) -> None:
        if self.status not in _CAPABILITY_STATES:
            raise ValueError(f"Unknown SAM capability status {self.status!r}")
        if not self.name or not self.detail:
            raise ValueError("SAM capabilities require a name and explanatory detail")


@dataclass(frozen=True)
class SamModelMetadata:
    """Identity and semantic surface of one configured SAM formulation."""

    name: str
    label: str
    upstream_repository: str
    upstream_revision: str
    formulation: str
    qualification: str
    time_unit: str
    time_unit_in_gyr: float
    state_variables: Tuple[VariableMetadata, ...]
    forcing_variables: Tuple[VariableMetadata, ...]
    parameter_variables: Tuple[VariableMetadata, ...]
    observable_variables: Tuple[VariableMetadata, ...]
    processes: Tuple[ProcessMetadata, ...]
    capabilities: Tuple[Capability, ...]

    def __post_init__(self) -> None:
        for kind, values in (
            ("state", self.state_variables),
            ("forcing", self.forcing_variables),
            ("parameter", self.parameter_variables),
            ("observable", self.observable_variables),
            ("process", self.processes),
            ("capability", self.capabilities),
        ):
            names = tuple(value.name for value in values)
            if len(set(names)) != len(names):
                raise ValueError(f"Duplicate {kind} names in model {self.name}: {names}")

    @property
    def process_control_names(self) -> Tuple[str, ...]:
        """Processes accepting dimensionless log-rate perturbations."""

        return tuple(process.name for process in self.processes if process.perturbable)

    def capability(self, name: str) -> Capability:
        """Return one declared capability or fail instead of assuming support."""

        for capability in self.capabilities:
            if capability.name == name:
                return capability
        raise KeyError(f"Model {self.name!r} does not declare capability {name!r}")

    def to_dict(self):
        """Return a JSON-ready semantic manifest for reports and agents."""

        return asdict(self)


@dataclass(frozen=True)
class ConservationQuantity:
    """One tracked conserved quantity and its explicit open-system boundary."""

    name: str
    value: Array
    unit: str
    description: str
    external_sources: Tuple[str, ...] = ()
    external_sinks: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ConservationBalance:
    """Instantaneous conservation residual after named sources and sinks."""

    name: str
    residual: Array
    source_rate: Array
    sink_rate: Array
    unit: str
    description: str


@dataclass(frozen=True)
class SamJacobians:
    """State and parameter Jacobians at one physically identified operating point."""

    state_jacobian: Array
    parameter_jacobian: Array
    parameter_values: Array
    point: LinearizationPoint
    state_coordinates: Tuple[ResponseCoordinate, ...]
    parameter_coordinates: Tuple[ResponseCoordinate, ...]
    derivative_method: str = "jax.jacfwd"


class Sage16ContinuousForcing(NamedTuple):
    """Fixed halo and disk-size forcing for the SAGE16 continuous subset."""

    halo: Any
    disk_scale_radius: Array


class SharkContinuousForcing(NamedTuple):
    """Fixed structural forcing for the controlled SHARK Lagos23 disk subset."""

    disk: Any
    reincorporation_rate: Array


class SharkCommonRhsResult(NamedTuple):
    """Continuous SHARK derivative with native rates and pre-ODE diagnostics."""

    derivative: Any
    rates: Any
    cold_gas_metallicity: Array
    effective_yield: Array
    reincorporation: Array


def get_parameter_path(parameters, path: str):
    """Read a dot-separated field path from nested immutable parameter records."""

    current = parameters
    for component in path.split("."):
        current = getattr(current, component)
    return current


def replace_parameter_path(parameters, path: str, value):
    """Replace one nested immutable parameter field by dot-separated path."""

    components = path.split(".")
    if not components or any(not component for component in components):
        raise ValueError(f"Invalid parameter path {path!r}")

    def replace(current, remaining):
        name = remaining[0]
        if len(remaining) == 1:
            return current._replace(**{name: value})
        child = getattr(current, name)
        return current._replace(**{name: replace(child, remaining[1:])})

    return replace(parameters, components)


@dataclass(frozen=True)
class ConfiguredSamModel:
    """Executable common boundary around one already configured SAM formulation."""

    metadata: SamModelMetadata
    default_parameters: Any
    _rhs_and_rates: Callable = field(repr=False)
    _conserved_quantities: Callable = field(repr=False)
    _conservation_balances: Callable = field(repr=False)
    _rate_value: Callable = field(repr=False)
    _observable_value: Callable = field(repr=False)

    def _controls(self, log_process_perturbations=None):
        count = len(self.metadata.process_control_names)
        if log_process_perturbations is None:
            return jnp.zeros((count,), dtype=jnp.float64)
        controls = jnp.asarray(log_process_perturbations, dtype=jnp.float64)
        if controls.shape != (count,):
            raise ValueError(
                f"{self.metadata.name} expects {count} process controls in order "
                f"{self.metadata.process_control_names}, received shape {controls.shape}"
            )
        return controls

    def rhs_and_rates(
        self,
        time,
        state,
        forcing,
        parameters=None,
        log_process_perturbations=None,
    ):
        """Evaluate the configured native rate layer and reservoir derivative."""

        parameters = self.default_parameters if parameters is None else parameters
        controls = self._controls(log_process_perturbations)
        return self._rhs_and_rates(time, state, forcing, parameters, controls)

    def rhs(
        self,
        time,
        state,
        forcing,
        parameters=None,
        log_process_perturbations=None,
    ):
        """Return only ``dx/dt`` through the common continuous-flow boundary."""

        return self.rhs_and_rates(
            time, state, forcing, parameters, log_process_perturbations
        ).derivative

    def rate_value(self, result, process_name: str):
        """Return a named physical rate from a model-native RHS result."""

        if process_name not in self.metadata.process_control_names:
            raise KeyError(f"{process_name!r} is not a process control for {self.metadata.name}")
        return self._rate_value(result, process_name)

    def conserved_quantities(self, state) -> Tuple[ConservationQuantity, ...]:
        """Return common scientific ledgers without changing model ownership."""

        return tuple(self._conserved_quantities(state))

    def observable_value(self, state, result, observable_name: str):
        """Return one declared local observable without imposing a catalogue ontology."""

        available = {variable.name for variable in self.metadata.observable_variables}
        if observable_name not in available:
            raise KeyError(
                f"{observable_name!r} is not a local observable for {self.metadata.name}; "
                f"choose from {tuple(sorted(available))}"
            )
        return self._observable_value(state, result, observable_name)

    def conservation_balances(self, result) -> Tuple[ConservationBalance, ...]:
        """Return rate-level residuals after explicit sources and sinks."""

        return tuple(self._conservation_balances(result))

    def parameter_response(
        self,
        observable_fn: Callable[[Any], Array],
        *,
        parameter_names: Sequence[str],
        parameters=None,
        **response_options,
    ):
        """Return dimensionless parameter responses, including nested parameters."""

        parameters = self.default_parameters if parameters is None else parameters
        units = {variable.name: variable.unit for variable in self.metadata.parameter_variables}
        response_options.setdefault(
            "parameter_units", tuple(units.get(name, "unspecified") for name in parameter_names)
        )
        response = parameter_response_matrix(
            observable_fn,
            parameters,
            parameter_names=parameter_names,
            parameter_getter=get_parameter_path,
            parameter_replacer=replace_parameter_path,
            **response_options,
        )
        return replace(
            response,
            model=self.metadata.label,
            formulation=self.metadata.formulation,
            qualification=self.metadata.qualification,
        )

    def local_response(
        self,
        *,
        time,
        state,
        forcing,
        output: Callable,
        output_coordinates: Sequence[ResponseCoordinate],
        parameters=None,
        redshift: Optional[float] = None,
        halo_mass: Optional[float] = None,
        halo_mass_unit: str = "unspecified",
        qualification: Optional[str] = None,
    ) -> AnnotatedStateSpace:
        """Linearize process perturbations around one nonlinear model state.

        ``output`` receives ``(state, forcing, parameters, controls)``. Inputs
        are dimensionless log-rate perturbations in
        ``metadata.process_control_names`` order, so each input derivative has
        the practitioner-facing meaning "response to a fractional change in
        this process at this operating point."
        """

        parameters = self.default_parameters if parameters is None else parameters
        controls = self._controls()
        processes = {process.name: process for process in self.metadata.processes}
        input_coordinates = tuple(
            ResponseCoordinate(
                name,
                processes[name].label,
                "fractional process change",
                processes[name].description,
            )
            for name in self.metadata.process_control_names
        )

        def controlled_rhs(current_state, current_controls):
            return self.rhs(time, current_state, forcing, parameters, current_controls)

        def controlled_output(current_state, current_controls):
            return output(current_state, forcing, parameters, current_controls)

        point = LinearizationPoint(
            model=self.metadata.label,
            formulation=self.metadata.formulation,
            time=float(time),
            time_unit=self.metadata.time_unit,
            time_unit_in_gyr=self.metadata.time_unit_in_gyr,
            redshift=redshift,
            halo_mass=halo_mass,
            halo_mass_unit=halo_mass_unit,
            qualification=(self.metadata.qualification if qualification is None else qualification),
        )
        return linearize_annotated(
            controlled_rhs,
            controlled_output,
            state,
            controls,
            point=point,
            state_coordinates=tuple(
                variable.response_coordinate() for variable in self.metadata.state_variables
            ),
            input_coordinates=input_coordinates,
            output_coordinates=output_coordinates,
        )

    def jacobians(
        self,
        *,
        time,
        state,
        forcing,
        parameter_names: Sequence[str],
        parameters=None,
        redshift: Optional[float] = None,
        halo_mass: Optional[float] = None,
        halo_mass_unit: str = "unspecified",
    ) -> SamJacobians:
        """Return ``A = df/dx`` and ``B_theta = df/dtheta`` for selected parameters."""

        parameters = self.default_parameters if parameters is None else parameters
        parameter_names = tuple(parameter_names)
        if not parameter_names:
            raise ValueError("At least one parameter path is required")
        selected = jnp.stack(
            tuple(jnp.asarray(get_parameter_path(parameters, name)) for name in parameter_names)
        )
        if any(
            not jnp.issubdtype(jnp.asarray(value).dtype, jnp.inexact) for value in tuple(selected)
        ):
            raise TypeError("Selected parameters must have differentiable floating-point dtype")
        flat_state, unravel_state = ravel_pytree(state)

        def evaluate(current_state, current_parameters):
            resolved = parameters
            for name, value in zip(parameter_names, current_parameters):
                resolved = replace_parameter_path(resolved, name, value)
            derivative = self.rhs(time, unravel_state(current_state), forcing, resolved)
            return ravel_pytree(derivative)[0]

        state_jacobian, parameter_jacobian = jax.jacfwd(evaluate, argnums=(0, 1))(
            flat_state, selected
        )
        parameter_units = {
            variable.name: variable.unit for variable in self.metadata.parameter_variables
        }
        return SamJacobians(
            state_jacobian=state_jacobian,
            parameter_jacobian=parameter_jacobian,
            parameter_values=selected,
            point=LinearizationPoint(
                model=self.metadata.label,
                formulation=self.metadata.formulation,
                time=float(time),
                time_unit=self.metadata.time_unit,
                time_unit_in_gyr=self.metadata.time_unit_in_gyr,
                redshift=redshift,
                halo_mass=halo_mass,
                halo_mass_unit=halo_mass_unit,
                qualification=self.metadata.qualification,
            ),
            state_coordinates=tuple(
                variable.response_coordinate() for variable in self.metadata.state_variables
            ),
            parameter_coordinates=tuple(
                ResponseCoordinate(
                    name,
                    name.replace("_", " "),
                    parameter_units.get(name, "unspecified"),
                    "Model-owned parameter derivative coordinate.",
                )
                for name in parameter_names
            ),
        )


def _leaf_parameter_metadata(parameters, prefix: str = "") -> Tuple[VariableMetadata, ...]:
    values = []
    for name in parameters._fields:
        value = getattr(parameters, name)
        path = f"{prefix}.{name}" if prefix else name
        if hasattr(value, "_fields"):
            values.extend(_leaf_parameter_metadata(value, path))
        else:
            values.append(
                VariableMetadata(
                    path,
                    path.replace("_", " "),
                    "dimensionless or native model unit",
                    "parameter",
                    "Model-owned parameter; consult the SHARK prescription metadata for units.",
                )
            )
    return tuple(values)


def _sage16_model(**options) -> ConfiguredSamModel:
    from mimic_jax.sage16 import (
        fiducial_parameters,
        load_cooling_tables,
        process_perturbations,
        sage16_ode_rhs_and_rates,
        sage16_units,
    )

    parameters = options.pop("parameters", None)
    units = options.pop("units", None)
    cooling_tables = options.pop("cooling_tables", None)
    if options:
        raise TypeError(f"Unknown SAGE16 model options: {sorted(options)}")
    parameters = fiducial_parameters() if parameters is None else parameters
    units = sage16_units() if units is None else units
    cooling_tables = load_cooling_tables() if cooling_tables is None else cooling_tables
    control_names = ("cooling", "star_formation", "sn_reheating", "sn_ejection", "reincorporation")

    state_descriptions = {
        "ColdGas": "Cold star-forming gas in the central galaxy.",
        "HotGas": "Quasi-hydrostatic halo gas owned by the central galaxy.",
        "EjectedGas": "Gas outside the hot halo awaiting reincorporation.",
        "StellarMass": "Total stellar mass; bulge mass is a subset, not an additional reservoir.",
        "MetalsColdGas": "Metal mass in cold gas.",
        "MetalsHotGas": "Metal mass in hot halo gas.",
        "MetalsEjectedGas": "Metal mass in ejected gas.",
        "MetalsStellarMass": "Metal mass locked in stars.",
    }
    state_variables = tuple(
        VariableMetadata(
            name,
            name,
            "1e10 Msun/h",
            "metal_reservoir" if name.startswith("Metals") else "reservoir",
            state_descriptions[name],
        )
        for name in state_descriptions
    )
    parameter_variables = tuple(
        VariableMetadata(
            name,
            name,
            "dimensionless",
            "parameter",
            "Fiducial SAGE16 parameter retaining its upstream MIMIC name.",
        )
        for name in parameters._fields
    )
    processes = (
        ProcessMetadata(
            "cooling",
            "Cooling",
            "flow",
            "Hot gas cooling into the cold disk.",
            "piecewise smooth",
            "mimic_jax.sage16.ode.calculate_continuous_cooling_rate",
            ("HotGas",),
            ("ColdGas",),
            True,
        ),
        ProcessMetadata(
            "star_formation",
            "Star formation",
            "flow",
            "Cold gas converted into long-lived stars with instantaneous recycling.",
            "thresholded piecewise smooth",
            "mimic_jax.sage16.ode._star_formation_rate",
            ("ColdGas",),
            ("StellarMass",),
            True,
        ),
        ProcessMetadata(
            "sn_reheating",
            "SN reheating",
            "flow",
            "Cold gas reheated into the hot halo by stellar feedback.",
            "piecewise smooth",
            "mimic_jax.sage16.ode._supernova_rates",
            ("ColdGas",),
            ("HotGas",),
            True,
        ),
        ProcessMetadata(
            "sn_ejection",
            "SN ejection",
            "flow",
            "Hot gas ejected beyond the halo by stellar feedback.",
            "piecewise smooth",
            "mimic_jax.sage16.ode._supernova_rates",
            ("HotGas",),
            ("EjectedGas",),
            True,
        ),
        ProcessMetadata(
            "reincorporation",
            "Reincorporation",
            "flow",
            "Ejected gas returned to the hot halo.",
            "thresholded piecewise smooth",
            "mimic_jax.sage16.ode._reincorporation_rate",
            ("EjectedGas",),
            ("HotGas",),
            True,
        ),
        ProcessMetadata(
            "halo_infall",
            "Halo infall",
            "forcing",
            "Tree-driven baryon supply applied as an explicit finite budget in reference SAGE16.",
            "piecewise smooth finite map",
            "mimic_jax.sage16.processes.infall",
            (),
            ("HotGas",),
        ),
        ProcessMetadata(
            "radio_mode_agn",
            "Radio-mode AGN",
            "projection",
            "Cooling suppression with a history-dependent heating-radius state in the hybrid formulation.",
            "thresholded and history dependent",
            "mimic_jax.sage16.processes.radio_mode_heating",
            ("HotGas",),
            (),
        ),
        ProcessMetadata(
            "satellite_stripping",
            "Satellite stripping",
            "flow",
            "Finite environment-driven transfer from satellite hot/ejected reservoirs.",
            "thresholded finite transfer",
            "mimic_jax.sage16.processes.satellite_stripping",
        ),
        ProcessMetadata(
            "disk_instability",
            "Disk instability",
            "event",
            "Threshold-triggered finite disk-to-bulge transfer and possible starburst/BH growth.",
            "thresholded event map",
            "mimic_jax.sage16.processes.disk_instability",
        ),
        ProcessMetadata(
            "merger",
            "Galaxy merger",
            "event",
            "Topology-changing progenitor combination and merger-driven transfers.",
            "discrete event map",
            "mimic_jax.sage16.processes.mergers",
        ),
    )
    metadata = SamModelMetadata(
        name="sage16",
        label="MIMIC/SAGE16",
        upstream_repository="https://github.com/darrencroton/mimic",
        upstream_revision="69590cc60dcb7b8b6510ee0b16b1ed921a6c4853",
        formulation="continuous quiescent central subset",
        qualification="Fixed-halo continuous limit; finite reference maps and events remain separate.",
        time_unit="internal MIMIC time",
        time_unit_in_gyr=float(units.UnitTime_in_s) / 3.15576e16,
        state_variables=state_variables,
        forcing_variables=(
            VariableMetadata(
                "halo.Mvir",
                "halo virial mass",
                "1e10 Msun/h",
                "forcing",
                "Tree-supplied virial mass.",
            ),
            VariableMetadata(
                "halo.Rvir",
                "halo virial radius",
                "Mpc/h",
                "forcing",
                "Tree-supplied virial radius.",
            ),
            VariableMetadata(
                "halo.Vvir",
                "halo virial velocity",
                "km/s",
                "forcing",
                "Tree-supplied virial velocity.",
            ),
            VariableMetadata(
                "disk_scale_radius",
                "disk scale radius",
                "Mpc/h",
                "forcing",
                "Stored disk scale radius held fixed over the local interval.",
            ),
        ),
        parameter_variables=parameter_variables,
        observable_variables=(
            VariableMetadata(
                "stellar_mass",
                "stellar mass",
                "1e10 Msun/h",
                "observable",
                "Total stellar reservoir in the continuous subset.",
            ),
            VariableMetadata(
                "cold_gas",
                "cold-gas mass",
                "1e10 Msun/h",
                "observable",
                "Cold star-forming gas reservoir.",
            ),
            VariableMetadata(
                "hot_gas", "hot-gas mass", "1e10 Msun/h", "observable", "Hot halo gas reservoir."
            ),
            VariableMetadata(
                "ejected_gas",
                "ejected-gas mass",
                "1e10 Msun/h",
                "observable",
                "Feedback-ejected gas reservoir.",
            ),
            VariableMetadata(
                "star_formation_rate",
                "star-formation rate",
                "1e10 Msun/h per internal time",
                "observable",
                "Instantaneous quiescent star-formation rate.",
            ),
        ),
        processes=processes,
        capabilities=(
            Capability(
                "continuous_rhs",
                "supported",
                "Five quiescent central flows plus metals under fixed halo forcing.",
            ),
            Capability(
                "events",
                "model_specific",
                "Merger, instability, and stripping maps remain explicit SAGE operators.",
            ),
            Capability(
                "constraints",
                "model_specific",
                "Infall budgets and source caps are finite SAGE reference constraints.",
            ),
            Capability(
                "conservation",
                "supported",
                "Baryon and metal ledgers include stellar-yield sources explicitly.",
            ),
            Capability(
                "full_tree_physics_parity",
                "supported",
                "The independent full-tree SAGE16 JAX population path is evaluated.",
            ),
            Capability(
                "independent_topology_driver",
                "supported",
                "JAX owns the SAGE16 L-Halo branch topology and event schedule.",
            ),
        ),
    )

    def rhs_and_rates(time, state, forcing, current_parameters, controls):
        perturbations = process_perturbations(**dict(zip(control_names, tuple(controls))))
        return sage16_ode_rhs_and_rates(
            time,
            state,
            forcing.halo,
            forcing.disk_scale_radius,
            current_parameters,
            units,
            cooling_tables,
            perturbations,
        )

    def quantities(state):
        return (
            ConservationQuantity(
                "baryons",
                sum(state[:4]),
                "1e10 Msun/h",
                "Mass in cold, hot, ejected, and stellar reservoirs.",
            ),
            ConservationQuantity(
                "metals",
                sum(state[4:]),
                "1e10 Msun/h",
                "Tracked metal mass; stellar nucleosynthetic yield is an explicit source.",
                ("stellar yield",),
            ),
        )

    def balances(result):
        baryon_residual = sum(result.derivative[:4])
        metal_rate = sum(result.derivative[4:])
        source = result.rates.produced_metals
        zero = jnp.zeros_like(source)
        return (
            ConservationBalance(
                "baryons",
                baryon_residual,
                zero,
                zero,
                "1e10 Msun/h per internal time",
                "Closed transfer residual for the continuous subset.",
            ),
            ConservationBalance(
                "metals",
                metal_rate - source,
                source,
                zero,
                "1e10 Msun/h per internal time",
                "Metal residual after the explicit stellar-yield source.",
            ),
        )

    def observable_value(state, result, name):
        values = {
            "stellar_mass": state.StellarMass,
            "cold_gas": state.ColdGas,
            "hot_gas": state.HotGas,
            "ejected_gas": state.EjectedGas,
            "star_formation_rate": result.rates.star_formation,
        }
        return values[name]

    return ConfiguredSamModel(
        metadata,
        parameters,
        rhs_and_rates,
        quantities,
        balances,
        lambda result, name: getattr(result.rates, name),
        observable_value,
    )


def _shark_model(**options) -> ConfiguredSamModel:
    from mimic_jax.shark import (
        SHARK_ODE_STATE_DESCRIPTIONS,
        SHARK_ODE_STATE_NAMES,
        SHARK_UPSTREAM_REPOSITORY,
        SHARK_UPSTREAM_REVISION,
        cold_gas_metallicity,
        effective_stellar_yield,
        lagos23_disk_flow_rates,
        lagos23_model_parameters,
        shark_continuous_rhs_from_rates,
    )

    parameters = options.pop("parameters", None)
    if options:
        raise TypeError(f"Unknown SHARK model options: {sorted(options)}")
    parameters = lagos23_model_parameters() if parameters is None else parameters
    control_names = (
        "cooling",
        "star_formation",
        "sn_reheating",
        "sn_ejection",
        "qso_reheating",
        "qso_ejection",
        "reincorporation",
    )
    mass_names = SHARK_ODE_STATE_NAMES[:6]
    metal_names = SHARK_ODE_STATE_NAMES[6:12]
    tracker_names = SHARK_ODE_STATE_NAMES[12:14]

    def state_role(name):
        if name in mass_names:
            return "reservoir"
        if name in metal_names:
            return "metal_reservoir"
        if name in tracker_names:
            return "tracker"
        return "angular_momentum"

    def state_unit(name):
        if name in mass_names or name in metal_names or name in tracker_names:
            return "Msun/h"
        return "Msun Mpc km/s / h^2"

    state_variables = tuple(
        VariableMetadata(
            name,
            name.replace("_", " "),
            state_unit(name),
            state_role(name),
            SHARK_ODE_STATE_DESCRIPTIONS[name],
        )
        for name in SHARK_ODE_STATE_NAMES
    )
    processes = (
        ProcessMetadata(
            "cooling",
            "Cooling",
            "flow",
            "Hot halo gas transferred directly to the cold disk in the continuous formulation.",
            "piecewise smooth",
            "mimic_jax.shark.flows.direct_cooling_flow_derivative",
            ("hot_halo_gas",),
            ("cold_gas",),
            True,
        ),
        ProcessMetadata(
            "star_formation",
            "Star formation",
            "flow",
            "BR06 molecular star formation with instantaneous recycling and metal production.",
            "piecewise smooth with activity threshold",
            "mimic_jax.shark.prescriptions.disk.lagos23_disk_flow_rates",
            ("cold_gas",),
            ("stellar_mass",),
            True,
        ),
        ProcessMetadata(
            "sn_reheating",
            "SN reheating",
            "flow",
            "Lagos13 stellar-feedback transfer from cold gas to the hot halo.",
            "piecewise smooth",
            "mimic_jax.shark.prescriptions.stellar_feedback.lagos13_feedback_loadings",
            ("cold_gas",),
            ("hot_halo_gas",),
            True,
        ),
        ProcessMetadata(
            "sn_ejection",
            "SN ejection",
            "flow",
            "Lagos13 stellar-feedback transfer from the hot halo to ejected gas.",
            "piecewise smooth",
            "mimic_jax.shark.prescriptions.stellar_feedback.lagos13_feedback_loadings",
            ("hot_halo_gas",),
            ("ejected_gas",),
            True,
        ),
        ProcessMetadata(
            "qso_reheating",
            "QSO reheating",
            "flow",
            "AGN/QSO loading that reheats cold gas into the hot halo.",
            "piecewise smooth under fixed forcing",
            "mimic_jax.shark.prescriptions.agn.lagos23_qso_outflow_loadings",
            ("cold_gas",),
            ("hot_halo_gas",),
            True,
        ),
        ProcessMetadata(
            "qso_ejection",
            "QSO ejection",
            "flow",
            "AGN/QSO loading that transfers gas into SHARK's lost reservoir.",
            "piecewise smooth under fixed forcing",
            "mimic_jax.shark.prescriptions.agn.lagos23_qso_outflow_loadings",
            ("cold_gas",),
            ("lost_gas",),
            True,
        ),
        ProcessMetadata(
            "reincorporation",
            "Reincorporation",
            "flow",
            "Ejected gas returned continuously to the hot halo.",
            "piecewise smooth",
            "mimic_jax.shark.flows.reincorporation_flow_derivative",
            ("ejected_gas",),
            ("hot_halo_gas",),
            True,
        ),
        ProcessMetadata(
            "halo_infall",
            "Halo infall",
            "forcing",
            "Finite cosmological supply and baryon-cap update in the SHARK reference interval.",
            "finite constrained map",
            "mimic_jax.shark.hybrid.apply_cosmological_infall",
        ),
        ProcessMetadata(
            "agn_heating_memory",
            "AGN heating memory",
            "projection",
            "Heating radius and excess jet-power history promoted in the augmented hybrid state where faithful.",
            "thresholded hybrid state",
            "mimic_jax.shark.prescriptions.agn.project_lagos23_heating_radius",
        ),
        ProcessMetadata(
            "disk_instability",
            "Disk instability",
            "event",
            "Threshold-triggered disk-to-bulge transfer, burst, and black-hole update.",
            "thresholded event map",
            "mimic_jax.shark.hybrid.apply_disk_instability_event",
        ),
        ProcessMetadata(
            "merger",
            "Galaxy merger",
            "event",
            "Topology-changing galaxy combination, burst, and black-hole update.",
            "discrete event map",
            "mimic_jax.shark.hybrid.apply_galaxy_merger_event",
        ),
    )
    metadata = SamModelMetadata(
        name="shark",
        label="SHARK Lagos23",
        upstream_repository=SHARK_UPSTREAM_REPOSITORY,
        upstream_revision=SHARK_UPSTREAM_REVISION,
        formulation="controlled continuous Lagos23 disk subset",
        qualification="Nonlinear disk/SN rates with fixed structural and prepared cooling/AGN forcing; full hybrid intervals remain model-owned.",
        time_unit="Gyr",
        time_unit_in_gyr=1.0,
        state_variables=state_variables,
        forcing_variables=(
            VariableMetadata(
                "disk.cooling_rate",
                "cooling rate",
                "Msun/h/Gyr",
                "forcing",
                "Prepared cooling supply held fixed over the local interval.",
            ),
            VariableMetadata(
                "disk.gas_half_mass_radius",
                "gas half-mass radius",
                "Mpc/h",
                "forcing",
                "Disk structural forcing used by BR06 star formation.",
            ),
            VariableMetadata(
                "disk.stellar_half_mass_radius",
                "stellar half-mass radius",
                "Mpc/h",
                "forcing",
                "Stellar structural forcing used by BR06 star formation.",
            ),
            VariableMetadata(
                "disk.galaxy_velocity",
                "galaxy velocity",
                "km/s",
                "forcing",
                "Circular-velocity proxy used by the disk prescriptions.",
            ),
            VariableMetadata(
                "reincorporation_rate",
                "reincorporation rate",
                "Msun/h/Gyr",
                "forcing",
                "Explicit ejected-to-hot flow for the continuous alternative.",
            ),
        ),
        parameter_variables=_leaf_parameter_metadata(parameters),
        observable_variables=(
            VariableMetadata(
                "stellar_mass",
                "stellar mass",
                "Msun/h",
                "observable",
                "Stellar mass in the controlled disk component.",
            ),
            VariableMetadata(
                "cold_gas",
                "cold-gas mass",
                "Msun/h",
                "observable",
                "Cold interstellar gas in the controlled disk component.",
            ),
            VariableMetadata(
                "hot_gas", "hot-halo gas mass", "Msun/h", "observable", "Hot halo gas reservoir."
            ),
            VariableMetadata(
                "ejected_gas",
                "ejected-gas mass",
                "Msun/h",
                "observable",
                "Stellar-feedback ejected reservoir.",
            ),
            VariableMetadata(
                "star_formation_rate",
                "star-formation rate",
                "Msun/h/Gyr",
                "observable",
                "Instantaneous BR06 disk star-formation rate.",
            ),
        ),
        processes=processes,
        capabilities=(
            Capability(
                "continuous_rhs",
                "partial",
                "Complete nonlinear disk/SN rate layer plus prepared cooling/QSO and reincorporation forcing.",
            ),
            Capability(
                "events",
                "model_specific",
                "Merger and instability maps remain explicit SHARK operators.",
            ),
            Capability(
                "constraints",
                "model_specific",
                "Infall, baryon caps, and heating projections remain in the hybrid interval.",
            ),
            Capability(
                "conservation",
                "supported",
                "Baryon, metal, and angular-momentum ledgers are explicit; QSO loss is an internal tracked reservoir.",
            ),
            Capability(
                "full_tree_physics_parity",
                "supported",
                "Every realized native full-tree continuous physics state has been replayed through the JAX kernel.",
            ),
            Capability(
                "independent_topology_driver",
                "unavailable",
                "Native SHARK still owns variable-cardinality topology and event scheduling; this is separate from evaluated physics parity.",
            ),
        ),
    )

    def rhs_and_rates(time, state, forcing, current_parameters, controls):
        epsilon = dict(zip(control_names, tuple(controls)))
        rates = lagos23_disk_flow_rates(
            time,
            state,
            forcing.disk,
            current_parameters.star_formation,
            current_parameters.stellar_feedback,
        )
        rates = rates._replace(
            cooling=rates.cooling * jnp.exp(epsilon["cooling"]),
            star_formation=rates.star_formation * jnp.exp(epsilon["star_formation"]),
            star_formation_angular_momentum=(
                rates.star_formation_angular_momentum * jnp.exp(epsilon["star_formation"])
            ),
            stellar_reheating_loading=(
                rates.stellar_reheating_loading * jnp.exp(epsilon["sn_reheating"])
            ),
            stellar_ejection_loading=(
                rates.stellar_ejection_loading * jnp.exp(epsilon["sn_ejection"])
            ),
            angular_momentum_reheating_loading=(
                rates.angular_momentum_reheating_loading * jnp.exp(epsilon["sn_reheating"])
            ),
            angular_momentum_ejection_loading=(
                rates.angular_momentum_ejection_loading * jnp.exp(epsilon["sn_ejection"])
            ),
            qso_reheating_loading=(rates.qso_reheating_loading * jnp.exp(epsilon["qso_reheating"])),
            qso_ejection_loading=(rates.qso_ejection_loading * jnp.exp(epsilon["qso_ejection"])),
        )
        reincorporation = forcing.reincorporation_rate * jnp.exp(epsilon["reincorporation"])
        derivative = shark_continuous_rhs_from_rates(
            time,
            state,
            rates,
            current_parameters.flow,
            reincorporation_rate=reincorporation,
        )
        metallicity = cold_gas_metallicity(state, current_parameters.flow)
        effective_yield = effective_stellar_yield(metallicity, current_parameters.flow)
        return SharkCommonRhsResult(
            derivative, rates, metallicity, effective_yield, reincorporation
        )

    def quantities(state):
        mass = sum(state[:6])
        metals = sum(state[6:12])
        angular_momentum = sum(state[14:19])
        return (
            ConservationQuantity(
                "baryons", mass, "Msun/h", "Mass in SHARK's six physical reservoirs."
            ),
            ConservationQuantity(
                "metals",
                metals,
                "Msun/h",
                "Tracked metal mass; stellar yield is an explicit source.",
                ("stellar yield",),
            ),
            ConservationQuantity(
                "angular_momentum",
                angular_momentum,
                "Msun Mpc km/s / h^2",
                "Angular momentum in five tracked baryonic components.",
            ),
        )

    def balances(result):
        derivative = result.derivative
        baryon = sum(derivative[:6])
        metal_rate = sum(derivative[6:12])
        source = result.effective_yield * result.rates.star_formation
        angular_momentum = sum(derivative[14:19])
        zero = jnp.zeros_like(source)
        return (
            ConservationBalance(
                "baryons",
                baryon,
                zero,
                zero,
                "Msun/h/Gyr",
                "Closed transfer residual including SHARK's tracked lost-gas reservoir.",
            ),
            ConservationBalance(
                "metals",
                metal_rate - source,
                source,
                zero,
                "Msun/h/Gyr",
                "Metal residual after the explicit stellar-yield source.",
            ),
            ConservationBalance(
                "angular_momentum",
                angular_momentum,
                zero,
                zero,
                "Msun Mpc km/s / h^2/Gyr",
                "Closed angular-momentum residual for the controlled flow subset.",
            ),
        )

    def rate_value(result, name):
        aliases = {
            "sn_reheating": "stellar_reheating_loading",
            "sn_ejection": "stellar_ejection_loading",
            "qso_reheating": "qso_reheating_loading",
            "qso_ejection": "qso_ejection_loading",
        }
        if name == "reincorporation":
            return result.reincorporation
        return getattr(result.rates, aliases.get(name, name))

    def observable_value(state, result, name):
        values = {
            "stellar_mass": state.stellar_mass,
            "cold_gas": state.cold_gas,
            "hot_gas": state.hot_halo_gas,
            "ejected_gas": state.ejected_gas,
            "star_formation_rate": result.rates.star_formation,
        }
        return values[name]

    return ConfiguredSamModel(
        metadata,
        parameters,
        rhs_and_rates,
        quantities,
        balances,
        rate_value,
        observable_value,
    )


def _sapphire_model(**options):
    from mimic_jax.sapphire import configured_sapphire

    return configured_sapphire(**options)


_MODEL_FACTORIES = {"sage16": _sage16_model, "shark": _shark_model, "sapphire": _sapphire_model}
_MODEL_ALIASES = {
    "sage": "sage16",
    "mimic_sage16": "sage16",
    "shark_lagos23": "shark",
    "saphire": "sapphire",
    "pandya23": "sapphire",
}


def available_models() -> Tuple[str, ...]:
    """Return stable registry keys for configured established SAMs."""

    return tuple(_MODEL_FACTORIES)


def load_model(name: str, **options):
    """Load one configured common SAM adapter without changing native APIs."""

    key = _MODEL_ALIASES.get(name.lower(), name.lower())
    try:
        factory = _MODEL_FACTORIES[key]
    except KeyError as error:
        raise KeyError(
            f"Unknown SAM {name!r}; available models are {available_models()}"
        ) from error
    return factory(**options)
