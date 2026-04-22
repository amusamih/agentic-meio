"""Lightweight benchmark-adapter boundaries for qualification work."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.util import find_spec

from meio.data.stockpyl_adapter import StockpylSerialAdapter


def _module_available(module_name: str) -> bool:
    """Return whether a Python module appears importable in this environment."""

    return find_spec(module_name) is not None


@dataclass(frozen=True, slots=True)
class BenchmarkAdapterStatus:
    """Availability and integration status for one benchmark candidate."""

    candidate_id: str
    candidate_name: str
    topology_style: str
    smoke_testable_now: bool
    available_modules: tuple[str, ...] = field(default_factory=tuple)
    missing_modules: tuple[str, ...] = field(default_factory=tuple)
    integration_work_remaining: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for value in (
            self.candidate_id,
            self.candidate_name,
            self.topology_style,
        ):
            if not value.strip():
                raise ValueError("candidate identifiers and topology_style must be non-empty.")
        object.__setattr__(self, "available_modules", tuple(self.available_modules))
        object.__setattr__(self, "missing_modules", tuple(self.missing_modules))
        object.__setattr__(
            self,
            "integration_work_remaining",
            tuple(self.integration_work_remaining),
        )
        object.__setattr__(self, "notes", tuple(self.notes))
        for value in (
            self.available_modules
            + self.missing_modules
            + self.integration_work_remaining
            + self.notes
        ):
            if not value.strip():
                raise ValueError("status tuple fields must contain non-empty strings.")


@dataclass(frozen=True, slots=True)
class BenchmarkAdapterBoundary:
    """Static benchmark-adapter boundary used for qualification only."""

    candidate_id: str
    candidate_name: str
    topology_style: str
    required_modules: tuple[str, ...] = field(default_factory=tuple)
    related_modules: tuple[str, ...] = field(default_factory=tuple)
    smoke_testable_when_available: bool = False
    integration_work_remaining: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for value in (self.candidate_id, self.candidate_name, self.topology_style):
            if not value.strip():
                raise ValueError("adapter identifiers and topology_style must be non-empty.")
        object.__setattr__(self, "required_modules", tuple(self.required_modules))
        object.__setattr__(self, "related_modules", tuple(self.related_modules))
        object.__setattr__(
            self,
            "integration_work_remaining",
            tuple(self.integration_work_remaining),
        )
        object.__setattr__(self, "notes", tuple(self.notes))

    def describe(self) -> BenchmarkAdapterStatus:
        """Describe current availability without executing benchmark logic."""

        available_modules = tuple(
            module_name for module_name in self.all_modules if _module_available(module_name)
        )
        missing_modules = tuple(
            module_name for module_name in self.all_modules if module_name not in available_modules
        )
        smoke_testable_now = (
            self.smoke_testable_when_available
            and all(module_name in available_modules for module_name in self.required_modules)
        )
        return BenchmarkAdapterStatus(
            candidate_id=self.candidate_id,
            candidate_name=self.candidate_name,
            topology_style=self.topology_style,
            smoke_testable_now=smoke_testable_now,
            available_modules=available_modules,
            missing_modules=missing_modules,
            integration_work_remaining=self.integration_work_remaining,
            notes=self.notes,
        )

    @property
    def all_modules(self) -> tuple[str, ...]:
        """Return the ordered module checks used for qualification."""

        seen: list[str] = []
        for module_name in self.required_modules + self.related_modules:
            if module_name not in seen:
                seen.append(module_name)
        return tuple(seen)


@dataclass(frozen=True, slots=True)
class StockpylSerialAdapterBoundary(BenchmarkAdapterBoundary):
    """Qualification wrapper for the Stockpyl serial candidate."""

    candidate_id: str = "stockpyl_serial"
    candidate_name: str = "Stockpyl serial"
    topology_style: str = "canonical_serial_multi_echelon"
    required_modules: tuple[str, ...] = ("stockpyl",)
    related_modules: tuple[str, ...] = ()
    smoke_testable_when_available: bool = True
    integration_work_remaining: tuple[str, ...] = (
        "Extend the current Stockpyl wrapper from initial typed state construction into controlled benchmark rollouts.",
        "Add typed cost and service summaries beyond the current placeholder evaluation fields.",
        "Build the first controlled comparison baseline on the active Stockpyl path.",
    )
    notes: tuple[str, ...] = (
        "Active primary integration path for the canonical serial benchmark.",
    )


@dataclass(frozen=True, slots=True)
class OrGymInventoryAdapterBoundary(BenchmarkAdapterBoundary):
    """Qualification wrapper for the OR-Gym inventory candidate."""

    candidate_id: str = "or_gym_inventory"
    candidate_name: str = "OR-Gym inventory"
    topology_style: str = "inventory_environment_family"
    required_modules: tuple[str, ...] = ("or_gym",)
    related_modules: tuple[str, ...] = ("or_gym_inventory",)
    smoke_testable_when_available: bool = False
    integration_work_remaining: tuple[str, ...] = (
        "Confirm the exact environment package split and stable environment entry points.",
        "Bridge the environment step loop into typed SimulationState, Observation, and RuntimeEvidence wrappers.",
        "Check whether the environment action interface can preserve optimizer-only order authority cleanly.",
    )
    notes: tuple[str, ...] = (
        "Treat as a secondary candidate until the canonical serial path is stable.",
    )


@dataclass(frozen=True, slots=True)
class MabimAdapterBoundary(BenchmarkAdapterBoundary):
    """Qualification wrapper for the MABIM candidate."""

    candidate_id: str = "mabim"
    candidate_name: str = "MABIM"
    topology_style: str = "unverified_external_inventory_benchmark"
    required_modules: tuple[str, ...] = ("mabim",)
    related_modules: tuple[str, ...] = ()
    smoke_testable_when_available: bool = False
    integration_work_remaining: tuple[str, ...] = (
        "Verify package availability, API shape, and access requirements.",
        "Check fit to the single-orchestration-agent and optimizer-boundary framing before integration work.",
        "Design a typed wrapper only if the benchmark earns a place in the paper shortlist.",
    )
    notes: tuple[str, ...] = (
        "Current repository-fit is not yet demonstrated.",
    )


def available_benchmark_adapters() -> tuple[BenchmarkAdapterBoundary, ...]:
    """Return the benchmark candidates currently tracked by the repo."""

    return (
        StockpylSerialAdapterBoundary(),
        OrGymInventoryAdapterBoundary(),
        MabimAdapterBoundary(),
    )


def primary_benchmark_adapter() -> StockpylSerialAdapter:
    """Return the active primary benchmark adapter."""

    return StockpylSerialAdapter()


__all__ = [
    "BenchmarkAdapterBoundary",
    "BenchmarkAdapterStatus",
    "MabimAdapterBoundary",
    "OrGymInventoryAdapterBoundary",
    "StockpylSerialAdapterBoundary",
    "available_benchmark_adapters",
    "primary_benchmark_adapter",
]
