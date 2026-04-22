"""Stockpyl-backed serial benchmark adapter for the primary MEIO path."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import Any

from meio.config.schemas import BenchmarkConfig, SerialStageConfig
from meio.contracts import RegimeLabel


def stockpyl_available() -> bool:
    """Return whether Stockpyl appears importable in this environment."""

    return find_spec("stockpyl.supply_chain_network") is not None


def _require_stockpyl_serial_system() -> Any:
    """Return the Stockpyl serial-system builder or fail clearly."""

    try:
        from stockpyl.supply_chain_network import serial_system
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Stockpyl serial integration requires the installed stockpyl package."
        ) from exc
    return serial_system


def _coerce_regime_families(values: tuple[str, ...]) -> tuple[RegimeLabel, ...]:
    try:
        return tuple(RegimeLabel(value) for value in values)
    except ValueError as exc:
        raise ValueError("scenario_families must use supported RegimeLabel values.") from exc


def _resolve_stage_configs(config: BenchmarkConfig) -> tuple[SerialStageConfig, ...]:
    if config.system.stages:
        return tuple(config.system.stages)
    return tuple(
        SerialStageConfig(
            stage_index=index,
            stage_name=f"stage_{index}",
            initial_inventory=0,
            shipment_lead_time=2,
            base_stock_level=0,
        )
        for index in range(1, config.echelon_count + 1)
    )


@dataclass(frozen=True, slots=True)
class StockpylStageMetadata:
    """Typed metadata for one Stockpyl serial stage."""

    stage_index: int
    node_index: int
    stage_name: str
    initial_inventory: int
    base_stock_level: float
    shipment_lead_time: float
    local_holding_cost: float
    stockout_cost: float
    predecessor_stage_indices: tuple[int, ...] = field(default_factory=tuple)
    successor_stage_indices: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.stage_index <= 0:
            raise ValueError("stage_index must be positive.")
        if self.node_index <= 0:
            raise ValueError("node_index must be positive.")
        if not self.stage_name.strip():
            raise ValueError("stage_name must be a non-empty string.")
        for value in (
            self.initial_inventory,
            self.base_stock_level,
            self.shipment_lead_time,
            self.local_holding_cost,
            self.stockout_cost,
        ):
            if value < 0:
                raise ValueError("stage metadata numeric fields must be non-negative.")
        object.__setattr__(
            self,
            "predecessor_stage_indices",
            tuple(self.predecessor_stage_indices),
        )
        object.__setattr__(
            self,
            "successor_stage_indices",
            tuple(self.successor_stage_indices),
        )


@dataclass(frozen=True, slots=True)
class StockpylSerialInstance:
    """Typed wrapper around a Stockpyl serial benchmark instance."""

    benchmark_id: str
    topology: str
    echelon_count: int
    stockpyl_network: object
    stage_metadata: tuple[StockpylStageMetadata, ...]
    scenario_families: tuple[RegimeLabel, ...]
    demand_mean: float
    demand_standard_deviation: float
    evidence_history_window: int = 3
    adapter_name: str = "stockpyl_serial"
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.benchmark_id.strip():
            raise ValueError("benchmark_id must be a non-empty string.")
        if self.topology != "serial":
            raise ValueError("StockpylSerialInstance only supports serial topology.")
        if self.echelon_count <= 0:
            raise ValueError("echelon_count must be positive.")
        if self.evidence_history_window <= 0:
            raise ValueError("evidence_history_window must be positive.")
        object.__setattr__(self, "stage_metadata", tuple(self.stage_metadata))
        object.__setattr__(self, "scenario_families", tuple(self.scenario_families))
        object.__setattr__(self, "notes", tuple(self.notes))
        if len(self.stage_metadata) != self.echelon_count:
            raise ValueError("stage_metadata must match echelon_count.")
        if self.demand_mean < 0.0 or self.demand_standard_deviation < 0.0:
            raise ValueError("demand summary values must be non-negative.")
        for family in self.scenario_families:
            if not isinstance(family, RegimeLabel):
                raise TypeError("scenario_families must contain RegimeLabel values.")
        for note in self.notes:
            if not note.strip():
                raise ValueError("notes must contain non-empty strings.")

    @property
    def stage_names(self) -> tuple[str, ...]:
        """Return stage names in MEIO stage-index order."""

        return tuple(stage.stage_name for stage in self.stage_metadata)

    @property
    def initial_inventory(self) -> tuple[int, ...]:
        """Return initial inventory in MEIO stage-index order."""

        return tuple(stage.initial_inventory for stage in self.stage_metadata)

    @property
    def base_stock_levels(self) -> tuple[float, ...]:
        """Return base-stock references in MEIO stage-index order."""

        return tuple(stage.base_stock_level for stage in self.stage_metadata)

    @property
    def shipment_lead_times(self) -> tuple[int, ...]:
        """Return integer shipment lead times in MEIO stage-index order."""

        return tuple(
            max(1, int(round(stage.shipment_lead_time)))
            for stage in self.stage_metadata
        )

    @property
    def holding_costs(self) -> tuple[float, ...]:
        """Return local holding costs in MEIO stage-index order."""

        return tuple(stage.local_holding_cost for stage in self.stage_metadata)

    @property
    def stockout_costs(self) -> tuple[float, ...]:
        """Return stockout costs in MEIO stage-index order."""

        return tuple(stage.stockout_cost for stage in self.stage_metadata)

    @property
    def downstream_stage(self) -> StockpylStageMetadata:
        """Return the downstream-most stage metadata."""

        return self.stage_metadata[0]

    @property
    def primary_inbound_stage_index(self) -> int:
        """Return the stage feeding the downstream-most node."""

        if self.downstream_stage.predecessor_stage_indices:
            return self.downstream_stage.predecessor_stage_indices[0]
        return self.downstream_stage.stage_index

    @property
    def primary_inbound_lead_time(self) -> float:
        """Return the primary inbound lead time for the downstream stage."""

        return self.downstream_stage.shipment_lead_time


@dataclass(frozen=True, slots=True)
class StockpylSerialAdapter:
    """Primary concrete adapter for the kept Stockpyl serial benchmark path."""

    evidence_history_window: int = 3
    adapter_name: str = "stockpyl_serial"

    def build_instance(self, config: BenchmarkConfig) -> StockpylSerialInstance:
        """Build a Stockpyl-backed serial instance from the MEIO benchmark config."""

        if config.topology != "serial":
            raise ValueError("StockpylSerialAdapter only supports serial topology.")
        serial_system = _require_stockpyl_serial_system()
        stages = _resolve_stage_configs(config)
        stage_indices = tuple(stage.stage_index for stage in stages)
        node_order_in_system = list(reversed(stage_indices))
        node_order_in_lists = list(stage_indices)
        network = serial_system(
            config.echelon_count,
            node_order_in_system=node_order_in_system,
            node_order_in_lists=node_order_in_lists,
            local_holding_cost=[config.costs.holding_cost for _ in stages],
            stockout_cost=config.costs.backorder_cost,
            shipment_lead_time=[stage.shipment_lead_time for stage in stages],
            initial_inventory_level=[stage.initial_inventory for stage in stages],
            policy_type="BS",
            base_stock_level=[
                stage.base_stock_level
                if stage.base_stock_level is not None
                else stage.initial_inventory
                for stage in stages
            ],
            demand_type="P",
            mean=config.demand_mean,
        )
        for stage in stages:
            network.nodes_by_index[stage.stage_index].name = stage.stage_name

        stage_metadata = tuple(
            self._build_stage_metadata(network=network, stage=stage, config=config)
            for stage in stages
        )
        downstream_node = network.nodes_by_index[stages[0].stage_index]
        demand_source = downstream_node.demand_source
        demand_mean = float(getattr(demand_source, "mean", config.demand_mean) or 0.0)
        demand_standard_deviation = float(
            getattr(demand_source, "standard_deviation", 0.0) or 0.0
        )
        return StockpylSerialInstance(
            benchmark_id=f"{config.topology}_{config.echelon_count}_echelon",
            topology=config.topology,
            echelon_count=config.echelon_count,
            stockpyl_network=network,
            stage_metadata=stage_metadata,
            scenario_families=_coerce_regime_families(config.scenario_families),
            demand_mean=demand_mean,
            demand_standard_deviation=demand_standard_deviation,
            evidence_history_window=self.evidence_history_window,
            adapter_name=self.adapter_name,
            notes=(
                "Stockpyl serial network constructed for typed benchmark integration.",
            ),
        )

    def _build_stage_metadata(
        self,
        network: object,
        stage: SerialStageConfig,
        config: BenchmarkConfig,
    ) -> StockpylStageMetadata:
        node = network.nodes_by_index[stage.stage_index]
        policy = node.inventory_policy
        base_stock_level = getattr(policy, "base_stock_level", None)
        if base_stock_level is None:
            base_stock_level = (
                stage.base_stock_level
                if stage.base_stock_level is not None
                else stage.initial_inventory
            )
        return StockpylStageMetadata(
            stage_index=stage.stage_index,
            node_index=node.index,
            stage_name=stage.stage_name,
            initial_inventory=int(getattr(node, "initial_inventory_level", stage.initial_inventory)),
            base_stock_level=float(base_stock_level),
            shipment_lead_time=float(getattr(node, "shipment_lead_time", stage.shipment_lead_time)),
            local_holding_cost=float(
                getattr(node, "local_holding_cost", config.costs.holding_cost)
                or config.costs.holding_cost
            ),
            stockout_cost=float(
                getattr(node, "stockout_cost", config.costs.backorder_cost)
                or 0.0
            ),
            predecessor_stage_indices=tuple(int(value) for value in node.predecessor_indices()),
            successor_stage_indices=tuple(int(value) for value in node.successor_indices()),
        )


__all__ = [
    "StockpylSerialAdapter",
    "StockpylSerialInstance",
    "StockpylStageMetadata",
    "stockpyl_available",
]
