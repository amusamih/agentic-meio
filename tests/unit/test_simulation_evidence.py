from __future__ import annotations

import pytest

from meio.contracts import RegimeLabel
from meio.simulation.evidence import DemandEvidence, LeadTimeEvidence, RuntimeEvidence


def test_runtime_evidence_accepts_typed_components() -> None:
    runtime_evidence = RuntimeEvidence(
        time_index=0,
        demand=DemandEvidence(
            history=(10.0, 11.0, 12.0),
            latest_realization=(12.0,),
            stage_index=1,
        ),
        leadtime=LeadTimeEvidence(
            history=(2.0, 2.0, 3.0),
            latest_realization=(3.0,),
            upstream_stage_index=3,
            downstream_stage_index=2,
        ),
        scenario_families=(RegimeLabel.NORMAL, RegimeLabel.RECOVERY),
        demand_baseline_value=10.0,
        leadtime_baseline_value=2.0,
        notes=("typed_smoke_evidence",),
    )

    assert runtime_evidence.demand.latest_value == 12.0
    assert runtime_evidence.scenario_families[1] is RegimeLabel.RECOVERY
    assert runtime_evidence.demand_baseline_value == 10.0


def test_demand_evidence_rejects_negative_history_values() -> None:
    with pytest.raises(ValueError, match="history"):
        DemandEvidence(history=(10.0, -1.0))
