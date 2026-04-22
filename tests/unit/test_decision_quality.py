from __future__ import annotations

from meio.evaluation.decision_quality import compute_decision_quality
from meio.evaluation.logging_schema import StepTraceRecord


def test_compute_decision_quality_tracks_regime_accuracy_and_interventions() -> None:
    records = (
        StepTraceRecord(
            episode_id="episode_1",
            mode="llm_orchestrator",
            tool_ablation_variant="full",
            schedule_name="shift_recovery",
            run_seed=20260417,
            period_index=0,
            true_regime_label="normal",
            predicted_regime_label="normal",
            confidence=0.95,
            selected_subgoal="no_action",
            selected_tools=(),
            update_requests=("keep_current",),
            request_replan=False,
            abstain_or_no_action=True,
            demand_outlook=10.0,
            leadtime_outlook=2.0,
            scenario_adjustment_summary={
                "demand_outlook": 10.0,
                "leadtime_outlook": 2.0,
                "safety_buffer_scale": 1.0,
            },
            optimizer_orders=(0.0, 0.0, 0.0),
            inventory_by_echelon=(20.0, 30.0, 40.0),
            pipeline_by_echelon=(0.0, 0.0, 0.0),
            backorders_by_echelon=(0.0, 0.0, 0.0),
            per_period_cost=90.0,
            per_period_fill_rate=1.0,
            decision_changed_optimizer_input=False,
            optimizer_output_changed_state=True,
            intervention_changed_outcome=False,
        ),
        StepTraceRecord(
            episode_id="episode_1",
            mode="llm_orchestrator",
            tool_ablation_variant="full",
            schedule_name="shift_recovery",
            run_seed=20260417,
            period_index=1,
            true_regime_label="demand_regime_shift",
            predicted_regime_label="demand_regime_shift",
            confidence=0.86,
            selected_subgoal="request_replan",
            selected_tools=("forecast_tool", "leadtime_tool", "scenario_tool"),
            update_requests=("switch_demand_regime", "widen_uncertainty"),
            request_replan=True,
            abstain_or_no_action=False,
            demand_outlook=17.25,
            leadtime_outlook=2.0,
            scenario_adjustment_summary={
                "demand_outlook": 17.25,
                "leadtime_outlook": 2.0,
                "safety_buffer_scale": 1.2,
            },
            optimizer_orders=(30.0, 32.0, 43.0),
            inventory_by_echelon=(10.0, 30.0, 40.0),
            pipeline_by_echelon=(0.0, 0.0, 0.0),
            backorders_by_echelon=(0.0, 0.0, 0.0),
            per_period_cost=133.0,
            per_period_fill_rate=1.0,
            decision_changed_optimizer_input=True,
            optimizer_output_changed_state=True,
            intervention_changed_outcome=True,
        ),
        StepTraceRecord(
            episode_id="episode_1",
            mode="llm_orchestrator",
            tool_ablation_variant="full",
            schedule_name="shift_recovery",
            run_seed=20260417,
            period_index=2,
            true_regime_label="recovery",
            predicted_regime_label="normal",
            confidence=0.7,
            selected_subgoal="no_action",
            selected_tools=(),
            update_requests=("keep_current",),
            request_replan=False,
            abstain_or_no_action=True,
            demand_outlook=11.0,
            leadtime_outlook=2.0,
            scenario_adjustment_summary={
                "demand_outlook": 11.0,
                "leadtime_outlook": 2.0,
                "safety_buffer_scale": 1.0,
            },
            optimizer_orders=(8.0, 9.0, 0.0),
            inventory_by_echelon=(14.0, 24.0, 44.0),
            pipeline_by_echelon=(0.0, 0.0, 0.0),
            backorders_by_echelon=(0.0, 0.0, 0.0),
            per_period_cost=90.5,
            per_period_fill_rate=1.0,
            decision_changed_optimizer_input=False,
            optimizer_output_changed_state=True,
            intervention_changed_outcome=False,
        ),
    )

    summary = compute_decision_quality(records)

    assert summary.step_count == 3
    assert summary.regime_prediction_accuracy == 2 / 3
    assert dict(summary.predicted_regime_counts) == {
        "demand_regime_shift": 1,
        "normal": 2,
    }
    assert summary.no_action_rate == 2 / 3
    assert summary.replan_rate == 1 / 3
    assert summary.intervention_rate == 1 / 3
    assert summary.missed_intervention_count == 1
    assert summary.unnecessary_intervention_count == 0
    assert summary.average_confidence == (0.95 + 0.86 + 0.7) / 3
