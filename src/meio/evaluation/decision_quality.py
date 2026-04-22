"""Typed decision-quality summaries for bounded benchmark runs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from meio.evaluation.logging_schema import StepTraceRecord


@dataclass(frozen=True, slots=True)
class DecisionQualitySummary:
    """Compact regime-prediction and intervention-quality summary."""

    step_count: int
    regime_prediction_accuracy: float | None
    predicted_regime_counts: tuple[tuple[str, int], ...]
    confusion_counts: tuple[tuple[str, str, int], ...]
    no_action_rate: float
    replan_rate: float
    intervention_rate: float
    missed_intervention_count: int
    unnecessary_intervention_count: int
    average_confidence: float | None = None

    def __post_init__(self) -> None:
        if self.step_count < 0:
            raise ValueError("step_count must be non-negative.")
        if self.regime_prediction_accuracy is not None and not 0.0 <= self.regime_prediction_accuracy <= 1.0:
            raise ValueError("regime_prediction_accuracy must be within [0.0, 1.0] when provided.")
        for field_name in ("no_action_rate", "replan_rate", "intervention_rate"):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be within [0.0, 1.0].")
        for field_name in ("missed_intervention_count", "unnecessary_intervention_count"):
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative.")
        if self.average_confidence is not None and not 0.0 <= self.average_confidence <= 1.0:
            raise ValueError("average_confidence must be within [0.0, 1.0] when provided.")
        for label, count in self.predicted_regime_counts:
            if not label.strip():
                raise ValueError("predicted_regime_counts must use non-empty labels.")
            if count < 0:
                raise ValueError("predicted_regime_counts must use non-negative counts.")
        for true_label, predicted_label, count in self.confusion_counts:
            if not true_label.strip() or not predicted_label.strip():
                raise ValueError("confusion_counts must use non-empty labels.")
            if count < 0:
                raise ValueError("confusion_counts must use non-negative counts.")

    def to_record(self) -> dict[str, object]:
        """Return a compact JSON-serializable decision-quality record."""

        return {
            "step_count": self.step_count,
            "regime_prediction_accuracy": self.regime_prediction_accuracy,
            "predicted_regime_counts": {
                label: count for label, count in self.predicted_regime_counts
            },
            "confusion_counts": [
                {
                    "true_regime_label": true_label,
                    "predicted_regime_label": predicted_label,
                    "count": count,
                }
                for true_label, predicted_label, count in self.confusion_counts
            ],
            "no_action_rate": self.no_action_rate,
            "replan_rate": self.replan_rate,
            "intervention_rate": self.intervention_rate,
            "missed_intervention_count": self.missed_intervention_count,
            "unnecessary_intervention_count": self.unnecessary_intervention_count,
            "average_confidence": self.average_confidence,
        }


def compute_decision_quality(
    step_trace_records: tuple[StepTraceRecord, ...],
) -> DecisionQualitySummary:
    """Compute compact decision-quality metrics from step trace records."""

    step_count = len(step_trace_records)
    if step_count == 0:
        return DecisionQualitySummary(
            step_count=0,
            regime_prediction_accuracy=None,
            predicted_regime_counts=(),
            confusion_counts=(),
            no_action_rate=0.0,
            replan_rate=0.0,
            intervention_rate=0.0,
            missed_intervention_count=0,
            unnecessary_intervention_count=0,
            average_confidence=None,
        )
    predicted_records = tuple(
        record for record in step_trace_records if record.predicted_regime_label is not None
    )
    predicted_counts = Counter(
        record.predicted_regime_label for record in predicted_records if record.predicted_regime_label is not None
    )
    confusion_counts = Counter(
        (record.true_regime_label, record.predicted_regime_label)
        for record in predicted_records
        if record.predicted_regime_label is not None
    )
    correct_predictions = sum(
        1
        for record in predicted_records
        if record.predicted_regime_label == record.true_regime_label
    )
    regime_prediction_accuracy = None
    if predicted_records:
        regime_prediction_accuracy = correct_predictions / len(predicted_records)
    no_action_count = sum(
        1
        for record in step_trace_records
        if record.selected_subgoal == "no_action"
    )
    replan_count = sum(1 for record in step_trace_records if record.request_replan)
    intervention_count = sum(1 for record in step_trace_records if _is_intervention(record))
    missed_intervention_count = sum(
        1
        for record in step_trace_records
        if record.true_regime_label != "normal" and not _is_intervention(record)
    )
    unnecessary_intervention_count = sum(
        1
        for record in step_trace_records
        if record.true_regime_label == "normal" and _is_intervention(record)
    )
    confidence_values = tuple(
        record.confidence for record in step_trace_records if record.confidence is not None
    )
    average_confidence = None
    if confidence_values:
        average_confidence = sum(confidence_values) / len(confidence_values)
    return DecisionQualitySummary(
        step_count=step_count,
        regime_prediction_accuracy=regime_prediction_accuracy,
        predicted_regime_counts=tuple(sorted(predicted_counts.items())),
        confusion_counts=tuple(
            sorted(
                (
                    true_label,
                    predicted_label,
                    count,
                )
                for (true_label, predicted_label), count in confusion_counts.items()
            )
        ),
        no_action_rate=no_action_count / step_count,
        replan_rate=replan_count / step_count,
        intervention_rate=intervention_count / step_count,
        missed_intervention_count=missed_intervention_count,
        unnecessary_intervention_count=unnecessary_intervention_count,
        average_confidence=average_confidence,
    )


def _is_intervention(record: StepTraceRecord) -> bool:
    if record.request_replan or record.selected_tools:
        return True
    return any(update_request != "keep_current" for update_request in record.update_requests)


__all__ = ["DecisionQualitySummary", "compute_decision_quality"]
