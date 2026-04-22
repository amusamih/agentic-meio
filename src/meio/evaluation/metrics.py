"""Metric identifiers for the first MEIO milestone."""

from __future__ import annotations

from enum import StrEnum


class MetricName(StrEnum):
    """Supported metric identifiers for evaluation boundaries."""

    TOTAL_COST = "total_cost"
    FILL_RATE = "fill_rate"
    AVERAGE_INVENTORY = "average_inventory"
    BACKORDER_LEVEL = "backorder_level"
    INTERVENTION_FREQUENCY = "intervention_frequency"
    FALSE_ALARM_RATE = "false_alarm_rate"
    DETECTION_DELAY = "detection_delay"


DEFAULT_METRIC_NAMES: tuple[MetricName, ...] = tuple(MetricName)


def metric_names() -> tuple[str, ...]:
    """Return metric names as stable strings for reporting boundaries."""

    return tuple(metric.value for metric in DEFAULT_METRIC_NAMES)


__all__ = ["DEFAULT_METRIC_NAMES", "MetricName", "metric_names"]
