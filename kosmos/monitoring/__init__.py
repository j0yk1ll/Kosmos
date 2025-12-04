"""
Monitoring module for Kosmos AI Scientist.

Provides metrics collection, alerting, and observability.
"""

from kosmos.monitoring.alerts import (
    Alert,
    AlertManager,
    AlertRule,
    AlertSeverity,
    AlertStatus,
    evaluate_alerts,
    get_active_alerts,
    get_alert_history,
    get_alert_manager,
)
from kosmos.monitoring.metrics import (
    MetricsCollector,
    export_metrics,
    get_metrics_collector,
    get_metrics_content_type,
)


__all__ = [
    # Metrics
    "get_metrics_collector",
    "export_metrics",
    "get_metrics_content_type",
    "MetricsCollector",
    # Alerts
    "get_alert_manager",
    "evaluate_alerts",
    "get_active_alerts",
    "get_alert_history",
    "Alert",
    "AlertRule",
    "AlertSeverity",
    "AlertStatus",
    "AlertManager",
]
