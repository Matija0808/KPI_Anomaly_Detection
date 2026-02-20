"""KPI anomaly detection project (JetBrains internship prep).

Modules:
- kpi_ad.data: loading/splitting helpers
- kpi_ad.windows: sliding window utilities
- kpi_ad.model: TimeSeriesModel wrapper
"""

from .data import load_kpi_dataset, split_by_kpi, get_anomaly_intervals
from .windows import sliding_window, sliding_window_with_labels, reverse_sliding_window
from .model import TimeSeriesModel

__all__ = [
    "load_kpi_dataset",
    "split_by_kpi",
    "get_anomaly_intervals",
    "sliding_window",
    "sliding_window_with_labels",
    "reverse_sliding_window",
    "TimeSeriesModel",
]
