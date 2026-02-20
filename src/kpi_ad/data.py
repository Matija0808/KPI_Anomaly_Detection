import numpy as np
import pandas as pd


def load_kpi_dataset(path: str) -> pd.DataFrame:
    """Load KPI dataset CSV and sort by KPI ID and timestamp.

    Expected columns include: timestamp (unix seconds), value, label, KPI ID.
    """
    df = pd.read_csv(path)

    # Convert timestamps (Yahoo/KPI datasets often store unix seconds)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")

    # Sort safely
    sort_cols = [c for c in ["KPI ID", "timestamp"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    return df.reset_index(drop=True)


def split_by_kpi(df: pd.DataFrame) -> dict:
    """Split dataframe into a dict of {kpi_id: df_kpi_sorted_by_time}."""
    if "KPI ID" not in df.columns:
        raise ValueError("Expected column 'KPI ID' in dataframe.")

    out = {}
    for kpi_id, group in df.groupby("KPI ID"):
        if "timestamp" in group.columns:
            group = group.sort_values("timestamp")
        out[kpi_id] = group.reset_index(drop=True)
    return out


def get_anomaly_intervals(labels):
    """Return contiguous anomaly intervals from 0/1 labels.

    Returns:
        num_intervals (int), starts (np.ndarray), ends (np.ndarray)

    Note: end indices are inclusive.
    """
    y = np.asarray(labels, dtype=int)
    starts = []
    ends = []

    in_interval = False
    start_idx = None

    for i, v in enumerate(y):
        if v == 1 and not in_interval:
            in_interval = True
            start_idx = i
        elif v == 0 and in_interval:
            ends.append(i - 1)
            starts.append(start_idx)
            in_interval = False
            start_idx = None

    # Close interval if series ends with 1s
    if in_interval and start_idx is not None:
        starts.append(start_idx)
        ends.append(len(y) - 1)

    starts_arr = np.array(starts, dtype=int)
    ends_arr = np.array(ends, dtype=int)
    return len(starts_arr), starts_arr, ends_arr
