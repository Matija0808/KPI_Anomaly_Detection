import pandas as pd
import numpy as np

def load_kpi_dataset(url):
    df = pd.read_csv(url)

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    # VERY IMPORTANT: sort by KPI then time
    df = df.sort_values(["KPI ID", "timestamp"]).reset_index(drop=True)

    return df

def split_by_kpi(df):
    kpi_dict = {}

    for kpi_id, group in df.groupby("KPI ID"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        kpi_dict[kpi_id] = group

    return kpi_dict

def get_anomaly_intervals(labels):
    labels = np.asarray(labels)
    num_of_intervals = 0
    interval_flag = 0
    intervals_start = []
    intervals_end = []
    for i in range(len(labels)):
        if ((labels[i] == 1) & (interval_flag == 0)):
            interval_flag = 1
            intervals_start.append(i)
        if ((labels[i] == 0) & (interval_flag == 1)):
                intervals_end.append(i)
                interval_flag = 0
                num_of_intervals += 1


    return num_of_intervals, np.stack(intervals_start), np.stack(intervals_end)

