from kpi_ad.data import get_anomaly_intervals


def test_get_anomaly_intervals_basic():
    y = [0, 1, 1, 0, 0, 1, 0]
    n, s, e = get_anomaly_intervals(y)
    assert n == 2
    assert s.tolist() == [1, 5]
    assert e.tolist() == [2, 5]


def test_get_anomaly_intervals_end_with_one():
    y = [0, 1, 1]
    n, s, e = get_anomaly_intervals(y)
    assert n == 1
    assert s.tolist() == [1]
    assert e.tolist() == [2]
