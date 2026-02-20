import numpy as np


def sliding_window_with_labels(W, H, values, labels, step=1):
    """Create supervised windows.

    Past window length W (uses [t-W+1..t]) predicts whether there is any anomaly
    in the next H steps (uses [t+1..t+H]).

    Args:
        W: window length
        H: forecast horizon
        values: 1D array-like
        labels: 1D array-like 0/1
        step: stride

    Returns:
        X: (S, W) array
        y: (S,) array
    """
    X_windows = []
    y_windows = []

    x = np.asarray(values).reshape(-1, 1)
    y = np.asarray(labels).reshape(-1)

    m, _ = x.shape

    if y.ndim != 1 or len(y) != m:
        raise ValueError(f"labels must be 1D and have same length as values (got {len(y)} vs {m}).")
    if W <= 0 or H <= 0:
        raise ValueError("W and H must be positive integers.")
    if m < W + H:
        raise ValueError(f"Not enough data: need at least W+H={W + H} timesteps, got N={m}.")

    t = W - 1
    while t < m - H:
        X_t = x[t - W + 1 : t + 1]  # (W, 1)
        y_t = int(y[t + 1 : t + H + 1].max())
        X_windows.append(X_t)
        y_windows.append(y_t)
        t += step

    X_out = np.stack(X_windows, axis=0)  # (S, W, 1)
    y_out = np.asarray(y_windows, dtype=int)

    return X_out[:, :, 0], y_out


def sliding_window(W, values, step=1):
    """Create inference windows without labels.

    Returns X: (S, W)
    """
    X_windows = []
    x = np.asarray(values).reshape(-1, 1)
    m, _ = x.shape

    if W <= 0:
        raise ValueError("W must be positive integer.")
    if m < W:
        raise ValueError(f"Not enough data: need at least W={W} timesteps, got N={m}.")

    t = W - 1
    while t < m:
        X_t = x[t - W + 1 : t + 1]
        X_windows.append(X_t)
        t += step

    X_out = np.stack(X_windows, axis=0)
    return X_out[:, :, 0]


def reverse_sliding_window(W, H, y_predict_window, num_of_alarms=0):
    """Expand window-level predictions back to point-level labels.

    If a window prediction is 1 at index i, mark the next H points as anomalous.
    """
    y_predict_window = np.asarray(y_predict_window, dtype=int)
    y_predict = np.zeros_like(y_predict_window)

    for i in range(len(y_predict_window)):
        if y_predict_window[i] == 1:
            y_predict[i : i + H] = 1

    return y_predict
