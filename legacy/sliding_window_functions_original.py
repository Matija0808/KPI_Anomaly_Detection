import numpy as np

def sliding_window_with_labels(W, H, values, labels, step=1):
    X_windows = []
    y_windows = []

    x = np.asarray(values).reshape(-1, 1)
    y = np.asarray(labels).reshape(-1)

    m, n = x.shape

    if y.ndim != 1 or len(y) != m:
        raise ValueError(f"y_train must be 1D and have same length as x_train (got {len(y)} vs {m}).")
    if W <= 0 or H <= 0:
        raise ValueError("W and H must be positive integers.")
    if m < W + H:
        raise ValueError(f"Not enough data: need at least W+H={W + H} timesteps, got N={m}.")

    # t is the END index of the past window
    # Past window uses [t-W+1, ..., t]
    # Future horizon uses [t+1, ..., t+H]
    t = W - 1
    while(t < m-H):
        X_t = x[t - W + 1: t + 1]  # (W, n)
        # If there is any incident in next H stamps y_t will be 1
        y_t = int(y[t + 1: t + H + 1].max())  # scalar 0/1
        X_windows.append(X_t)
        y_windows.append(y_t)
        if (y_t == 0):
            t += step
        else:
            t += step

    # for t in range(W - 1, m - H, 5):
    #     X_t = x[t - W + 1: t + 1, :]  # (W, n)
    #     # If there is any incident in next H stamps y_t will be 1
    #     y_t = int(y[t + 1: t + H + 1].max())  # scalar 0/1
    #     X_windows.append(X_t)
    #     y_windows.append(y_t)

    X_out = np.stack(X_windows, axis=0)  # (S, W, n);
    y_out = np.asarray(y_windows, dtype=int)

    X_out = X_out[:, :, 0]
    print(f"X_train.shape= {X_out.shape} \n")

    return X_out, y_out

def sliding_window(W,values, step = 1):
    X_windows = []

    x = np.asarray(values).reshape(-1, 1)

    m, n = x.shape

    if W <= 0:
        raise ValueError("W must be positive integer.")

    t = W - 1
    while (t < m):
        X_t = x[t - W + 1: t + 1]  # (W, n)
        X_windows.append(X_t)
        t += step

    X_out = np.stack(X_windows, axis=0)  # (S, W, n);
    X_out = X_out[:, :, 0]

    return X_out

def reverse_sliding_window(W, H, y_predict_window, num_of_alarms):
    y_predict = np.zeros_like(y_predict_window)
    alarm_cnt = 0
    alarm_flag = 0
    for i in range(0, len(y_predict_window)):
            if (y_predict_window[i] == 1):
                y_predict[i:i+H] = 1
                alarm_flag = 1
                alarm_cnt += 1

            # if (y_predict_window == 0):
            #     alarm_flag = 0
            #     alarm_cnt = 0

            # if (alarm_cnt == num_of_alarms):
                # y_predict[i] = 1

    return y_predict
