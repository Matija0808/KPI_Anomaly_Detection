import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from .windows import reverse_sliding_window, sliding_window, sliding_window_with_labels


class TimeSeriesModel:
    """A simple wrapper around an sklearn classifier for time-series windows."""

    def __init__(self, model, W: int, H: int, min_precision: float = 0.7, random_state: int = 29):
        self.model = model
        self.H = int(H)
        self.W = int(W)
        self.min_precision = float(min_precision)
        self.random_state = int(random_state)
        self.scaler = None
        self.threshold = 0.5

    def choose_threshold_for_precision(self, y_true: np.ndarray, val_prob: np.ndarray) -> float:
        """Pick threshold maximizing recall while keeping precision >= min_precision."""
        thr = np.arange(0.001, 1.0, 0.001)
        prec = np.zeros_like(thr)
        rec = np.zeros_like(thr)

        for i, t in enumerate(thr):
            y_pred = (val_prob >= t).astype(int)
            prec[i] = precision_score(y_true, y_pred, zero_division=0)
            rec[i] = recall_score(y_true, y_pred, zero_division=0)

        ok = prec >= self.min_precision
        if not np.any(ok):
            # If impossible, fall back to the threshold that gives best precision
            return float(thr[int(np.argmax(prec))])

        best = int(np.argmax(rec[ok]))
        return float(thr[np.where(ok)[0][best]])

    def fit(self, values, labels):
        values = np.asarray(values).reshape(-1)
        labels = np.asarray(labels).reshape(-1).astype(int)

        X_all, y_all = sliding_window_with_labels(self.W, self.H, values, labels)

        # time-safe split for val: last 20% of windows
        split = int(0.8 * len(X_all))
        X_train, X_val = X_all[:split], X_all[split:]
        y_train, y_val = y_all[:split], y_all[split:]

        X_train, y_train = shuffle(X_train, y_train, random_state=self.random_state)

        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val)

        self.model.fit(X_train_s, y_train)

        val_prob = self.model.predict_proba(X_val_s)[:, 1]
        self.threshold = self.choose_threshold_for_precision(y_val, val_prob)

        y_pred = (val_prob >= self.threshold).astype(int)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        cm = confusion_matrix(y_val, y_pred)

        print("VALIDATION")
        print(f"  Train windows: X={X_train.shape}, pos={y_train.sum()} ({y_train.mean():.3f})")
        print(f"  Val windows:   X={X_val.shape},   pos={y_val.sum()} ({y_val.mean():.3f})")
        print(f"  Chosen threshold: {self.threshold:.3f} (min_precision={self.min_precision})")
        print(f"  Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f}")
        print(f"  Confusion matrix [[TN FP],[FN TP]]:\n{cm}\n")

        return self

    def predict(self, values):
        if self.scaler is None:
            raise RuntimeError("Call fit() before predict().")

        values = np.asarray(values).reshape(-1)
        X = sliding_window(self.W, values)
        X_s = self.scaler.transform(X)
        prob = self.model.predict_proba(X_s)[:, 1]
        y_window = (prob >= self.threshold).astype(int)
        return reverse_sliding_window(self.W, self.H, y_window)

    @staticmethod
    def report(y_true, y_pred, title="TEST"):
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        print(title)
        print(f"  y: {y_true.shape}, pos={y_true.sum()} ({y_true.mean():.3f})")
        print(f"  Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f}")
        print(f"  Confusion matrix [[TN FP],[FN TP]]:\n{cm}\n")
        return prec, rec, f1, cm
