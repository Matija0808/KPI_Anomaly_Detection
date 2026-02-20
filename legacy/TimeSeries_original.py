import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sliding_window_functions import sliding_window_with_labels, sliding_window, reverse_sliding_window

class TimeSeriesModel():

    def __init__(self, model, W, H):
        self.model = model
        self.H = H
        self.W = W


    def choose_treshold_by_precision(self, y_true: np.ndarray, val_prob: np.ndarray):
        thr = np.arange(0.001, 1, 0.001)
        prec = np.zeros_like(thr)
        rec = np.zeros_like(thr)
        for i in range(len(thr)):
            y_pred = (val_prob >= thr[i]).astype(int)
            # y_pred = reverse_sliding_window(self.W, self.H, pred, 0)
            prec[i] = precision_score(y_true, y_pred, zero_division=0)
            rec[i] = recall_score(y_true, y_pred, zero_division=0)

        idx = (prec >= 0.7)  # Minimum 80% of accuracy
        available_rec = rec[idx]
        available_thr = thr[idx]

        best = np.argmax(available_rec)  # Best possible Recall for that accuracy
        best_thr = available_thr[best]
        plt.figure()
        plt.scatter(prec, rec)
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.title('Precision/Recal curve for chosing thrashold')

        return best_thr

    def fit(self, values, labels):

        values = np.asarray(values).reshape(-1, 1)
        labels = np.asarray(labels)

        # sliding window
        X_train, y_train = sliding_window_with_labels(self.W, self.H, values, labels)

        train_end = int(0.8*len(values))
        X_val = X_train[train_end:, :]
        X_train = X_train[:train_end, :]
        y_val = y_train[train_end:]
        y_train = y_train[:train_end]

        # --- Shuffling ---
        X_train, y_train = shuffle(X_train, y_train, random_state=29)

        # --- Scaling ---
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        self.scaler = scaler

        # --- Training---
        self.model.fit(X_train_s, y_train)

        # --- probabilities ---
        val_prob = self.model.predict_proba(X_val_s)[:, 1]

        # --- threshold choosing on validation data ---
        thr = self.choose_treshold_by_precision(y_val, val_prob)
        self.threshold = thr

        # --- predictions + metrics ---
        y_pred = (val_prob >= thr).astype(int)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        cm = confusion_matrix(y_val, y_pred)

        print("VALIDATION \n")
        print(f"  Train: X={X_train.shape}, pos={y_train.sum()} ({y_train.mean():.3f})")
        print(f"  Val:   X={X_val.shape},   pos={y_val.sum()} ({y_val.mean():.3f})")
        print(f"  Best threshold (F1 on val): {thr:.3f}")
        print(f"  Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f}")
        print(f"  Confusion matrix [[TN FP],[FN TP]]:\n{cm}")
        print("")

    def predict(self, x_test):

        x_test = np.asarray(x_test).reshape(-1, 1)

        # sliding window
        x_test_sliding_window = sliding_window(self.W, x_test)
        print(f"X_test.shape= {x_test.shape} \n")

        # Scaling
        x_test_s = self.scaler.transform(x_test_sliding_window)

        # Prediction
        y_prob = self.model.predict_proba(x_test_s)[:, 1]

        # Apply threshold
        y_predict_window = (y_prob >= self.threshold).astype(int)

        # Find predicted anomaly intervals
        y_predict = reverse_sliding_window(self.W, self.H, y_predict_window, 0)

        return y_predict



    def get_accuracy_metrics(self, y_true, y_predict):
        prec = precision_score(y_true, y_predict, zero_division=0)
        rec = recall_score(y_true, y_predict, zero_division=0)
        f1 = f1_score(y_true, y_predict, zero_division=0)
        cm = confusion_matrix(y_true, y_predict)

        print("TEST \n")
        print(f"  Test: X={y_true.shape}, pos={y_true.sum()} ({y_true.mean():.3f})")
        print(f"  Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f}")
        print(f"  Confusion matrix [[TN FP],[FN TP]]:\n{cm}")
        print("")



