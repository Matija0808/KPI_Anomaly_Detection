import os
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from TimeSeries import TimeSeriesModel
from load_functions import load_kpi_dataset, split_by_kpi, get_anomaly_intervals
from sliding_window_functions import sliding_window, sliding_window_with_labels, reverse_sliding_window
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':

    print("TRAINING...\n")

    # LOAD TRAIN DATA
    data = load_kpi_dataset("train.csv")
    data_split_by_KPI_ID = split_by_kpi(data)

    # GET FIRST KPI
    first_id_train, df_train = next(iter(data_split_by_KPI_ID.items()))

    values = df_train["value"].values
    labels = df_train["label"].values.astype(int)

    # SPLIT TRAIN, TEST
    train_end = int(0.8 * len(values))

    values_train = values[:train_end]
    values_test = values[train_end:]

    labels_train = labels[:train_end]
    labels_test = labels[train_end:]

    print(f"Number of clas 1 in train = {np.sum(labels_train)}")
    print(f"Number of clas 1 in test = {np.sum(labels_test)}")


    # PLOT TRAIN DATA
    plt.figure(figsize=(12, 4))

    # plot full series
    plt.plot(values_train, color="blue", linewidth=1, label="value")

    # get anomaly intervals
    num_of_intervals, intervals_start, intervals_end = get_anomaly_intervals(labels_train)
    print(f"Number of anomaly intervals in train set is {num_of_intervals}")
    # highlight anomaly start and end
    for i in intervals_start:
        plt.axvline(i, color="red", label="Anomaly start")
    for i in intervals_end:
        plt.axvline(i, color="green", label="Anomaly end")

    # shade anomaly regions
    # t = np.arange(len(values_train))
    # plt.fill_between(
    #     t,
    #     values_train.min(),
    #     values_train.max(),
    #     where=(labels_train == 1),
    #     color="green",
    #     alpha=0.2,
    #     label="Anomaly"
    # )

    plt.title(f"Train data KPI ID :{first_id_train}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    # plt.show()

    # TRAIN MODEL
    H = 20
    W = 50

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",

    )

    # model = RandomForestClassifier(
    #     n_estimators=500,
    #     max_depth= 5,
    #     min_samples_split=10,
    #     min_samples_leaf=5,
    #     class_weight="balanced_subsample",
    #     random_state=42,
    #     n_jobs=-1
    # )

    tsm = TimeSeriesModel(model, W, H)
    tsm.fit(values_train, labels_train)

    # TEST MODEL
    labels_test_predict = tsm.predict(values_test)
    tsm.get_accuracy_metrics(labels_test[W-1:], labels_test_predict)

    # Get anomaly intervals for test data
    num_of_intervals, intervals_start, intervals_end = get_anomaly_intervals(labels_test[W-1:])
    print("TRUE")
    print(f"Number of anomaly intervals in test set is {num_of_intervals}")
    print(f"Intervals start :")
    for i in range(num_of_intervals):
        print(f"[{intervals_start[i]}, {intervals_end[i]}]")
    print("")

    # get anomaly intervals for test data
    num_of_intervals, intervals_start, intervals_end = get_anomaly_intervals(labels_test_predict)
    print("PREDICTION")
    print(f"Number of anomaly intervals in test set is {num_of_intervals}")
    print(f"Intervals start :")
    for i in range(num_of_intervals):
        print(f"[{intervals_start[i]}, {intervals_end[i]}]")




