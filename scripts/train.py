import argparse

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from kpi_ad import TimeSeriesModel, get_anomaly_intervals, load_kpi_dataset, split_by_kpi


def build_sklearn_model(name: str):
    name = name.lower()
    if name in {"logreg", "logistic", "lr"}:
        return LogisticRegression(max_iter=2000, class_weight="balanced")
    if name in {"rf", "random_forest"}:
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model '{name}'. Use: logreg | rf")


def main():
    p = argparse.ArgumentParser(description="Train/evaluate anomaly model on a single KPI series.")
    p.add_argument("--train_csv", default="data/raw/train.csv")
    p.add_argument("--test_csv", default="data/raw/test.csv")
    p.add_argument("--kpi_id", default=None, help="KPI ID to use. If omitted, uses the first KPI in the file.")
    p.add_argument("--model", default="logreg", help="logreg or rf")
    p.add_argument("--W", type=int, default=50, help="Past window length")
    p.add_argument("--H", type=int, default=20, help="Future horizon")
    p.add_argument("--min_precision", type=float, default=0.7, help="Target minimum precision for threshold selection")
    args = p.parse_args()

    print("Loading train CSV...", args.train_csv)
    train_df = load_kpi_dataset(args.train_csv)
    train_dict = split_by_kpi(train_df)

    if args.kpi_id is None:
        kpi_id, df_train = next(iter(train_dict.items()))
    else:
        if args.kpi_id not in train_dict:
            raise KeyError(f"KPI ID '{args.kpi_id}' not found in train file. Available: {list(train_dict.keys())[:5]}...")
        kpi_id, df_train = args.kpi_id, train_dict[args.kpi_id]

    values = df_train["value"].to_numpy()
    labels = df_train["label"].to_numpy().astype(int)

    split = int(0.8 * len(values))
    values_train, values_holdout = values[:split], values[split:]
    labels_train, labels_holdout = labels[:split], labels[split:]

    print(f"Using KPI: {kpi_id}")
    print(f"Train points: {len(values_train)} | Holdout points: {len(values_holdout)}")
    print(f"Pos in train: {labels_train.sum()} ({labels_train.mean():.4f})")

    n_int, s, e = get_anomaly_intervals(labels_train)
    print(f"Anomaly intervals in train: {n_int}")

    base_model = build_sklearn_model(args.model)
    tsm = TimeSeriesModel(base_model, W=args.W, H=args.H, min_precision=args.min_precision)
    tsm.fit(values_train, labels_train)

    # Evaluate on holdout portion of the same KPI
    y_pred = tsm.predict(values_holdout)
    TimeSeriesModel.report(labels_holdout[args.W - 1 :], y_pred, title="HOLDOUT")

    # Optional: evaluate on separate test.csv if provided
    if args.test_csv:
        try:
            test_df = load_kpi_dataset(args.test_csv)
            test_dict = split_by_kpi(test_df)
            if kpi_id in test_dict:
                df_test = test_dict[kpi_id]
                v_test = df_test["value"].to_numpy()
                y_test = df_test["label"].to_numpy().astype(int)
                y_pred_test = tsm.predict(v_test)
                TimeSeriesModel.report(y_test[args.W - 1 :], y_pred_test, title="TEST")
            else:
                print(f"Note: KPI {kpi_id} not present in test file; skipping test eval.")
        except FileNotFoundError:
            print("Test CSV not found; skipping.")


if __name__ == "__main__":
    main()
