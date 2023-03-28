"""
Check Score procedure
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import load
from utils import get_cat_features, compute_model_metrics, process_data
import logging


def check_score():
    """
    Execute score checking
    """
    df = pd.read_csv("data/prepared/census.csv")
    _, test = train_test_split(df, test_size=0.20)

    trained_model = load("artifacts/models/model.joblib")
    encoder = load("artifacts/models/encoder.joblib")
    lb = load("artifacts/models/label_binarizer.joblib")

    slice_values = []

    for cat in get_cat_features():
        for cls in test[cat].unique():
            df_temp = test[test[cat] == cls]

            X_test, y_test, _, _ = process_data(
                df_temp,
                categorical_features=get_cat_features(),
                label="salary", encoder=encoder, lb=lb, training=False)

            y_preds = trained_model.predict(X_test)

            prc, rcl, fb = compute_model_metrics(y_test, y_preds)

            line = "[%s->%s] Precision: %s " \
                   "Recall: %s FBeta: %s" % (cat, cls, prc, rcl, fb)
            logging.info(line)
            slice_values.append(line)

    with open('artifacts/slice_output.txt', 'w') as out:
        for slice_value in slice_values:
            out.write(slice_value + '\n')


if __name__ == "__main__":
    check_score()
