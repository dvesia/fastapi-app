"""
Train model procedure
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump
import utils


def train_test_model():
    """
    Execute model training
    """
    os.makedirs("artifacts/models", exist_ok=True)

    df = pd.read_csv("data/prepared/census.csv")
    train, _ = train_test_split(df, test_size=0.20)
    X_train, y_train, encoder, lb = utils.process_data(
        train, categorical_features= utils.get_cat_features(),
        label="salary", training=True
    )
    trained_model = utils.train_model(X_train, y_train)

    dump(trained_model, "artifacts/models/model.joblib")
    dump(encoder, "artifacts/models/encoder.joblib")
    dump(lb, "artifacts/models/lb.joblib")


if __name__ == "__main__":
    train_test_model()
