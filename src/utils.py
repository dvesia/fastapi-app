import logging
from typing import List, Tuple, Any
import numpy as np
from numpy import mean, std
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import pandas as pd


def get_cat_features() -> List[str]:
    """Return feature categories"""
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]


def process_data(
    X: pd.DataFrame,
    categorical_features: List[str] = [],
    label: str = None,
    training: bool = True,
    encoder: OneHotEncoder = None,
    lb: LabelBinarizer = None
) -> Tuple[np.ndarray, np.ndarray, OneHotEncoder, LabelBinarizer]:
    """Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features
    and a label binarizer for the labels. This can be used in either training
    or inference/validation.

    Note: depending on the type of model used, you may want to add in
    functionality that scales the continuous data.

    Args:
        X (pd.DataFrame): Dataframe containing the features and label.
        categorical_features (List[str]): List containing the names of the
            categorical features (default=[]).
        label (str): Name of the label column in `X`. If None, then an empty
            array will be returned for y (default=None).
        training (bool): Indicator if training mode or inference/validation
            mode.
        encoder (OneHotEncoder): Trained sklearn OneHotEncoder, only used if
            training=False.
        lb (LabelBinarizer): Trained sklearn LabelBinarizer, only used if
            training=False.

    Returns:
        np.ndarray: Processed data.
        np.ndarray: Processed labels if labeled=True, otherwise empty
            np.ndarray.
        OneHotEncoder: Trained OneHotEncoder if training is True, otherwise
            returns the encoder passed in.
        LabelBinarizer: Trained LabelBinarizer if training is True, otherwise
            returns the binarizer passed in.
    """
    y = X[label] if label is not None else np.array([])
    X_categorical = X[categorical_features].values
    X_continuous = X.drop(columns=categorical_features)
    if training:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        except AttributeError:
            pass
    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> GradientBoostingClassifier:
    """Trains a machine learning model and returns it.

    Args:
        X_train (np.array): Training data.
        y_train (np.array): Labels.

    Returns:
        GradientBoostingClassifier: Trained machine learning model.
    """
    cv = KFold(n_splits=10, shuffle=True, random_state=1)
    model = GradientBoostingClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    scores = cross_val_score(model, X_train, y_train, scoring='accuracy',
                             cv=cv, n_jobs=-1)
    logging.info('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
    return model


def compute_model_metrics(y: np.array, preds: np.array) -> Tuple[float, float, float]:
    """
    Validates the trained machine learning model using precision, recall,
    and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta: float = fbeta_score(y, preds, beta=1, zero_division=1)
    precision: float = precision_score(y, preds, zero_division=1)
    recall: float = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model: Any, X: np.ndarray) -> np.ndarray:
    """ Run model inferences and return the predictions.

    Args:
        model (Any): Trained machine learning model.
        X (np.ndarray): Data used for prediction.

    Returns:
        np.ndarray: Predictions from the model.
    """
    y_preds = model.predict(X)
    return y_preds
