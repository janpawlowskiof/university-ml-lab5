import pandas as pd
from sklearn.ensemble import BaggingClassifier

from src.base import get_classifier_scores


def get_bagging_n_estimators_df(X, y, dataset_name: str):
    scores = [
        get_classifier_scores(
            BaggingClassifier(n_estimators=n_estimators),
            X=X,
            y=y,
            classifier_name="bagging",
            dataset_name=dataset_name,
            custom_columns={"n_estimators": n_estimators}
        )
        for n_estimators in [2, 5, 10, 20, 100]
    ]
    return pd.concat(scores)


def get_bagging_n_samples_df(X, y, dataset_name: str):
    scores = [
        get_classifier_scores(BaggingClassifier(max_samples=samples), X=X, y=y, classifier_name="bagging", dataset_name=dataset_name, custom_columns={"samples": samples})
        for samples in [0.05, 0.2, 0.5, 0.8, 1.0]
    ]
    return pd.concat(scores)


def get_bagging_n_features_df(X, y, dataset_name: str):
    scores = [
        get_classifier_scores(BaggingClassifier(max_features=features), X=X, y=y, classifier_name="bagging", dataset_name=dataset_name, custom_columns={"features": features})
        for features in [0.05, 0.2, 0.5, 0.8, 1.0]
    ]
    return pd.concat(scores)


def get_bagging_bootstrap_df(X, y, dataset_name: str):
    scores = [
        get_classifier_scores(BaggingClassifier(bootstrap=bootstrap), X=X, y=y, classifier_name="bagging", dataset_name=dataset_name, custom_columns={"bootstrap": bootstrap})
        for bootstrap in [True, False]
    ]
    return pd.concat(scores)
