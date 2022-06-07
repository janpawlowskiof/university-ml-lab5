import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.base import get_classifier_scores


def get_rfc_n_estimators_df(X, y, dataset_name: str):
    scores = [
        get_classifier_scores(
            RandomForestClassifier(n_estimators=n_estimators),
            X=X,
            y=y,
            classifier_name="bagging",
            dataset_name=dataset_name,
            custom_columns={"n_estimators": n_estimators}
        )
        for n_estimators in [2, 5, 10, 20, 100]
    ]
    return pd.concat(scores)


def get_rfc_n_samples_df(X, y, dataset_name: str):
    scores = [
        get_classifier_scores(RandomForestClassifier(max_samples=samples), X=X, y=y, classifier_name="bagging", dataset_name=dataset_name, custom_columns={"samples": samples})
        for samples in [0.05, 0.2, 0.5, 0.8, 1.0]
    ]
    return pd.concat(scores)


def get_rfc_n_features_df(X, y, dataset_name: str):
    scores = [
        get_classifier_scores(RandomForestClassifier(max_features=features), X=X, y=y, classifier_name="bagging", dataset_name=dataset_name, custom_columns={"features": features})
        for features in [0.05, 0.2, 0.5, 0.8, 1.0]
    ]
    return pd.concat(scores)


def get_rfc_depth_df(X, y, dataset_name: str):
    scores = [
        get_classifier_scores(RandomForestClassifier(max_depth=depth), X=X, y=y, classifier_name="bagging", dataset_name=dataset_name, custom_columns={"depth": str(depth)})
        for depth in [2, 5, 10, 100, None]
    ]
    return pd.concat(scores)
