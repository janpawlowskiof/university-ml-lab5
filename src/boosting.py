import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from src.base import get_classifier_scores


def get_boosting_n_estimators_df(X, y, dataset_name: str):
    scores = [
        get_classifier_scores(
            AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=n_estimators),
            X=X,
            y=y,
            classifier_name="boosting",
            dataset_name=dataset_name,
            custom_columns={"n_estimators": n_estimators}
        )
        for n_estimators in [2, 5, 10, 20, 100]
    ]
    return pd.concat(scores)


def get_boosting_lr_df(X, y, dataset_name: str):
    scores = [
        get_classifier_scores(
            AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), learning_rate=learning_rate),
            X=X,
            y=y,
            classifier_name="boosting",
            dataset_name=dataset_name,
            custom_columns={"learning_rate": learning_rate}
        )
        for learning_rate in [0.001, 0.01, 0.1, 1.0, 10.0]
    ]
    return pd.concat(scores)
