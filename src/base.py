from typing import Dict, Any, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

cv = StratifiedKFold(n_splits=5, shuffle=True)


def get_base_classifier():
    return DecisionTreeClassifier()


def get_boosting_classifier():
    return AdaBoostClassifier(base_estimator=get_base_classifier())


def get_rfc():
    return RandomForestClassifier()


def get_classifier_scores(classifier, X, y, classifier_name: str, dataset_name: str, custom_columns: Dict[str, Any] = None):
    custom_columns = custom_columns or {}
    scores = cross_validate(classifier, X=X, y=y, scoring='f1_weighted', cv=cv, error_score='raise', n_jobs=-1, return_train_score=True)
    scores_df = pd.DataFrame.from_dict(scores)
    scores_df["classifier"] = classifier_name
    scores_df["dataset"] = dataset_name

    for k, v in custom_columns.items():
        scores_df[k] = v
    return scores_df


def plot_scores(scores: List[pd.DataFrame], x_key: str, suptitle: str):
    df = pd.concat(scores)
    datasets = df["dataset"].unique()
    fig, axs = plt.subplots(1, len(datasets), figsize=(len(datasets) * 8, 8))
    for dataset_index, dataset_name in enumerate(datasets):
        dataset_df = df[df["dataset"] == dataset_name]
        ax = axs[dataset_index]
        sns.boxplot(data=dataset_df, y="test_score", x=x_key, ax=ax).set_title(dataset_name)
    fig.suptitle(suptitle, fontsize=16)
    plt.show()
