import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

cv = StratifiedKFold(n_splits=5, shuffle=True)


def get_base_classifier():
    return DecisionTreeClassifier()


def get_classifier_scores(classifier, X, y, classifier_name: str, dataset_name: str):
    scores = cross_validate(classifier, X=X, y=y, scoring='f1_weighted', cv=cv, error_score='raise', n_jobs=-1, return_train_score=True)
    scores_df = pd.DataFrame.from_dict(scores)
    scores_df["classifier"] = classifier_name
    scores_df["dataset"] = dataset_name
    return scores_df


def plot_scores(df: pd.DataFrame):
    datasets = df["dataset"].unique()
    # fig, axs = plt.subplots(1, len(datasets), figsize=(8, len(datasets) * 8))
    # for dataset_index, dataset_name in enumerate(datasets):
        # dataset_df = df[df["dataset"] == dataset_name]
        # ax = axs[dataset_index]
        # sns.boxplot(data=dataset_df, y="test_score", x="classifier", ax=ax)
    sns.boxplot(data=df, y="test_score", x="classifier")
