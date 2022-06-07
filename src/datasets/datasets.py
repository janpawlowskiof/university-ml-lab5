from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_wine():
    wine = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
        names=["class", "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
               "color_intensity", "hue", "ods", "proline"]
    )
    X = wine.drop(["class"], axis="columns")
    y = wine["class"]
    return X, y


def load_glass():
    glass = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data",
        names=["id", "ri", "na", "mg", "al", "si", "k", "ca", "ba", "fe", "class"],
        index_col=0
    )

    X = glass.drop(["class"], axis="columns")
    y = glass["class"]
    return X, y


def load_seeds():
    seeds_path = Path(__file__).parent / "seeds_dataset.txt"
    seeds = pd.read_csv(
        str(seeds_path),
        names=["area", "perimeter", "compactness", "length_of_kernel", "width_of_kernel", "asymmetry", "length_of_kernel_groove", "class"],
        sep="\t"
    )

    X = seeds.drop(["class"], axis="columns")
    y = seeds["class"]
    return X, y

