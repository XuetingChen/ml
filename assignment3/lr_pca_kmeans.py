
from sklearn import  datasets, metrics, decomposition
from clustertesters import ExpectationMaximizationTestCluster as emtc
from clustertesters import KMeansTestCluster as kmtc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


def encode_target(df, target_column):
    """Add column to df with integers for the target.

    Args
    ----
    df -- pandas DataFrame.
    target_column -- column to map to int, producing
                     new Target column.

    Returns
    -------
    df_mod -- modified DataFrame.
    targets -- list of target names.
    """
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod[target_column].replace(map_to_int, inplace=True)
    return (df_mod, map_to_int)

if __name__ == "__main__":
    letter_recognition = pd.read_csv("letter-recognition.csv")
    dft, mapping = encode_target(letter_recognition, "letter")

    X = (dft.ix[:, 1:])
    y = dft.ix[:, 0]

    start = time.time()
    pca = decomposition.pca.PCA()
    pca.fit(X)
    X_new = pca.fit_transform(X)
    print "Elapsed time: {}".format(time.time() - start)

    """
    Plot Variance Ratio for PCA
    """
    plt.bar(range(0, 16), pca.explained_variance_ratio_)
    plt.xlabel('Dimension')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio vs Dimension')
    plt.show()

    pca = decomposition.pca.PCA(n_components=14)
    X_new = pca.fit_transform(X)

    tester = kmtc.KMeansTestCluster(X_new, y, clusters=range(1,40), plot=True, targetcluster=26, stats=True)
    tester.run()

