from sklearn import  datasets, metrics, decomposition, random_projection
from clustertesters import ExpectationMaximizationTestCluster as emtc
from clustertesters import KMeansTestCluster as kmtc

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
import pandas as pd
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
    breast_cancer = datasets.load_breast_cancer()

    X, y = breast_cancer.data, breast_cancer.target

    start = time.time()
    transformer = random_projection.SparseRandomProjection(n_components=2)
    X_new = transformer.fit_transform(X)
    print "Elapsed time: {}".format(time.time() - start)
    print X_new
    print X.shape

    tester = emtc.ExpectationMaximizationTestCluster(X, y, clusters=range(1,10), plot=True, targetcluster=3, stats=True)
    tester.run()

