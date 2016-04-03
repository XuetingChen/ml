
from sklearn import  datasets, metrics, decomposition
from clustertesters import ExpectationMaximizationTestCluster as emtc
from clustertesters import KMeansTestCluster as kmtc

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
import pandas as pd

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

    kurts = []
    for i in range (1, 17):
        ica = decomposition.FastICA(n_components=i, whiten=True)
        output =ica.fit_transform(X)
        kurt = np.average(kurtosis(output))
        kurts.append(kurt)

    """
    Plot Kurtosis for ICA
    """
    plt.plot(kurts)
    plt.xlabel('Dimension')
    plt.ylabel('Average Kurtosis')
    plt.title('Average Kurtosis vs Dimension')
    plt.show()

    ipca = decomposition.FastICA(n_components=14, whiten=True)
    print X
    X_new = ipca.fit_transform(X, y)
    print X_new
    print X.shape

    tester = kmtc.KMeansTestCluster(X, y, clusters=range(1,40), plot=True, targetcluster=26, stats=True)
    tester.run()

