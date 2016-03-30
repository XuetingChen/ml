import numpy as np
from scipy.spatial.distance import cdist

from sklearn import metrics
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
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

    clusters = range(1,40)
    meandist=[]
    scores=[]


    for k in clusters:
        model=KMeans(n_clusters=k)
        model.fit(X)
        clusassign = model.predict(X)
        meandist.append(sum(np.min(cdist(X, model.cluster_centers_, 'euclidean'), axis=1))/ X.shape[0])
        scores.append(model.score(X, y))

    """
    Plot average distance from observations from the cluster centroid
    to use the Elbow Method to identify number of clusters to choose
    """

    plt.plot(clusters, scores)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average distance')
    plt.title('Selecting k with the Elbow Method')
    plt.show()

