import numpy as np
from scipy.spatial.distance import cdist

import pylab as pl

from pybrain.datasets import ClassificationDataSet
from pybrain.utilities           import percentError
from sklearn import  datasets, cluster
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

if __name__ == "__main__":
    breast_cancer = datasets.load_breast_cancer()

    X, y = breast_cancer.data, breast_cancer.target

    # print type(X)
    # ds = ClassificationDataSet(30, nb_classes=2)
    # for k in xrange(len(X)):
    #     ds.addSample(X[k],y[k])
    #
    # tstdata, trndata = ds.splitWithProportion(0.3)
    # trndata._convertToOneOfMany()
    # tstdata._convertToOneOfMany()

    # length = X.shape[0]
    # km = cluster.KMeans(n_clusters=2)
    # labels = KMeans.fit(km, X[:length*.7])
    # print labels
    # labels = km.predict(X[length*.7:])
    # print labels

    clusters = range(1,10)
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

