from sklearn import  cluster, datasets, metrics, decomposition, random_projection, lda
from clustertesters import ExpectationMaximizationTestCluster as emtc
from clustertesters import KMeansTestCluster as kmtc

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
import pandas as pd
import time



if __name__ == "__main__":
    breast_cancer = datasets.load_breast_cancer()
    X, y = breast_cancer.data, breast_cancer.target
    start = time.time()
    transformer = lda.LDA(n_components=3)
    X_new = transformer.fit_transform(X, y)
    print "Elapsed time: {}".format(time.time() - start)

    tester = kmtc.KMeansTestCluster(X_new, y, clusters=range(1,10), plot=True, targetcluster=3, stats=True)
    tester.run()

