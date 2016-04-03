
from sklearn import  datasets, metrics, decomposition, cluster
from clustertesters import ExpectationMaximizationTestCluster as emtc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

if __name__ == "__main__":
    breast_cancer = datasets.load_breast_cancer()

    X, y = breast_cancer.data, breast_cancer.target

    pca = decomposition.pca.PCA()
    pca.fit(X)
    print pca.explained_variance_
    print pca.explained_variance_ratio_

    """
    Plot Variance Ratio for PCA
    """
    plt.bar(range(1, 31), pca.explained_variance_ratio_)
    plt.xlabel('Dimension')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio vs Dimension')
    plt.show()
    plt.clf()


    """
    Plot the PCA dimensions
    """
    fig = plt.figure(1, figsize=(8, 6))

    start = time.time()

    X_new = decomposition.pca.PCA(n_components=2).fit_transform(X)

    print "Elapsed time: {}".format(time.time() - start)

    ax = plt.scatter(X_new[:, 0], X_new[:, 1],  c=y,
               cmap=plt.cm.Paired)
    plt.legend
    plt.title("First three PCA directions")
    plt.xlabel("1st eigenvector")
    plt.ylabel("2nd eigenvector")
    plt.show()

    plt.clf()

    """
    Plot the clustering
    """
    fig = plt.figure(1)
    plt.clf()
    plt.cla()

    model = cluster.KMeans(n_clusters=3)
    labels = model.fit_predict(X_new)
    totz = np.concatenate((X_new,  np.expand_dims(labels, axis=1), np.expand_dims(y, axis=1),), axis=1)

    # for each cluster
    colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple']
    fig = plt.figure()

    for clust in range(0, 3):
        totz_clust = totz[totz[:,-2] == clust]
        print "Cluster Size"
        print totz_clust.shape

        benign = totz_clust[totz_clust[:,-1] == 1]
        malignant = totz_clust[totz_clust[:,-1] == 0]

        plt.scatter(benign[:, 0], benign[:, 1],  color=colors[clust], marker=".")
        plt.scatter(malignant[:, 0], malignant[:, 1],  color=colors[clust], marker="x")

    centroids = model.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
        marker='x', s=169, linewidths=3, color="black",
         zorder=10)

    plt.title("Breast Cancer Clustering KMeans")
    plt.xlabel("1st Component")
    plt.ylabel("2nd Component")
    plt.show()

    tester = emtc.ExpectationMaximizationTestCluster(X_new, y, clusters=range(2,10), plot=False, targetcluster=3, stats=True)
    tester.run()


