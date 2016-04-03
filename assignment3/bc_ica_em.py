
from clustertesters import ExpectationMaximizationTestCluster as emtc
from sklearn import  datasets, metrics, decomposition, cluster, mixture
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis

if __name__ == "__main__":
    breast_cancer = datasets.load_breast_cancer()

    X, y = breast_cancer.data, breast_cancer.target

    kurts = []
    for i in range (1, 31):
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

    ipca = decomposition.FastICA(n_components=2, whiten=True)
    X_new = ipca.fit_transform(X, y)

    """
    Plot the clustering
    """
    fig = plt.figure(1)
    plt.clf()
    plt.cla()

    model =  mixture.GMM(n_components=3, covariance_type='full')
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

    # centroids = model.cluster_centers_
    # plt.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
    #     marker='x', s=169, linewidths=3, color="black",
    #      zorder=10)

    plt.title("Breast Cancer Clustering EM")
    plt.xlabel("1st Component")
    plt.ylabel("2nd Component")
    plt.show()

    tester = emtc.ExpectationMaximizationTestCluster(X_new, y, clusters=range(2,10), plot=False, targetcluster=3, stats=True)
    tester.run()

