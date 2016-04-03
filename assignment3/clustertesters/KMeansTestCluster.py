"""
A K means experimenter
__author__      = "Jonathan Satria"
__date__ = "April 01, 2016"
"""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn import metrics, decomposition
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D


class KMeansTestCluster():
    def __init__(self, X, y, clusters, plot=False, targetcluster=3, stats=False):
        self.X = X
        self.y = y
        self.clusters = clusters
        self.gen_plot = plot
        self.targetcluster = targetcluster
        self.stats = stats

    def run(self):
        meandist=[]
        homogeneity_scores=[]
        completeness_scores=[]
        rand_scores=[]
        silhouettes=[]

        for k in self.clusters:
            model = KMeans(n_clusters=k, max_iter=5000, init='k-means++')
            labels = model.fit_predict(self.X)

            if k == self.targetcluster and self.stats:
                nd_data = np.concatenate((self.X, np.expand_dims(labels, axis=1),np.expand_dims(self.y, axis=1)), axis=1)
                pd_data = pd.DataFrame(nd_data)
                pd_data.to_csv("cluster.csv", index=False, index_label=False, header=False)
                print model.cluster_centers_

                for i in range (0,3):
                    print "Cluster {}".format(i)
                    cluster = pd_data.loc[pd_data.iloc[:,-2]==i].iloc[:,-2:]
                    print cluster.shape[0]
                    print float(cluster.loc[cluster.iloc[:,-1]==0].shape[0])/cluster.shape[0]
                    print float(cluster.loc[cluster.iloc[:,-1]==1].shape[0])/cluster.shape[0]

            meandist.append(sum(np.min(cdist(self.X, model.cluster_centers_, 'euclidean'), axis=1))/ self.X.shape[0])
            homogeneity_scores.append(metrics.homogeneity_score(self.y, labels))
            completeness_scores.append(metrics.completeness_score(self.y, labels))
            rand_scores.append(metrics.adjusted_rand_score(self.y, labels))

        if self.gen_plot:
            #self.visualize()

            self.plot(meandist, homogeneity_scores, completeness_scores, rand_scores, silhouettes)

    def visualize(self):
        """
        Generate scatter plot of Kmeans with Centroids shown
        """
        fig = plt.figure(1)
        plt.clf()
        plt.cla()

        X_new = decomposition.pca.PCA(n_components=3).fit_transform(self.X)
        model = KMeans(n_clusters=self.targetcluster, max_iter=5000, init='k-means++')
        labels = model.fit_predict(X_new)
        totz = np.concatenate((X_new,  np.expand_dims(labels, axis=1), np.expand_dims(self.y, axis=1),), axis=1)

        # for each cluster
        colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple']
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for clust in range(0, self.targetcluster):
            totz_clust = totz[totz[:,-2] == clust]
            print "Cluster Size"
            print totz_clust.shape

            benign = totz_clust[totz_clust[:,-1] == 1]
            malignant = totz_clust[totz_clust[:,-1] == 0]

            ax.scatter(benign[:, 0], benign[:, 1], benign[:, 2], color=colors[clust], marker=".")
            ax.scatter(malignant[:, 0], malignant[:, 1], malignant[:, 2], color=colors[clust], marker="x")

        centroids = model.cluster_centers_
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
            marker='x', s=169, linewidths=3, color="black",
             zorder=10)

        # ax.title("Breast Cancer Clustering")
        ax.set_xlabel("1st Component")
        ax.set_ylabel("2nd Component")
        ax.set_zlabel("3rd Component")
        plt.show()

    def plot(self, meandist, homogeneity, completeness, rand, silhouettes):
            """
            Plot average distance from observations from the cluster centroid
            to use the Elbow Method to identify number of clusters to choose
            """
            plt.plot(self.clusters, meandist)
            plt.xlabel('Number of clusters')
            plt.ylabel('Average distance')
            plt.title('Average distance vs. K Clusters')
            plt.show()

            plt.clf()

            """
            Plot homogeneity from observations from the cluster centroid
            to use the Elbow Method to identify number of clusters to choose
            """
            plt.plot(self.clusters, homogeneity)
            plt.xlabel('Number of clusters')
            plt.ylabel('Homogeneity Score')
            plt.title('Homogeneity Score vs. K Clusters')
            plt.show()

            plt.clf()


            """
            Plot completeness from observations from the cluster centroid
            to use the Elbow Method to identify number of clusters to choose
            """
            plt.plot(self.clusters, completeness)
            plt.xlabel('Number of clusters')
            plt.ylabel('Completeness Score')
            plt.title('Completeness Score vs. K Clusters')
            plt.show()

            plt.clf()

            """
            Plot Adjusted RAND Score from observations from the cluster centroid
            to use the Elbow Method to identify number of clusters to choose
            """
            plt.plot(self.clusters, rand)
            plt.xlabel('Number of clusters')
            plt.ylabel('Adjusted RAND Score')
            plt.title('RAND Score vs. K Clusters')
            plt.show()

            """
            Plot Silhouette Score from observations from the cluster centroid
            to use the Elbow Method to identify number of clusters to choose
            """
            # plt.plot(self.clusters, silhouettes)
            # plt.xlabel('Number of clusters')
            # plt.ylabel('Silhouette Score')
            # plt.title('Silhouette Score vs. K Clusters')
            # plt.show()