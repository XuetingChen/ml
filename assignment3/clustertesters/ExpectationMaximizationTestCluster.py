"""
A K means experimenter
__author__      = "Jonathan Satria"
__date__ = "April 01, 2016"
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import mixture, metrics
from scipy.spatial.distance import cdist

class ExpectationMaximizationTestCluster():
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
        bic=[]
        aic=[]

        for k in self.clusters:
            model = mixture.GMM(n_components=k, covariance_type='full')
            labels = model.fit_predict(self.X)
            if k == self.targetcluster and self.stats:
                nd_data = np.concatenate((self.X, np.expand_dims(labels, axis=1),np.expand_dims(self.y, axis=1)), axis=1)
                pd_data = pd.DataFrame(nd_data)
                pd_data.to_csv("cluster.csv", index=False, index_label=False, header=False)

                for i in range (0,3):
                    print "Cluster {}".format(i)
                    cluster = pd_data.loc[pd_data.iloc[:,-2]==i].iloc[:,-2:]
                    print cluster.shape[0]
                    print float(cluster.loc[cluster.iloc[:,-1]==0].shape[0])/cluster.shape[0]
                    print float(cluster.loc[cluster.iloc[:,-1]==1].shape[0])/cluster.shape[0]

            #meandist.append(sum(np.min(cdist(self.X, model.cluster_centers_, 'euclidean'), axis=1))/ self.X.shape[0])

            homogeneity_scores.append(metrics.homogeneity_score(self.y, labels))
            completeness_scores.append(metrics.completeness_score(self.y, labels))
            rand_scores.append(metrics.adjusted_rand_score(self.y, labels))
            bic.append(model.bic(self.X))
            aic.append(model.aic(self.X))
            #silhouettes.append(metrics.silhouette_score(self.X, model.labels_ , metric='euclidean',sample_size=self.X.shape[0]))

        if self.gen_plot:
            self.plot(meandist, homogeneity_scores, completeness_scores, rand_scores, bic, aic)

    def plot(self, meandist, homogeneity, completeness, rand, bic, aic):
            # """
            # Plot average distance from observations from the cluster centroid
            # to use the Elbow Method to identify number of clusters to choose
            # """
            # plt.plot(self.clusters, meandist)
            # plt.xlabel('Number of clusters')
            # plt.ylabel('Average distance')
            # plt.title('Average distance vs. K Clusters')
            # plt.show()
            #
            # plt.clf()

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
            Plot BIC Score from observations from the cluster centroid
            to use the Elbow Method to identify number of clusters to choose
            """
            plt.plot(self.clusters, bic)
            plt.xlabel('Number of clusters')
            plt.ylabel('BIC Score')
            plt.title('BIC Score vs. K Clusters')
            plt.show()

            """
            Plot AIC Score from observations from the cluster centroid
            to use the Elbow Method to identify number of clusters to choose
            """
            plt.plot(self.clusters, aic)
            plt.xlabel('Number of clusters')
            plt.ylabel('AIC Score')
            plt.title('AIC Score vs. K Clusters')
            plt.show()