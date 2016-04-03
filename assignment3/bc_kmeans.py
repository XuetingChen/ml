from sklearn import datasets

from clustertesters import KMeansTestCluster as kmtc

if __name__ == "__main__":
    breast_cancer = datasets.load_breast_cancer()
    X, y = breast_cancer.data, breast_cancer.target

    tester = kmtc.KMeansTestCluster(X, y, clusters=range(2,10), plot=True, targetcluster=3, stats=True)
    tester.run()






