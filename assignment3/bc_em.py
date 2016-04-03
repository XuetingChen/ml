
from sklearn import  datasets, metrics
from clustertesters import ExpectationMaximizationTestCluster as emtc


if __name__ == "__main__":
    breast_cancer = datasets.load_breast_cancer()

    X, y = breast_cancer.data, breast_cancer.target

    tester = emtc.ExpectationMaximizationTestCluster(X, y, clusters=range(1,10), plot=True, targetcluster=3, stats=True)
    tester.run()

