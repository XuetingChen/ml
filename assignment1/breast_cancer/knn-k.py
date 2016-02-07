from sklearn.cross_validation import cross_val_score
from sklearn import neighbors, datasets
from numpy import *
import matplotlib.pyplot as pl

if __name__ == "__main__":
    breast_cancer = datasets.load_breast_cancer()

    offset = int(0.6*len(breast_cancer.data))
    X_train = breast_cancer.data[:offset]
    Y_train = breast_cancer.target[:offset]
    X_test = breast_cancer.data[offset:]
    Y_test = breast_cancer.target[offset:]

    # Vary k nearest neighbors from 1 to 11
    k = arange(1, 100)
    train_err = zeros(len(k))
    test_err = zeros(len(k))
    cross_val_scores = zeros(len(k))

    for i, d in enumerate(k):
        print "kNN with %d nearest neighbors" % d
        classifier = neighbors.KNeighborsClassifier(n_neighbors=d, weights='uniform', p=1)

        # Fit the learner to the training data
        classifier.fit(X_train, Y_train)

        # Find the error on the training set
        train_err[i] = 1 - classifier.score(X_train, Y_train)

        # Find the error on the testing set
        test_err[i] = 1 - classifier.score(X_test, Y_test)

        scores = cross_val_score(classifier, X_train, Y_train, cv=10)
        cross_val_scores[i] = 1 - scores.mean()

    # Plot training and test error as a function of the depth of the decision tree learnt
    pl.figure()
    pl.title('Breast Cancer kNN: Error Rate vs Number of Neighbors')
    pl.plot(k, test_err, lw=2, label = 'test error')
    pl.plot(k, train_err, lw=2, label = 'training error')
    pl.plot(k, cross_val_scores, lw=2, label = 'cross validation error')
    pl.legend(loc=0)
    pl.xlabel('k neighbors')
    pl.ylabel('Error Rate')
    pl.grid()
    pl.show()


