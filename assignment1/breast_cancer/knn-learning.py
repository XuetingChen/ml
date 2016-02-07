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

    # We will vary the depth of decision trees from 2 to 50
    offset = arange(1,9)
    train_err = zeros(len(offset))
    test_err = zeros(len(offset))
    cross_val_scores = zeros(len(offset))

    for i, d in enumerate(offset):
        offsets = int((d/10.)*len(breast_cancer.data))
        X_train = breast_cancer.data[:offsets]
        Y_train = breast_cancer.target[:offsets]
        X_test = breast_cancer.data[offsets:]
        Y_test = breast_cancer.target[offsets:]

        print "k-NN with training set size: %d" % d
        classifier = neighbors.KNeighborsClassifier(n_neighbors=4, weights='uniform', p=1)

        # Fit the learner to the training data
        classifier.fit(X_train, Y_train)

        # Find the error rate on the training set
        train_err[i] = 1 - classifier.score(X_train, Y_train)

        # Find the error rate on the testing set
        test_err[i] = 1 - classifier.score(X_test, Y_test)

        scores = cross_val_score(classifier, X_train, Y_train, cv=10)
        cross_val_scores[i] = 1 - scores.mean()

    pl.figure()
    pl.title('Breast Cancer k-NN: Error Rate vs Training Set Size')
    pl.plot(offset, test_err, lw=2, label = 'test error')
    pl.plot(offset, train_err, lw=2, label = 'training error')
    pl.plot(offset, cross_val_scores, lw=2, label = 'cross validation error')
    pl.legend(loc=0)
    pl.xlabel('Training Size')
    pl.ylabel('Error Rate')
    pl.grid()
    pl.show()





