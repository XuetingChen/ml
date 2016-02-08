from sklearn import svm, cross_validation, datasets
from numpy import *
import pylab as pl


if __name__ == "__main__":
    breast_cancer = datasets.load_breast_cancer()

    offsets = arange(1,10)
    train_err = zeros(len(offsets))
    test_err = zeros(len(offsets))
    cross_val_scores = zeros(len(offsets))

    for i, d in enumerate(offsets):
        print "--{}--".format(i)
        offset = int((d/10.)*len(breast_cancer.data))
        X_train = breast_cancer.data[:offset]
        Y_train = breast_cancer.target[:offset]
        X_test = breast_cancer.data[offset:]
        Y_test = breast_cancer.target[offset:]

        classifier = svm.SVC(kernel='linear')

        classifier.fit(X_train, Y_train)

        # Find the error rate on the training set
        train_err[i] = 1 - classifier.score(X_train, Y_train)

        # Find the error rate on the testing set
        test_err[i] = 1 - classifier.score(X_test, Y_test)

        scores = cross_validation.cross_val_score(classifier, X_train, Y_train, cv=10)
        cross_val_scores[i] = 1 - scores.mean()

    # Plot training and test error as a function of the depth of the decision tree learnt
    pl.figure()
    pl.title('Breast Cancer SVM: Error Rate vs Training Set Size')
    pl.plot(offsets, test_err, lw=2, label = 'test error')
    pl.plot(offsets, train_err, lw=2, label = 'training error')
    pl.plot(offsets, cross_val_scores, lw=2, label = 'cross validation error')
    pl.legend(loc=0)
    pl.xlabel('Training Size')
    pl.ylabel('Error Rate')
    pl.grid()
    pl.show()


