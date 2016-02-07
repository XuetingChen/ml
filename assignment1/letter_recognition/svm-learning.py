from sklearn import svm, cross_validation
from numpy import *
import pylab as pl
import pandas as pd
import util as util

if __name__ == "__main__":
    df = pd.read_csv("letter-recognition.csv")
    # dft = df
    dft, mapping = util.encode_target(df, "letter")

    offsets = arange(1,9)
    train_err = zeros(len(offsets))
    test_err = zeros(len(offsets))
    cross_val_scores = zeros(len(offsets))

    for i, d in enumerate(offsets):
        print "--{}--".format(i)
        offset = int((d/10.)*len(df))
        X_train = (dft.ix[:offset, 1:])
        Y_train = dft.ix[:offset, 0]
        X_test = (dft.ix[offset:, 1:])
        Y_test = dft.ix[offset:, 0]

        classifier = svm.SVC(kernel='rbf')

        classifier.fit(X_train, Y_train)

        # Find the error rate on the training set
        train_err[i] = 1 - classifier.score(X_train, Y_train)

        # Find the error rate on the testing set
        test_err[i] = 1 - classifier.score(X_test, Y_test)

        scores = cross_validation.cross_val_score(classifier, X_train, Y_train, cv=10)
        cross_val_scores[i] = 1 - scores.mean()

    # Plot training and test error as a function of the depth of the decision tree learnt
    pl.figure()
    pl.title('Letter Recognition Decision Trees: Error Rate vs Training Set Size')
    pl.plot(offsets, test_err, lw=2, label = 'test error')
    pl.plot(offsets, train_err, lw=2, label = 'training error')
    pl.plot(offsets, cross_val_scores, lw=2, label = 'cross validation error')
    pl.legend(loc=0)
    pl.xlabel('Training Size')
    pl.ylabel('Error Rate')
    pl.grid()
    pl.show()


