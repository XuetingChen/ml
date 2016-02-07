from sklearn.cross_validation import cross_val_score
from sklearn import svm
from numpy import *
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
import util as util

if __name__ == "__main__":
    df = pd.read_csv("letter-recognition.csv")
    # dft = df
    dft, mapping = util.encode_target(df, "letter")

    offset = int(0.70*len(df))
    lr_data_train = (dft.ix[:offset, 1:])
    lr_target_train = dft.ix[:offset, 0]
    lr_data_test = (dft.ix[offset:, 1:])
    lr_target_test = dft.ix[offset:, 0]

    # Vary k nearest neighbors from 1 to 11
    k = arange(1, 10)
    train_err = zeros(len(k))
    test_err = zeros(len(k))
    cross_val_scores = zeros(len(k))

    for i, d in enumerate(k):
        print "SVM with %d degree" % d
        classifier = svm.SVC(degree=d, kernel='rbf')

        # Fit the learner to the training data
        classifier.fit(np.array(lr_data_train), lr_target_train)

        # Find the error on the training set
        train_err[i] = 1 - classifier.score(lr_data_train, lr_target_train)
        print train_err[i]

        # Find the error on the testing set
        test_err[i] = 1 - classifier.score(lr_data_test, lr_target_test)
        print test_err[i]
        scores = cross_val_score(classifier, lr_data_train, lr_target_train, cv=10)
        cross_val_scores[i] = 1 - scores.mean()
        print cross_val_scores[i]

    # Plot training and test error as a function of the depth of the decision tree learnt
    pl.figure()
    pl.title('Letter Recognition SVM: Error Rate vs Polynomial Degree')
    pl.plot(k, test_err, lw=2, label = 'test error')
    pl.plot(k, train_err, lw=2, label = 'training error')
    pl.plot(k, cross_val_scores, lw=2, label = 'cross validation error')
    pl.legend(loc=0)
    pl.xlabel('degree')
    pl.ylabel('Error Rate')
    pl.grid()
    pl.show()


