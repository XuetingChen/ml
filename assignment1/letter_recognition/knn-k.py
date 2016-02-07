from sklearn.cross_validation import cross_val_score
from sklearn import neighbors, preprocessing
from numpy import *
import matplotlib.pyplot as pl
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("letter-recognition.csv")
    dft = df
    # dft, mapping = encode_target(df, "letter")

    offset = int(0.7*len(df))
    lr_data_train = preprocessing.normalize(dft.ix[:offset, 1:])
    lr_target_train = dft.ix[:offset, 0]
    lr_data_test = preprocessing.normalize(dft.ix[offset:, 1:])
    lr_target_test = dft.ix[offset:, 0]

    # Vary k nearest neighbors from 1 to 11
    k = arange(1, 100)
    train_err = zeros(len(k))
    test_err = zeros(len(k))
    cross_val_scores = zeros(len(k))

    for i, d in enumerate(k):
        print "kNN with %d nearest neighbors" % d
        classifier = neighbors.KNeighborsClassifier(n_neighbors=d)

        # Fit the learner to the training data
        classifier.fit(lr_data_train, lr_target_train)

        # Find the error on the training set
        train_err[i] = 1 - classifier.score(lr_data_train, lr_target_train)

        # Find the error on the testing set
        test_err[i] = 1 - classifier.score(lr_data_test, lr_target_test)

        scores = cross_val_score(classifier, lr_data_train, lr_target_train, cv=10)
        cross_val_scores[i] = 1 - scores.mean()

    # Plot training and test error as a function of the depth of the decision tree learnt
    pl.figure()
    pl.title('Letter Recognition kNN: Error Rate vs Number of Neighbors')
    pl.plot(k, test_err, lw=2, label = 'test error')
    pl.plot(k, train_err, lw=2, label = 'training error')
    pl.plot(k, cross_val_scores, lw=2, label = 'cross validation error')
    pl.legend(loc=0)
    pl.xlabel('k neighbors')
    pl.ylabel('Error Rate')
    pl.grid()
    pl.show()


