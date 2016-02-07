from sklearn.cross_validation import cross_val_score
from sklearn import neighbors, preprocessing
from numpy import *
import matplotlib.pyplot as pl
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("letter-recognition.csv")
    dft = df

    offset = arange(1,10)
    train_err = zeros(len(offset))
    test_err = zeros(len(offset))
    cross_val_scores = zeros(len(offset))

    for i, d in enumerate(offset):
        offsets = int((d/10.)*len(df))
        lr_data_train = preprocessing.normalize(dft.ix[:offsets, 1:])
        lr_target_train = dft.ix[:offsets, 0]
        lr_data_test = preprocessing.normalize(dft.ix[offsets:, 1:])
        lr_target_test = dft.ix[offsets:, 0]

        print "k-NN with training set size: %d" % d
        classifier = neighbors.KNeighborsClassifier(n_neighbors=4, weights='distance', p=2)

        # Fit the learner to the training data
        classifier.fit(lr_data_train, lr_target_train)

        # Find the error rate on the training set
        train_err[i] = 1 - classifier.score(lr_data_train, lr_target_train)

        # Find the error rate on the testing set
        test_err[i] = 1 - classifier.score(lr_data_test, lr_target_test)

        scores = cross_val_score(classifier, lr_data_train, lr_target_train, cv=10)
        cross_val_scores[i] = 1 - scores.mean()

    pl.figure()
    pl.title('Letter Recognition k-NN: Error Rate vs Training Set Size')
    pl.plot(offset, test_err, lw=2, label = 'test error')
    pl.plot(offset, train_err, lw=2, label = 'training error')
    pl.plot(offset, cross_val_scores, lw=2, label = 'cross validation error')
    pl.legend(loc=0)
    pl.xlabel('Training Size')
    pl.ylabel('Error Rate')
    pl.grid()
    pl.show()


