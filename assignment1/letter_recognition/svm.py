from sklearn import svm, datasets, cross_validation
from numpy import *
from time import time
import pandas as pd
import util as util

if __name__ == "__main__":
    df = pd.read_csv("letter-recognition.csv")
    # dft = df
    dft, mapping = util.encode_target(df, "letter")

    offset = int(0.70*len(df))
    X_train = (dft.ix[:offset, 1:])
    Y_train = dft.ix[:offset, 0]
    X_test = (dft.ix[offset:, 1:])
    Y_test = dft.ix[offset:, 0]

    classifier = svm.SVC(kernel='rbf')

    start = time()
    # Fit the learner to the training data
    classifier.fit(X_train, Y_train)
    end = time()
    print(("\nLearner took {:.4f} ").format(end - start))

    start = time()
    train_err = 1 - classifier.score(X_train, Y_train)
    end = time()
    print(("\nTraining took {:.4f} ").format(end - start))

    start = time()
    test_err = 1 - classifier.score(X_test, Y_test)
    end = time()
    print(("\nTesting took {:.4f} ").format(end - start))

    print "Train err: {:.2f}", train_err
    print "Train acc: {:.2f}", 1 - train_err
    print "Test err:  {:.2f}", test_err
    print "Test acc:  {:.2f}", 1 - test_err

    cross_val_err = 1 - cross_validation.cross_val_score(classifier, X_train, Y_train)

    print "Cross val err: {:.4f}", cross_val_err.mean()


