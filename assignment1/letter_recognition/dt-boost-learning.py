from sklearn.cross_validation import cross_val_score
from sklearn import tree, preprocessing, ensemble
from numpy import *
import pylab as pl
import pandas as pd

def encode_target(df, target_column):
    """Add column to df with integers for the target.

    Args
    ----
    df -- pandas DataFrame.
    target_column -- column to map to int, producing
                     new Target column.

    Returns
    -------
    df_mod -- modified DataFrame.
    targets -- list of target names.
    """
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod[target_column].replace(map_to_int, inplace=True)
    return (df_mod, map_to_int)

if __name__ == "__main__":
    df = pd.read_csv("letter-recognition.csv")
    dft = df
    # dft, mapping = encode_target(df, "letter")

    # We will vary the depth of decision trees from 2 to 50
    offset = arange(1,8)
    train_err = zeros(len(offset))
    test_err = zeros(len(offset))
    cross_val_scores = zeros(len(offset))

    for i, d in enumerate(offset):
        offsets = int((d/10.)*len(df))
        lr_data_train = preprocessing.normalize(dft.ix[:offsets, 1:])
        lr_target_train = dft.ix[:offsets, 0]
        lr_data_test = preprocessing.normalize(dft.ix[offsets:, 1:])
        lr_target_test = dft.ix[offsets:, 0]

        print "Decision tree with training set size: %d" % d
        # Setup a Decision Tree Classifier so that it learns a tree with depth d
        classifier = tree.DecisionTreeClassifier(criterion='entropy', max_depth=20, min_samples_leaf=1)
        dt = ensemble.AdaBoostClassifier(classifier, n_estimators=1, learning_rate=1.)

        # Fit the learner to the training data
        dt.fit(lr_data_train, lr_target_train)

        # Find the error rate on the training set
        train_err[i] = 1 - dt.score(lr_data_train, lr_target_train)

        # Find the error rate on the testing set
        test_err[i] = 1 - dt.score(lr_data_test, lr_target_test)

        scores = cross_val_score(dt, lr_data_train, lr_target_train, cv=10)
        cross_val_scores[i] = 1 - scores.mean()

    # Plot training and test error as a function of the depth of the decision tree learnt
    pl.figure()
    pl.title('Letter Recognition Decision Trees: Error Rate vs Training Set Size')
    pl.plot(offset, test_err, lw=2, label = 'test error')
    pl.plot(offset, train_err, lw=2, label = 'training error')
    pl.plot(offset, cross_val_scores, lw=2, label = 'cross validation error')
    pl.legend(loc=0)
    pl.xlabel('Training Size')
    pl.ylabel('Error Rate')
    pl.grid()
    pl.show()


