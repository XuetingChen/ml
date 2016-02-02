from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from numpy import *
import pylab as pl
import pandas as pd
from sklearn.metrics import mean_squared_error

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
    df = pd.read_csv("breast-cancer-wisconsin-clean.data", dtype=str)
    print df
    df.drop("id",1)
    # shuffle(df)
    offset = int(0.75*len(df))
    bc_data_train = df.ix[:offset, :-2]
    bc_target_train = df.ix[:offset, -1]
    bc_data_test = df.ix[offset:, :-2]
    bc_target_test = df.ix[offset:, -1]

    # decision tree
    clf = DecisionTreeClassifier(min_samples_split=100, random_state=99, max_depth=10)
    clf.fit(bc_data_train, bc_target_train)
    pred = clf.predict(bc_data_test)
    scores = cross_val_score(clf, bc_data_train, bc_target_train, cv=2)
    print ("mean: {:.3f} (std: {:.3f})".format(scores.mean(),scores.std()),"\n\n")
    metricScore = accuracy_score(bc_target_test, pred)
    print metricScore

    # We will vary the depth of decision trees from 2 to 25
    max_depth = arange(2, 40)
    train_err = zeros(len(max_depth))
    test_err = zeros(len(max_depth))

    for i, d in enumerate(max_depth):
        # Setup a Decision Tree Regressor so that it learns a tree with depth d
        classifier = DecisionTreeClassifier(max_depth=d)

        # Fit the learner to the training data
        classifier.fit(bc_data_train, bc_target_train)

        # Find the MSE on the training set
        pred_train = classifier.predict(bc_data_train)
        train_err[i] = mean_squared_error(bc_target_train, classifier.predict(bc_data_train))

        # Find the MSE on the testing set
        pred_test = classifier.predict(bc_data_test)
        test_err[i] = mean_squared_error(bc_target_test, classifier.predict(bc_data_test))

    # Plot training and test error as a function of the depth of the decision tree learnt
    pl.figure()
    pl.title('Decision Trees: Performance vs Max Depth')
    pl.plot(max_depth, test_err, lw=2, label = 'test error')
    pl.plot(max_depth, train_err, lw=2, label = 'training error')
    pl.legend()
    pl.xlabel('Max Depth')
    pl.ylabel('RMS Error')
    pl.show()


