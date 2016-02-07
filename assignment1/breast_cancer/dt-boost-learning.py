from sklearn.cross_validation import cross_val_score
from sklearn import tree, preprocessing, datasets, ensemble
from numpy import *
import pylab as pl

if __name__ == "__main__":
    breast_cancer = datasets.load_breast_cancer()

    # We will vary the depth of decision trees from 2 to 50
    offset = arange(1,10)
    train_err = zeros(len(offset))
    test_err = zeros(len(offset))
    cross_val_scores = zeros(len(offset))

    for i, d in enumerate(offset):
        offsets = int((d/10.)*len(breast_cancer.data))
        X_train = breast_cancer.data[:offsets]
        Y_train = breast_cancer.target[:offsets]
        X_test = breast_cancer.data[offsets:]
        Y_test = breast_cancer.target[offsets:]

        print "Decision tree with depth: %d" % d
        # Setup a Decision Tree Classifier so that it learns a tree with depth d
        classifier = tree.DecisionTreeClassifier(max_depth=10, min_samples_split=2, max_leaf_nodes=15, criterion='entropy', min_samples_leaf=1)
        dt = ensemble.AdaBoostClassifier(classifier, n_estimators=15, learning_rate=.5)

        # Fit the learner to the training data
        dt.fit(X_train, Y_train)

        # Find the error rate on the training set
        train_err[i] = 1 - dt.score(X_train, Y_train)

        # Find the error rate on the testing set
        test_err[i] = 1 - dt.score(X_test, Y_test)

        scores = cross_val_score(dt, X_train, Y_train, cv=10)
        cross_val_scores[i] = 1 - scores.mean()

    # Plot training and test error as a function of the depth of the decision tree learnt
    pl.figure()
    pl.title('Breast Cancer Learning Boosted Decision Trees: Error Rate vs Training Set Size')
    pl.plot(offset, test_err, lw=2, label = 'test error')
    pl.plot(offset, train_err, lw=2, label = 'training error')
    pl.plot(offset, cross_val_scores, lw=2, label = 'cross validation error')
    pl.legend(loc=0)
    pl.xlabel('Training Size')
    pl.ylabel('Error Rate')
    pl.grid()
    pl.show()


