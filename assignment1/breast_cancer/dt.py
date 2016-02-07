from sklearn.cross_validation import cross_val_score
from sklearn import tree, datasets
from numpy import *
import pylab as pl

if __name__ == "__main__":
    breast_cancer = datasets.load_breast_cancer()

    print (breast_cancer.data)
    print (breast_cancer.target)

    offset = int(0.6*len(breast_cancer.data))
    X_train = breast_cancer.data[:offset]
    Y_train = breast_cancer.target[:offset]
    X_test = breast_cancer.data[offset:]
    Y_test = breast_cancer.target[offset:]

    # We will vary the depth of decision trees from 2 to 50
    max_depth = arange(2, 50)
    train_err = zeros(len(max_depth))
    test_err = zeros(len(max_depth))
    cross_val_scores = zeros(len(max_depth))

    for i, d in enumerate(max_depth):
        print "Decision tree with depth: %d" % d
        # Setup a Decision Tree Classifier so that it learns a tree with depth d
        classifier = tree.DecisionTreeClassifier(max_depth=d, min_samples_split=2, max_leaf_nodes=15, criterion='entropy', min_samples_leaf=1)

        # Fit the learner to the training data
        classifier.fit(X_train, Y_train)

        # Find the error rate on the training set
        train_err[i] = 1 - classifier.score(X_train, Y_train)

        # Find the error rate on the testing set
        test_err[i] = 1 - classifier.score(X_test, Y_test)

        scores = cross_val_score(classifier, X_train, Y_train, cv=10)
        cross_val_scores[i] = 1 - scores.mean()

    # Plot training and test error as a function of the depth of the decision tree learnt
    pl.figure()
    pl.title('Breast Cancer Decision Trees: Error Rate vs Max Depth')
    pl.plot(max_depth, test_err, lw=2, label = 'test error')
    pl.plot(max_depth, train_err, lw=2, label = 'training error')
    pl.plot(max_depth, cross_val_scores, lw=2, label = 'cross validation error')
    pl.legend(loc=0)
    pl.xlabel('Max Depth')
    pl.ylabel('Error Rate')
    pl.grid()
    pl.show()


