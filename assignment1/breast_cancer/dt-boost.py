"""
Plots Model Complexity graphs for Decision Trees
For Decision Trees we vary complexity by changing the size of the decision tree
"""

from numpy import *
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble, datasets
from time import time
from sklearn.cross_validation import cross_val_score

breast_cancer = datasets.load_breast_cancer()

offset = int(0.6*len(breast_cancer.data))
X_train = breast_cancer.data[:offset]
Y_train = breast_cancer.target[:offset]
X_test = breast_cancer.data[offset:]
Y_test = breast_cancer.target[offset:]

# Setup a Decision Tree Classifier so that it learns a tree with depth d
classifier = DecisionTreeClassifier(max_depth=10, min_samples_split=2, max_leaf_nodes=15, criterion='entropy', min_samples_leaf=1)
ensemble = ensemble.AdaBoostClassifier(classifier, n_estimators=15, learning_rate=.5)

start = time()
# Fit the learner to the training data
ensemble.fit(X_train, Y_train)
end = time()
print(("\nLearner took {:.4f} ").format(end - start))


# Find the MSE on the training set
start = time()
train_err = 1 - ensemble.score(X_train, Y_train)
end = time()
print(("\nTraining took {:.4f} ").format(end - start))

start = time()
test_err = 1 - ensemble.score(X_test, Y_test)
end = time()
print(("\nTesting took {:.4f} ").format(end - start))


print "Train err: {:.2f}", train_err
print "Train acc: {:.2f}", 1-train_err
print "Test err:  {:.2f}", test_err
print "Test acc:  {:.2f}", 1-test_err
cross_val_err = 1 - cross_val_score(ensemble, X_train, Y_train)

print "Cross val err: {:.4f}", cross_val_err.mean()


