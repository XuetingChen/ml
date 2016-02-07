from sklearn.cross_validation import cross_val_score
from sklearn import neighbors, preprocessing
from numpy import *
import pandas as pd
from time import time


df = pd.read_csv("letter-recognition.csv")
dft = df
# dft, mapping = encode_target(df, "letter")

offset = int(0.7*len(df))
X_train = preprocessing.normalize(dft.ix[:offset, 1:])
Y_train = dft.ix[:offset, 0]
X_test = preprocessing.normalize(dft.ix[offset:, 1:])
Y_test = dft.ix[offset:, 0]

# Setup a Decision Tree Classifier so that it learns a tree with depth d
classifier = neighbors.KNeighborsClassifier(n_neighbors=4, weights='uniform', p=1)

start = time()
# Fit the learner to the training data
classifier.fit(X_train, Y_train)
end = time()
print(("\nLearner took {:.4f} ").format(end - start))


# Find the MSE on the training set
start = time()
train_err = 1 - classifier.score(X_train, Y_train)
end = time()
print(("\nTraining took {:.4f} ").format(end - start))

start = time()
test_err = 1 - classifier.score(X_test, Y_test)
end = time()
print(("\nTesting took {:.4f} ").format(end - start))


print "Train err: {:.4f}", train_err
print "Train acc: {:.4f}", 1-train_err
print "Test err:  {:.4f}", test_err
print "Test acc:  {:.4f}", 1-test_err
cross_val_err = 1 - cross_val_score(classifier, X_train, Y_train)

print "Cross val err: {:.4f}", cross_val_err.mean()

