from numpy import *
import pylab as pl

from pybrain.datasets import ClassificationDataSet, SupervisedDataSet
from pybrain.utilities           import percentError
from sklearn import  datasets, decomposition
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.structure.modules   import SigmoidLayer
import numpy as np


if __name__ == "__main__":
    breast_cancer = datasets.load_breast_cancer()

    X, y = breast_cancer.data, breast_cancer.target

    X = decomposition.pca.PCA(n_components=2).fit_transform(X)

    ds = ClassificationDataSet(X.shape[1], 2)
    for k in xrange(len(X)):
        ds.addSample(X[k],y[k])

    tstdata, trndata = ds.splitWithProportion(0.3)

    max_epochs = 100

    # List all the different networks we want to test
    net=buildNetwork(trndata.indim,15,trndata.outdim,  bias=True)

    # Setup a trainer that will use backpropogation for training
    trainer = BackpropTrainer(net, dataset=trndata, verbose=True, weightdecay=0.01, momentum=.9)
    train_errors = []
    test_errors = []

    # trainer.trainUntilConvergence(maxEpochs=1000, verbose=True, continueEpochs=10, validationProportion=0.25)
    # print percentError(trainer.testOnClassData(), trndata['target'])
    # print percentError(trainer.testOnClassData(dataset=tstdata), tstdata['target'])
    for i in range(max_epochs):
        error = trainer.train()

        print "Epoch: %d, Error: %7.4f" % (i, error)

        train_errors.append(trainer.testOnData(trndata))
        print train_errors[i]

        test_errors.append(trainer.testOnData(tstdata))
        print test_errors[i]

    # Plot training and test error as a function of the number of hidden layers
    pl.figure()
    pl.title('Neural Networks: Performance vs Epochs')
    pl.plot(range(max_epochs), test_errors, lw=2, label = 'test error')
    pl.plot(range(max_epochs), train_errors, lw=2, label = 'training error')
    pl.legend(loc=0)
    pl.xlabel('epoch')
    pl.ylabel('Error Rate')
    pl.show()
