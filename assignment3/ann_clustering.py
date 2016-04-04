from numpy import *
import pylab as pl

from pybrain.datasets import ClassificationDataSet, SupervisedDataSet
from pybrain.utilities           import percentError
from sklearn import  datasets, decomposition, cluster
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.structure.modules   import SigmoidLayer
import numpy as np
import time


if __name__ == "__main__":
    breast_cancer = datasets.load_breast_cancer()

    X, y = breast_cancer.data, breast_cancer.target

    model = cluster.KMeans(n_clusters=2)
    labels = model.fit_predict(X)

    print X.shape
    X = np.concatenate((X, np.expand_dims(labels, axis=1)), axis=1)
    print X.shape

    ds = ClassificationDataSet(X.shape[1], 2)
    for k in xrange(len(X)):
        ds.addSample(X[k],y[k])

    tstdata, trndata = ds.splitWithProportion(0.3)

    max_epochs = 1000

    # List all the different networks we want to test
    net=buildNetwork(trndata.indim,15,trndata.outdim, outclass=SigmoidLayer, bias=True)
    print net

    # Setup a trainer that will use backpropogation for training
    trainer = BackpropTrainer(net, dataset=trndata, verbose=True, weightdecay=0.01, momentum=.9)
    train_errors = []
    test_errors = []

    for i in range(max_epochs):
        start = time.time()
        error = trainer.train()

        print "Epoch: %d, Error: %7.4f" % (i, error)

        train_errors.append(trainer.testOnData(trndata))
        print train_errors[i]

        test_errors.append(trainer.testOnData(tstdata))
        print test_errors[i]
        print "Elapsed time: {}".format(time.time()-start)

    # Plot training and test error as a function of the number of hidden layers
    pl.figure()
    pl.title('Neural Networks: Performance vs Epochs')
    pl.plot(range(max_epochs), test_errors, lw=2, label = 'test error')
    pl.plot(range(max_epochs), train_errors, lw=2, label = 'training error')
    pl.legend(loc=0)
    pl.xlabel('epoch')
    pl.ylabel('Error Rate')
    pl.show()
    pl.show()
