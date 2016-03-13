from numpy import *
import pylab as pl

from pybrain.datasets import ClassificationDataSet
from pybrain.utilities           import percentError
from sklearn import  datasets
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.structure.modules   import SigmoidLayer


if __name__ == "__main__":
    breast_cancer = datasets.load_breast_cancer()

    X, y = breast_cancer.data, breast_cancer.target

    ds = ClassificationDataSet(30, nb_classes=2)
    for k in xrange(len(X)):
        ds.addSample(X[k],y[k])

    tstdata, trndata = ds.splitWithProportion(0.3)
    trndata._convertToOneOfMany()
    tstdata._convertToOneOfMany()

    print "Number of training patterns: ", len(trndata)
    print "Input and output dimensions: ", trndata.indim, trndata.outdim
    print "First sample (input, target, class):"

    max_epochs = 1000

    # List all the different networks we want to test
    net=(buildNetwork(trndata.indim,15,trndata.outdim, hiddenclass=SigmoidLayer, outclass=SigmoidLayer))

    # Setup a trainer that will use backpropogation for training
    trainer = BackpropTrainer(net, dataset=trndata, learningrate=0.001, verbose=True, weightdecay=0.01, momentum=.05)
    train_errors = []
    test_errors = []

    for i in range(max_epochs):
        error = trainer.train()
        print "Epoch: %d, Error: %7.4f" % (i, error)
        train_errors.append(percentError( trainer.testOnClassData(),
                              trndata['class']))
        print train_errors[i]

        test_errors.append(percentError(trainer.testOnClassData(dataset=tstdata), tstdata['class']))
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
