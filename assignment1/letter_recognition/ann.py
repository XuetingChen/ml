import pylab as pl
import pandas as pd
from numpy import *
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.structure.modules   import SigmoidLayer

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
    dft, mapping = encode_target(df, "letter")

    X = (dft.ix[:, 1:])
    y = dft.ix[:, 0]

    ds = ClassificationDataSet(X.shape[1], nb_classes=26)
    for k in range(len(X)):
        ds.addSample(X.ix[k,:],y[k])

    max_epochs = 500

    tstdata, trndata = ds.splitWithProportion(0.3)
    trndata._convertToOneOfMany()
    tstdata._convertToOneOfMany()
    train_errors = []
    test_errors = []

    # List all the different networks we want to test
    net = (buildNetwork(trndata.indim,8,trndata.outdim, bias=True, hiddenclass=SoftmaxLayer, outclass=SoftmaxLayer))

    # Setup a trainer that will use backpropogation for training
    trainer = BackpropTrainer(net, dataset=trndata, learningrate=0.001, verbose=True)

    for i in range(max_epochs):
        error = trainer.train()
        print "Epoch: %d, Error: %7.4f" % (i, error)
        train_errors.append(percentError( trainer.testOnClassData(),
                              trndata['class']))

        test_errors.append(percentError(trainer.testOnClassData(dataset=tstdata), tstdata['class']))

    # Plot training and test error as a function of the number of hidden layers
    pl.figure()
    pl.title('Neural Networks: Performance vs Epochs')
    pl.plot(range(max_epochs), test_errors, lw=2, label = 'test error')
    pl.plot(range(max_epochs), train_errors, lw=2, label = 'training error')
    pl.legend(loc=0)
    pl.xlabel('epoch')
    pl.ylabel('Error Rate')
    pl.show()



