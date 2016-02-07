from sklearn import preprocessing, ensemble, tree
import pandas as pd
import util as util


if __name__ == "__main__":

    df = pd.read_csv("letter-recognition.csv")
    dft = df

    # set of parameters to test
    param_grid = {"n_estimators": [1,5,10,15,30,40,50],
                  "learning_rate": [.1, .5, 1.]
                  }

    offset = int(0.7 * len(df))
    lr_data_train = preprocessing.normalize(dft.ix[:offset, 1:])
    lr_target_train = dft.ix[:offset, 0]
    lr_data_test = preprocessing.normalize(dft.ix[offset:, 1:])
    lr_target_test = dft.ix[offset:, 0]

    classifier = tree.DecisionTreeClassifier(criterion='entropy', max_depth=20, min_samples_leaf=1)
    dt = ensemble.AdaBoostClassifier(classifier)

    ts_gs = util.run_gridsearch(lr_data_train, lr_target_train, dt, param_grid, cv=10)

