from sklearn import tree, ensemble, datasets
import util


if __name__ == "__main__":
    breast_cancer = datasets.load_breast_cancer()

    # set of parameters to test
    param_grid = {"n_estimators": [1,5,10,15,30,40,50],
                  "learning_rate": [.1, .5, 1.]
                  }

    offset = int(0.6*len(breast_cancer.data))
    X_train = breast_cancer.data[:offset]
    Y_train = breast_cancer.target[:offset]
    X_test = breast_cancer.data[offset:]
    Y_test = breast_cancer.target[offset:]

    dt = tree.DecisionTreeClassifier(min_samples_split=2, max_leaf_nodes=20, criterion='entropy', max_depth=15, min_samples_leaf=1)
    classifier = ensemble.AdaBoostClassifier(dt)

    ts_gs = util.run_gridsearch(X_train, Y_train, classifier, param_grid, cv=10)

