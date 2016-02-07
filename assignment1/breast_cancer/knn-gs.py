from sklearn import neighbors, datasets
import numpy as np
from time import time
from operator import itemgetter
from sklearn.grid_search import GridSearchCV


def report(grid_scores, n_top=3):
    """Report top n_top parameters settings, default n_top=3.

    Args
    ----
    grid_scores -- output from grid or random search
    n_top -- how many to report, of top models

    Returns
    -------
    top_params -- [dict] top parameter settings found in
                  search
    """
    top_scores = sorted(grid_scores,
                        key=itemgetter(1),
                        reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print(("Mean validation score: "
               "{0:.3f} (std: {1:.3f})").format(
                score.mean_validation_score,
                np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

    return top_scores[0].parameters


def run_gridsearch(X, y, clf, param_grid, cv=5):
    """Run a grid search for best kNN parameters.

    Args
    ----
    X -- features
    y -- targets (classes)
    cf -- scikit-learn Decision Tree
    param_grid -- [dict] parameter settings to test
    cv -- fold of cross-validation, default 5

    Returns
    -------
    top_params -- [dict] from report()
    """
    grid_search = GridSearchCV(clf,
                               param_grid=param_grid,
                               cv=cv)
    start = time()
    grid_search.fit(X, y)

    print(("\nGridSearchCV took {:.2f} "
           "seconds for {:d} candidate "
           "parameter settings.").format(time() - start,
                                         len(grid_search.grid_scores_)))

    top_params = report(grid_search.grid_scores_, 3)
    return top_params


if __name__ == "__main__":
    breast_cancer = datasets.load_breast_cancer()

    # set of parameters to test
    param_grid = {"n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  "p": [1, 2],
                  "weights": ['uniform', 'distance']
                  }

    offset = int(0.6 * len(breast_cancer.data))
    X_train = breast_cancer.data[:offset]
    Y_train = breast_cancer.target[:offset]
    X_test = breast_cancer.data[offset:]
    Y_test = breast_cancer.target[offset:]

    classifier = neighbors.KNeighborsClassifier()

    ts_gs = run_gridsearch(X_train, Y_train, classifier, param_grid, cv=10)
