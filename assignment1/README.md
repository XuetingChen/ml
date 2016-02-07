#Assignment 1 - Supervised Learning#

This assignment and introduction into supervised learning was completed using Python, scikit-learn, and Pybrain. All of these tools are
readily available, easy to install and fully featured out of the box. scikit-learn is used to model decision trees, decision trees with boosting,
k-Nearest Neighbors, support vector machines (SVM) and also provided a sample dataset. Pybrain is used to model artificial
neural networks, particularly feedforward networks with a backpropagation trainer. 

##Installation and Usage##
1. Download and install Anaconda: https://www.continuum.io/downloads
2. Install PyBrain using: `$ easy_install pybrain`

##Description##
The two datasets used in this assignment are the letter recognition dataset and the Wisconsin breast cancer (diagnostic) dataset. The letter recognition
dataset was obtained from the UCI ML Repository: http://archive.ics.uci.edu/ml/datasets/Letter+Recognition and is included here as letter-recognition.csv. 
The Wisconsin breast cancer dataset is included in scikit-learn and was loaded for each experiment. More information regarding this 
dataset can be found here: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29.

##Execution##
`$ python [algorithm].py`
e.g. `python dt.py` for decision tree

##Folder Structure and File Description##
The supervised learning algorithms and experiments are replicated for each dataset and can be found under `/letter-recognition` and `/breast-cancer`
Within each dataset, the following files contain the algorithm with the optimal hyperparameters:
- dt.py - decision tree 
- dt-boost.py - decision tree with AdaBoost
- ann.py - artifical neural network (feedforwardnetwork w/ backpropagation)
- knn.py - k-Nearest neighbors
- svm.py - support vector machine with interchangeable kernel

Miscellaneous files:
There are additional files included that were used to understand the effects of training size, k neighbors, etc on the performance
of each algorithm. Those are:
- *-learning.py - depicts how an algorithm learns as the training size is adjusted
- *-gs.py - grid search used to search for optimal hyperparameters
- dt-maxdepth.py - varying max depth for decision trees
- knn-k.py - varying k neighbors for k-NN 
- util.py - simple utility class
