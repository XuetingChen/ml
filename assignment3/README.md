#Assignment 3 - Unsupervised Learning
This assignment investigates and analyzes a number of algorithms used in unsupervised learning. Clustering methods such as K-Means and Expectation Maximization (using Gaussian Mixture Models) and
dimensionality reduction algorithms such as PCA, ICA, LDA, and Random Projections are used to manipulate and investigate how clustering is affected. Lastly, using these feature transformating
methods, new features were added to existing datasets to be used to train neural network classifiers.

Implementations of these algorithms are provided by scikitlearn (clustering and dimensionality reduction) and PyBrain (neural networks)

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

Experiment files can all be found under the root directory of assignment 3. File names describe the dataset, if and what dimensionality reduction algorithm was performed, and the
clustering performed. For example, bc_pca_kmeans.py describes the Breast Cancer dataset, reduced using PCA and clustered using K-Means analysis.

Neural network experiments are found in ann_cluster.py and ann_pca.py, the first being neural networks run with the additional cluster label feature and the latter being a dimensionally reduced
dataset neural network. 
