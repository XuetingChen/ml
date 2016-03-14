#Assignment 2 - Randomized Optimization
This assignment investigates and analyzes four different randomized optimization problems: Random Hill Climbing, Simulated Annealing, Genetic Algorithms, and MIMIC. 
Implementations of these algorithms are provided by the ABAGAIL project: https://github.com/pushkar/ABAGAIL and were subsequently used out of the box.

Different optimization problems are also provided by ABAGAIL and used to demonstrate strengths and weaknesses of the RO problems. These optimization problems were the Four Peaks problem, Knapsack, and CountOnes problem.

##Installation, Compilation and Usage

Prerequisite: Java 1.7+ and ant are installed

```
git clone this repository
cd ABAGAIL
ant
java -cp ABAGAIL.jar [experiment file]
```

##Execution

`$ java -cp ABAGAIL.jar [experiment file]`

**Experiment Files:**
- `breastcancer.BreastCancer` - Optimization of the weights of a neural network for a classification task of Breast Cancer Wisconsin Diagnostic Data. This class will split the wdbc.csv data into a 60/40 train/test set split and optimize with RHC, SA, and GA. Training/testing time and accuracy are measured and outputted. 
- `exp.countones.CountOnesOptimizationTest` - optimization experiment runner for Count Ones problem. N can be modified to adjust the length of the input string. Executes optimization training for specified number of iterations
- `exp.fourpeaks.FourPeaksOptimizationTest` - optimization experiment runner for FourPeaks problem. N can be modified to adjust the length of the input bit string. T is set to 10% of N. Executes optimization training for specified number of iterations
- `exp.knapsack.KnapsackOptimizationTest` - optimization experiment runner for Knapsack problem. N can be modified to adjust the number of items. Executes optimization training for specified number of iterations


