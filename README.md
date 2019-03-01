# Isolation Forest Implementation

Implement Isolation Forest for anomaly detection
The goal of this project is to implement the original [Isolation Forest](paper_iforest.pdf) algorithm by Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou. Unlike other common anomly detection technoques that model what normal looks like, this isolation forest algorithm is focusing on the anomalies which are few and different. 

## Overview

The implementation is in [iforest.py](iforest.py), in this file, I defined following classes and methods: IsolationTreeEnsemble, Isolation Tree, Node classes. 


## Result Interpretation

In [iforest_example.ipynb](iforest_example.ipynb), I visualized the reuslt by the distribution of data, we can see the results of the isolation forest trying to detect anomalies. These data sets all have known targets indicating normal versus anomaly.
The metrics I used for classfication here is TPR, FPR and confusion matrix. Besides, small subset was used for tree visulization to indicate the overall structure of indivuals trees in the isolation forest.
