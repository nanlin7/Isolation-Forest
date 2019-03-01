# Isolation Forest Implementation

Implement Isolation Forest for anomaly detection
The goal of this project is to implement the original [Isolation Forest] (paper_iforest.pdf) algorithm by Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou. Unlike other common anomly detection technoques that model what normal looks like, this isolation forest algorithm is focusing on the anomalies which are few and different. 

## Overview

The implementation is in [iforest.py], in this file, I defined following classes and methods: IsolationTreeEnsemble, Isolation Tree, Node classes. 

<table border="0">
<tr>
<td width="50%" valign="top"><img src="images/iForest.png" width="350"></td><td width="50%" valign="top"><img src="images/iTree.png" width="350"></td>
</tr>
<tr>
<td valign="top">
<img src="images/PathLength.png" width="350">
</td>
<td valign="top">
Please use this version of average path length <tt>c()</tt>, not the one in the original paper:<br>
<img src="images/avgPathLength.png" width="320">

<p>Then finally here's the scoring formula:<br>

<img src="images/score.png" width="150">

<p>where "<i>H(i)</i> is the harmonic number and it can be estimated by <i>ln(i)</i> + 0.5772156649 (Euler’s constant)."
</td>
</tr>
</table>
  
## Result Interpretation

In [iforest_example.ipynb], I visualized the reuslt by the distribution of data, we can see the results of the isolation forest trying to detect anomalies. These data sets all have known targets indicating normal versus anomaly.
The metrics I used for classfication here is TPR, FPR and confusion matrix. Besides, small subset was used for tree visulization to indicate the overall structure of indivuals trees in the isolation forest.
