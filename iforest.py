# Follows algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf
import numpy as np
import pandas as pd
import random as rn
from sklearn.metrics import confusion_matrix
import os


def c_length(n):
    if n > 2:
        return 2.0 * (np.log(n - 1) + 0.5772156649) - (2.0 * (n - 1.) / (n * 1.0))
    elif n == 2:
        return 1
    else:
        return 0

class IsolationTreeEnsemble(object):
    def __init__(self, sample_size, n_trees=10, limit=None):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.trees = []
        self.limit = limit
        self.c = c_length(sample_size)
        # self.X = X

    def fit(self, X: np.ndarray, improved=True):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        lenX = len(X)

        if self.limit is None:
            self.limit = int(np.ceil(np.log2(self.sample_size)))

        for i in range(self.n_trees):
            ind = rn.sample(range(lenX), self.sample_size)
            X_p = X[ind]
            self.trees.append(IsolationTree(X_p, 0, self.limit))

        return self

    def path_length(self, X: np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """
        lenX = len(X)
        if isinstance(X, pd.DataFrame):
            X = X.values

        P = np.zeros(lenX)
        for i in range(lenX):
            p_temp = 0
            x = X[i]
            for j in range(self.n_trees):
                T = self.trees[j].root
                while T.ntype != 'exNode':
                    p_temp += 1
                    if x[T.q] < T.p:
                        T = T.left
                    else:
                        T = T.right

                p_temp += c_length(T.size)
            Eh = p_temp / self.n_trees
            P[i] = Eh
        return P

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """

        lenX = len(X)
        if isinstance(X, pd.DataFrame):
            X = X.values

        P = np.zeros(lenX)
        for i in range(lenX):
            p_temp = 0
            x = X[i]
            for j in range(self.n_trees):
                T = self.trees[j].root
                while T.ntype != 'exNode':
                    p_temp += 1
                    if x[T.q] < T.p:
                        T = T.left
                    else:
                        T = T.right

                p_temp += c_length(T.size)
            Eh = p_temp / self.n_trees
            P[i] = 2.0 ** (-Eh / self.c)
        return P

    def predict_from_anomaly_scores(self, scores: np.ndarray, threshold: float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        R = np.zeros(len(scores))
        for i in range(len(scores)):
            if scores[i] > threshold:
                R[i] = 1
            else:
                R[i] = 0
        return R

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."
        score_array = self.anomaly_score(X)

        return self.predict_from_anomaly_scores(score_array, threshold)


class Node(object):
    def __init__(self, X, depth, p, q, left, right, ntype=''):
        self.depth = depth  # current depth
        self.size = len(X)
        self.q = q  # attribute
        self.p = p  # split value
        self.left = left
        self.right = right
        self.ntype = ntype
        self.X = X
rn.seed(420)

class IsolationTree(object):

    def __init__(self, X, depth, height_limit):
        self.depth = depth  # current depth
        self.X = X  # save the data for now
        self.height_limit = height_limit  # height limit
        self.p = None  # splitvalue
        self.q = None
        # if added noise, filter out features with low variance
        self.n_dimension = np.where(X.var(axis=0) < np.quantile(X.var(axis=0), 0.75))[0]
        self.n_nodes = 0
        self.size = len(X) # n objects
        self.root = self.fit(X, depth, height_limit)


    def fit(self, X: np.ndarray, depth, height_limit, improved=True):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.

        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """
        self.depth = depth

        if depth >= height_limit or len(X) <= 1:
            left = None
            right = None
            self.n_nodes += 1
            return Node(X, depth, self.p, self.q, left, right, ntype='exNode')

        else:

            self.q = rn.choice(self.n_dimension)
            X_sub = X[:, self.q]
            maximum = np.amax(X_sub)
            minimum = np.amin(X_sub)
            self.p = rn.uniform(minimum, maximum)
            w = np.where(X[:, self.q] < self.p, True, False)
            self.n_nodes += 1
            return Node(X, depth, self.p, self.q,
                        left=self.fit(X[w], depth+1, height_limit),
                        right=self.fit(X[~w], depth+1, height_limit),
                        ntype='inNode')


def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """

    for thres in np.arange(1, 0, -0.01):
        y_pred = np.array([1 if s>= thres else 0 for s in scores])
        confusion = confusion_matrix(y, y_pred)
        TN,FP,FN,TP = confusion.flat
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        if TPR >= desired_TPR:
            return thres, FPR









