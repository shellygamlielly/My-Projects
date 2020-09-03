"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

Author: Gad Zalcberg
Date: February, 2019

"""
import numpy as np


class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None] * T  # list of base learners
        self.w = np.zeros(T)  # weights
        self.D = [None] * T  # Dist

    def train(self, X, y):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the last iteration.
        """
        num_samples, num_features = X.shape
        D1 = np.ones(num_samples) / num_samples
        D = np.array(D1).reshape(num_samples, )
        for t in range(self.T):
            self.D[t] = D
            # Find weak learner ht(x) that minimizes epsilon_t the sum of weights of misclassified points
            # h = WL(D,X,y) - constructs a weak classifier trained on X; y weighted by D.
            self.h[t] = self.WL(D, X, y)
            # h.predict(X) - returns the classifier's prediction on a set X.
            prediction = self.h[t].predict(X)
            #  sum of weights of misclassified points
            epsilon_t = np.sum(D[prediction - y != 0])
            # Set wt
            self.w[t] = 0.5 * np.log((1 / epsilon_t) - 1)
            # Update weights
            D = D * np.exp(- y * self.w[t] * prediction)
            # Renormalize weights
            D = D / np.sum(D)


    def predict(self, X, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """
        hs = np.array(self.h[:max_t])
        w = self.w[:max_t][:, np.newaxis]
        ht = []
        for i in range(max_t):
            # h.predict(X) - returns the classifier's prediction on a set X.
            ht.append(hs[i].predict(X))
        ht = np.array(ht)
        y_hat = np.sign(np.sum(w * ht, axis=0)).astype(np.int)
        return y_hat

    def error(self, X, y, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: error : the ratio of the correct predictions when predict only with max_t weak learners (float)
        """
        num_samples, num_features = X.shape
        y_hat = self.predict(X, max_t)
        incorrect_predictions = np.count_nonzero(y_hat - y != 0)
        ratio = incorrect_predictions / num_samples
        return ratio
