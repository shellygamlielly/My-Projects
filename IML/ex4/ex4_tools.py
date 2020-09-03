"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

This module provides some useful tools for Ex4.

Author: Gad Zalcberg
Date: February, 2019

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from itertools import product
from matplotlib.pyplot import imread
import os
from sklearn.model_selection import train_test_split

from ex4.adaboost import AdaBoost


def find_threshold(D, X, y, sign, j):
    """
    Finds the best threshold.
    D =  distribution
    S = (X, y) the data
    """
    # sort the data so that x1 <= x2 <= ... <= xm
    sort_idx = np.argsort(X[:, j])
    X, y, D = X[sort_idx], y[sort_idx], D[sort_idx]

    thetas = np.concatenate([[-np.inf], (X[1:, j] + X[:-1, j]) / 2, [np.inf]])
    minimal_theta_loss = np.sum(D[y == sign])  # loss of the smallest possible theta
    losses = np.append(minimal_theta_loss, minimal_theta_loss - np.cumsum(D * (y * sign)))
    min_loss_idx = np.argmin(losses)
    return losses[min_loss_idx], thetas[min_loss_idx]


class DecisionStump(object):
    """
    Decision stump classifier for 2D samples
    """

    def __init__(self, D, X, y):
        self.theta = 0
        self.j = 0
        self.sign = 0
        self.train(D, X, y)

    def train(self, D, X, y):
        """
        Train the classifier over the sample (X,y) w.r.t. the weights D over X
        Parameters
        ----------
        D : weights over the sample
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        """
        loss_star, theta_star = np.inf, np.inf
        for sign, j in product([-1, 1], range(X.shape[1])):
            loss, theta = find_threshold(D, X, y, sign, j)
            if loss < loss_star:
                self.sign, self.theta, self.j = sign, theta, j
                loss_star = loss

    def predict(self, X):
        """
        Parameters
        ----------
        X : shape=(num_samples, num_features)
        Returns
        -------
        y_hat : a prediction vector for X shape=(num_samples)
        """
        y_hat = self.sign * ((X[:, self.j] <= self.theta) * 2 - 1)
        return y_hat


def decision_boundaries(classifier, X, y, num_classifiers=1, weights=None):
    """
    Plot the decision boundaries of a binary classfiers over X \subseteq R^2

    Parameters
    ----------
    classifier : a binary classifier, implements classifier.predict(X)
    X : samples, shape=(num_samples, 2)
    y : labels, shape=(num_samples)
    title_str : optional title
    weights : weights for plotting X
    """
    cm = ListedColormap(['#AAAAFF', '#FFAAAA'])
    cm_bright = ListedColormap(['#0000FF', '#FF0000'])
    h = .003  # step size in the mesh
    # Plot the decision boundary.
    x_min, x_max = X[:, 0].min() - .2, X[:, 0].max() + .2
    y_min, y_max = X[:, 1].min() - .2, X[:, 1].max() + .2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()], num_classifiers)
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=cm)
    # Plot also the training points
    if weights is not None:
        plt.scatter(X[:, 0], X[:, 1], c=y, s=weights, cmap=cm_bright)
    else:
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks([])
    plt.yticks([])
    plt.title(f'num classifiers = {num_classifiers}')
    plt.draw()


def generate_data(num_samples, noise_ratio):
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X = np.random.rand(num_samples, 2) * 2 - 1
    radius = 0.5 ** 2
    in_circle = np.sum(X ** 2, axis=1) < radius
    y = np.ones(num_samples)
    y[in_circle] = -1
    y[np.random.choice(num_samples, int(noise_ratio * num_samples))] *= -1

    return X, y


def create_training_test_sets(num_samples_train, num_samples_test, noise_ratio):
    """
    Generate 5000 samples without noise (i.e. noise ratio=0).
    Generate another 200 samples without noise ("test set")
    :param num_samples_train: number of samples to generate for training set
    :param num_samples_test: number of samples to generate for test set
    :param noise_ratio: invert the label for this ratio of the samples
    :return: Train set and Test set
    """
    X_train, y_train = generate_data(num_samples_train, noise_ratio)
    X_test, y_test = generate_data(num_samples_test, noise_ratio)
    return X_train, y_train, X_test, y_test


def train_classifier(X_train, y_train, T):
    """
    Use the DecisionStump weak learner as a classifier, and T = 500.
    Train an Adaboost classifier over the training data.
    :param X_train: Train set data
    :param y_train: Train set labels
    :param T: T: the number of base learners to learn
    :return: learned classifier
    """
    WL = DecisionStump
    classifier = AdaBoost(WL, T)
    classifier.train(X_train, y_train)
    return classifier


def calculate_error(classifier, X_train, y_train, X_test, y_test, T):
    """
    Calculate the training error and test error, as a function of T.
    :param classifier: learned classifiers
    :param X_train: Train set data
    :param y_train: Train set labels
    :param X_test: Test set data
    :param y_test: Test set labels
    :param T: the number of base learners to learn
    :return: training error and test error
    """
    training_error = []
    test_error = []
    for t in range(1, T + 1):
        training_error.append(classifier.error(X_train, y_train, t))
        test_error.append(classifier.error(X_test, y_test, t))
    t = np.linspace(1, T, T)
    return training_error, test_error, t


def plot_error(training_error, test_error, t):
    """
    Plot the training error and test error, as a function of T.
    Plot the two curves on the same figure.
    :param training_error: Calculated training error
    :param test_error: Calculated test error
    :param t: the range of T (number of classifiers)
    :return:
    """
    plt.title("Training set error and test set error as a function of T")
    plt.xlabel("T")
    plt.ylabel("Error")
    plt.plot(t, training_error, label="Training error")
    plt.plot(t, test_error, label="Test error")
    plt.legend()
    plt.savefig('training and test error as func of T noise_ratio=0.01.png')
    plt.figure(figsize=(20, 10))
    plt.show()


def plot_error_by_T(classifier, X_train, y_train, X_test, y_test, T_array):
    """
    Plot the training error and test error, as a function of T, with with T_array= [5, 10, 50, 100, 200, 500]
    :param classifier: learned classifiers
    :param X_train: Train set data
    :param y_train: Train set labels
    :param X_test: Test set data
    :param y_test: Test set labels
    :param T_array: Array of 5 different number of base learners to learn
    :return:
    """
    training_error0, test_error0, t0 = calculate_error(classifier, X_train, y_train, X_test, y_test, T_array[0])
    training_error1, test_error1, t1 = calculate_error(classifier, X_train, y_train, X_test, y_test, T_array[1])
    training_error2, test_error2, t2 = calculate_error(classifier, X_train, y_train, X_test, y_test, T_array[2])
    training_error3, test_error3, t3 = calculate_error(classifier, X_train, y_train, X_test, y_test, T_array[3])
    training_error4, test_error4, t4 = calculate_error(classifier, X_train, y_train, X_test, y_test, T_array[4])
    training_error5, test_error5, t5 = calculate_error(classifier, X_train, y_train, X_test, y_test, T_array[5])

    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(20, 10))
    ax1.plot(t0, training_error0, label="Training error")
    ax1.plot(t0, test_error0, label="Test error")
    ax2.plot(t1, training_error1, label="Training error")
    ax2.plot(t1, test_error1, label="Test error")
    ax3.plot(t2, training_error2, label="Training error")
    ax3.plot(t2, test_error2, label="Test error")
    ax4.plot(t3, training_error3, label="Training error")
    ax4.plot(t3, test_error3, label="Test error")
    ax5.plot(t4, training_error4, label="Training error")
    ax5.plot(t4, test_error4, label="Test error")
    ax6.plot(t5, training_error5, label="Training error")
    ax6.plot(t5, test_error5, label="Test error")
    plt.legend()
    ax1.set(xlabel='T', ylabel='Error')
    ax2.set(xlabel='T', ylabel='Error')
    ax3.set(xlabel='T', ylabel='Error')
    ax4.set(xlabel='T', ylabel='Error')
    ax5.set(xlabel='T', ylabel='Error')
    ax6.set(xlabel='T', ylabel='Error')
    plt.savefig('training and test error as func of T array .png')
    plt.show()


def plot_Q11(classifier, X_test, y_test, T_array):
    """
    Plot the decisions of the learned classifiers with T= [5, 10, 50, 100, 200, 500] together with
    the test data.
    :param classifier: learned classifiers
    :param X_test: test set data
    :param y_test: test set labels
    :param T_array: Array of 5 different number of base learners to learn
    :return:
    """
    for t, i in zip(T_array, range(1, 7)):
        plt.subplot(3, 2, i)
        decision_boundaries(classifier, X_test, y_test, t)
    plt.savefig('decisions of the learned classifiers with T noise_ratio=0.01.png')
    plt.figure(figsize=(40, 20))
    plt.show()


def Q12(classifier, X_train, y_train, test_error, T):
    """
    Out of the different values we used for T, find the one that minimizes the test error.
    Find its test error.
    Plot the decision boundaries of this classifier together with the training data.
    :param classifier: learned classifiers
    :param X_train: Train set data
    :param y_train: Train set labels
    :param X_test: Test set data
    :param y_test: Test set labels
    :param T: number of base learners to learn
    :return:
    """
    T_array = np.arange(T)
    test_error = np.array(test_error)
    min_test_error_index = np.argmin(test_error[T_array])
    T_min_test_error = T_array[min_test_error_index]
    min_error = test_error[T_min_test_error]
    print("minimal test error is :", min_error, "with T : ", T_min_test_error)
    decision_boundaries(classifier, X_train, y_train, T_min_test_error)
    plt.savefig('decision boundaries of the best classifier noise_ratio=0.01.png')
    plt.figure(figsize=(40, 20))
    plt.show()


def Q13(classifier, X_train, y_train, T):
    """
    Take the weights of the samples in the last iteration of the training.
    Plot the training set with size proportional to its weight in D[T] , and color
    that indicates its label and normalize D: D = D / np.max(D) * 10.
    :param classifier: learned classifiers
    :param X_train: Train set data
    :param y_train: Train set labels
    :param X_test: Test set data
    :param y_test: Test set labels
    :param T: number of base learners to learn
    :return:
    """
    last_iter_weights = classifier.D[- 1]
    normalized_weights = last_iter_weights / np.max(last_iter_weights) * 10
    decision_boundaries(classifier, X_train, y_train, T, normalized_weights)
    plt.savefig('decision boundaries of the classifier with weights noise_ratio=0.01.png')
    plt.figure(figsize=(40, 20))
    plt.show()


def Q14(num_samples_train, num_samples_test, noise_ratio, T):
    """
    :param num_samples_train: number of samples to generate for training set
    :param num_samples_test: number of samples to generate for test set
    :param noise_ratio: invert the label for this ratio of the samples
    :param T: number of base learners to learn
    :return:
    """
    X_train, y_train, X_test, y_test = create_training_test_sets(num_samples_train, num_samples_test, noise_ratio)
    classifier = train_classifier(X_train, y_train, T)
    # Q10:
    training_error, test_error, t = calculate_error(classifier, X_train, y_train, X_test, y_test, T)
    plot_error(training_error, test_error, t)
    # Q11:
    T_array = [5, 10, 50, 100, 200, 500]
    plot_Q11(classifier, X_test, y_test, T_array)
    # extra
    # plot_error_by_T(classifier,X_train, y_train,X_test, y_test, T_array)
    # Q12:
    Q12(classifier, X_train, y_train, test_error, T)
    # Q13:
    Q13(classifier, X_train, y_train, T)


num_samples_train, num_samples_test, T = 5000, 200, 500
# X_train, y_train, X_test, y_test = create_training_test_sets(num_samples_train, num_samples_test, noise_ratio)
# classifier = train_classifier(X_train, y_train)
# # Q10:
# training_error, test_error, t = calculate_error(classifier, X_train, y_train, X_test, y_test, T)
# plot_error(training_error, test_error, t)
# # Q11:
# T_array = [5, 10, 50, 100, 200, 500]
# plot_Q11(classifier, X_test, y_test, T_array)
# # extra
# # plot_error_by_T(classifier,X_train, y_train,X_test, y_test, T_array)
# # Q12:
# Q12(classifier, X_train, y_train, test_error, T)
# # Q13:
# Q13(classifier, X_train, y_train, T)
noise_ratio = 0.01
Q14(num_samples_train, num_samples_test, noise_ratio, T)
