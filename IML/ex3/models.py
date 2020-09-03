import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


class Perceptron:
    def __init__(self, model=None):
        self.model = model

    def fit(self, X, y):
        """
        this method learns the parameters of the model and stores the trained model (namely, the variables that
        define hypothesis chosen) in self.model.
        :param X: a training set
        :param y: labels
        :return:
        """
        m, d = X.shape
        w = np.zeros((d + 1, 1))
        homogeneus_X = np.ones((m, d + 1))
        homogeneus_X[:m, :d] = X
        i = 0
        # while loop which stops when the classifier fits correctly all the training set.
        while (True):
            w = w + y[i] * homogeneus_X[i][:, np.newaxis]
            cond = ((y[:, np.newaxis] * np.dot(homogeneus_X, w)) <= 0).T[0]
            cond_where = np.argwhere(cond)
            is_cond = len(cond_where)
            if is_cond > 0:
                i = cond_where[0][0]
            else:
                self.model = w
                break

    def predict(self, X):
        """
        This function predicts the label of each sample.
        :param X: an unlabeled test set X
        :return: Returns a vector of predicted labels y
    """
        m, d = X.shape
        homogeneus_X = np.ones((m, d + 1))
        homogeneus_X[:m, :d] = X
        return np.sign(np.dot(homogeneus_X, self.model)).reshape(X.shape[0], )

    def score(self, X, y):
        """
        :param X: an unlabeled test set X
        :param y: true labels y of this test set
        :return: returns a dictionary with the following fields :
                        num samples: number of samples in the test set
                        error: error (misclassification) rate
                        accuracy: accuracy
                        FPR: false positive rate
                        TPR: true positive rate
                        precision: precision
                        recall: recall
        """
        num_samples = X.shape[0]
        self.fit(X, y)
        y_pred = self.predict(X)
        TP = np.count_nonzero((y + y_pred > 1))
        FN = np.count_nonzero((y - y_pred > 1))
        FP = np.count_nonzero((y - y_pred < -1))
        TN = np.count_nonzero((y + y_pred < -1))
        P = len(np.argwhere(y > 0))
        N = len(np.argwhere(y < 0))
        error = (FN + FP) / (P + N)
        accuracy = (TP + TN) / (P + N)
        FPR = FP / N
        TPR = TP / N
        precision = TP / (TP + FP)
        recall = TP / P
        dict = {'number samples': num_samples, 'error': error, 'accuracy': accuracy, 'FPR': FPR, 'TPR': TPR,
                'precision': precision
            , 'recall': recall}
        return dict


class LDA:

    def __init__(self, model=None):
        self.model = model

    def fit(self, X, Y):
        num_of_samples = X.shape[0]
        num_of_samples_minus_1 = np.count_nonzero(Y == -1)
        num_of_samples_1 = num_of_samples - num_of_samples_minus_1
        pi_minus_1 = num_of_samples_minus_1 / num_of_samples
        pi_1 = num_of_samples_1 / num_of_samples
        mu_minus_1 = self.estimated_mu(Y, -1, X, num_of_samples_minus_1)
        mu_1 = self.estimated_mu(Y, 1, X, num_of_samples_1)
        cov = self.estimated_cov(X, Y, mu_minus_1, mu_1)
        inversed_cov = np.linalg.inv(cov)
        # The functions delta+1,delta-1 are called discriminant functions: this classification rule predicts
        # the label, based on which of the two discriminant functions is larger at the sample x we
        # wish to classify.

        self.model = np.concatenate(
            (np.dot(inversed_cov ,mu_1) - np.dot(inversed_cov , mu_minus_1), - 0.5 * (np.dot(np.dot(mu_1.T, inversed_cov) , mu_1) -
                                    np.dot(np.dot(mu_minus_1.T , inversed_cov) , mu_minus_1)) + np.log(pi_1) - np.log(pi_minus_1)))

    def predict(self, X):
        """
        This function predicts the label of each sample.
        :param X: an unlabeled test set X
        :return: Returns a vector of predicted labels y
    """
        m, d = X.shape
        homogeneus_X = np.ones((m, d + 1))
        homogeneus_X[:m, :d] = X
        return np.sign(np.dot(homogeneus_X, self.model)).reshape(X.shape[0], )

    def score(self, X, y):
        """
        :param X: an unlabeled test set X
        :param y: true labels y of this test set
        :return: returns a dictionary with the following fields :
                        num samples: number of samples in the test set
                        error: error (misclassification) rate
                        accuracy: accuracy
                        FPR: false positive rate
                        TPR: true positive rate
                        precision: precision
                        recall: recall
        """
        num_samples = X.shape[0]
        self.fit(X, y)
        y_pred = self.predict(X)
        TP = np.count_nonzero((y + y_pred > 1))
        FN = np.count_nonzero((y - y_pred > 1))
        FP = np.count_nonzero((y - y_pred < -1))
        TN = np.count_nonzero((y + y_pred < -1))
        P = len(np.argwhere(y > 0))
        N = len(np.argwhere(y < 0))
        error = (FN + FP) / (P + N)
        accuracy = (TP + TN) / (P + N)
        FPR = FP / N
        TPR = TP / N
        precision = TP / (TP + FP)
        recall = TP / P
        dict = {'number samples': num_samples, 'error': error, 'accuracy': accuracy, 'FPR': FPR, 'TPR': TPR,
                'precision': precision
            , 'recall': recall}
        return dict

    def estimated_cov(self, X, Y, mu_minus_1, mu_1):
        m, n = X.shape
        cov = np.zeros((n, n))
        for (mu, y) in [(mu_minus_1, 0), (mu_1, 1)]:
            X_y = np.array(X[Y == y].T)
            center_X = X_y - np.array(mu)
            cov += np.dot(center_X ,center_X.T)
        return cov / (m - 2)

    def estimated_mu(self, Y, y, X, N_y):
        mu = np.sum(X[Y == y], axis=0)[:, np.newaxis]
        return mu / N_y


class SVM:
    def __init__(self, model=None):
        self.svm = SVC(C=1e10, kernel='linear')
        self.model = model

    def fit(self, X, y):
        """
        this method learns the parameters of the model and stores the trained model (namely, the variables that
        define hypothesis chosen) in self.model.
        :param X: a training set
        :param y: labels
        :return:
        """
        self.model = self.svm.fit(X, y)

    def predict(self, X):
        """
        This function predicts the label of each sample.
        :param X: an unlabeled test set X
        :return: Returns a vector of predicted labels y
    """
        # return self.model.predict(X)
        return self.svm.predict(X)

    def score(self, X, y):
        """
        :param X: an unlabeled test set X
        :param y: true labels y of this test set
        :return: returns a dictionary with the following fields :
                        num samples: number of samples in the test set
                        error: error (misclassification) rate
                        accuracy: accuracy
                        FPR: false positive rate
                        TPR: true positive rate
                        precision: precision
                        recall: recall
        """
        num_samples = X.shape[0]
        self.fit(X, y)
        y_pred = self.predict(X)
        TP = np.count_nonzero((y + y_pred > 1))
        FN = np.count_nonzero((y - y_pred > 1))
        FP = np.count_nonzero((y - y_pred < -1))
        TN = np.count_nonzero((y + y_pred < -1))
        P = len(np.argwhere(y > 0))
        N = len(np.argwhere(y < 0))
        error = (FN + FP) / (P + N)
        accuracy = (TP + TN) / (P + N)
        FPR = FP / N
        TPR = TP / N
        precision = TP / (TP + FP)
        recall = TP / P
        dict = {'number samples': num_samples, 'error': error, 'accuracy': accuracy, 'FPR': FPR, 'TPR': TPR,
                'precision': precision
            , 'recall': recall}
        return dict


class SOFT_SVM:
    def __init__(self, model=None):
        self.soft_svm = SVC(C=1, kernel='linear')
        self.model = model

    def fit(self, X, y):
        """
        this method learns the parameters of the model and stores the trained model (namely, the variables that
        define hypothesis chosen) in self.model.
        :param X: a training set
        :param y: labels
        :return:
        """
        self.model = self.soft_svm.fit(X, y)

    def predict(self, X):
        """
        This function predicts the label of each sample.
        :param X: an unlabeled test set X
        :return: Returns a vector of predicted labels y
    """
        # return self.model.predict(X)
        return self.soft_svm.predict(X)

    def score(self, X, y):
        """
        :param X: an unlabeled test set X
        :param y: true labels y of this test set
        :return: returns a dictionary with the following fields :
                        num samples: number of samples in the test set
                        error: error (misclassification) rate
                        accuracy: accuracy
                        FPR: false positive rate
                        TPR: true positive rate
                        precision: precision
                        recall: recall
        """
        num_samples = X.shape[0]
        self.fit(X, y)
        y_pred = self.predict(X)
        TP = np.count_nonzero((y + y_pred > 1))
        FN = np.count_nonzero((y - y_pred > 1))
        FP = np.count_nonzero((y - y_pred < -1))
        TN = np.count_nonzero((y + y_pred < -1))
        P = len(np.argwhere(y > 0))
        N = len(np.argwhere(y < 0))
        error = (FN + FP) / (P + N)
        accuracy = (TP + TN) / (P + N)
        FPR = FP / N
        TPR = TP / N
        precision = TP / (TP + FP)
        recall = TP / P
        dict = {'number samples': num_samples, 'error': error, 'accuracy': accuracy, 'FPR': FPR, 'TPR': TPR,
                'precision': precision
            , 'recall': recall}
        return dict

class Logistic:

    def __init__(self, model=None):
        self.logistic = LogisticRegression(solver='liblinear')
        self.model = model

    def fit(self, X, y):
        """
        this method learns the parameters of the model and stores the trained model (namely, the variables that
        define hypothesis chosen) in self.model.
        :param X: a training set
        :param y: labels
        :return:
        """
        self.model = self.logistic.fit(X, y)

    def predict(self, X):
        """
        This function predicts the label of each sample.
        :param X: an unlabeled test set X
        :return: Returns a vector of predicted labels y
    """
        return self.logistic.predict(X)

    def score(self, X, y):
        """
        :param X: an unlabeled test set X
        :param y: true labels y of this test set
        :return: returns a dictionary with the following fields :
                        num samples: number of samples in the test set
                        error: error (misclassification) rate
                        accuracy: accuracy
                        FPR: false positive rate
                        TPR: true positive rate
                        precision: precision
                        recall: recall
        """
        num_samples = X.shape[0]
        self.fit(X, y)
        y_pred = self.predict(X)
        TP = np.count_nonzero((y + y_pred > 1))
        FN = np.count_nonzero((y - y_pred > 1))
        FP = np.count_nonzero((y - y_pred < -1))
        TN = np.count_nonzero((y + y_pred < -1))
        P = len(np.argwhere(y > 0))
        N = len(np.argwhere(y < 0))
        error = (FN + FP) / (P + N)
        accuracy = (TP + TN) / (P + N)
        FPR = FP / N
        TPR = TP / N
        precision = TP / (TP + FP)
        recall = TP / P
        dict = {'number samples': num_samples, 'error': error, 'accuracy': accuracy, 'FPR': FPR, 'TPR': TPR,
                'precision': precision
            , 'recall': recall}
        return dict


class DecisionTree:
    def __init__(self, model=None):
        self.tree = DecisionTreeClassifier(max_depth=10)
        self.model = model

    def fit(self, X, y):
        """
        this method learns the parameters of the model and stores the trained model (namely, the variables that
        define hypothesis chosen) in self.model.
        :param X: a training set
        :param y: labels
        :return:
        """

        self.model = self.tree.fit(X, y)

    def predict(self, X):
        """
        This function predicts the label of each sample.
        :param X: an unlabeled test set X
        :return: Returns a vector of predicted labels y
    """
        return self.tree.predict(X)

    def score(self, X, y):
        """
        :param X: an unlabeled test set X
        :param y: true labels y of this test set
        :return: returns a dictionary with the following fields :
                        num samples: number of samples in the test set
                        error: error (misclassification) rate
                        accuracy: accuracy
                        FPR: false positive rate
                        TPR: true positive rate
                        precision: precision
                        recall: recall
        """
        num_samples = X.shape[0]
        self.fit(X, y)
        y_pred = self.predict(X)
        TP = np.count_nonzero((y + y_pred > 1))
        FN = np.count_nonzero((y - y_pred > 1))
        FP = np.count_nonzero((y - y_pred < -1))
        TN = np.count_nonzero((y + y_pred < -1))
        P = len(np.argwhere(y > 0))
        N = len(np.argwhere(y < 0))
        error = (FN + FP) / (P + N)
        accuracy = (TP + TN) / (P + N)
        FPR = FP / N
        TPR = TP / N
        precision = TP / (TP + FP)
        recall = TP / P
        dict = {'number samples': num_samples, 'error': error, 'accuracy': accuracy, 'FPR': FPR, 'TPR': TPR,
                'precision': precision
            , 'recall': recall}
        return dict