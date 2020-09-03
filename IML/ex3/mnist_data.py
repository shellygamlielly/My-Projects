import tensorflow as tf
import matplotlib.pyplot as plt
from models import *
# from ex3.models import *
from sklearn.neighbors import KNeighborsClassifier
import time


def create_data():
    """
    Download MNIST (a large database of handwritten digits that is commonly used for training
    various image processing systems) locally.
    :return: training set and test set
    """
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return x_train, y_train, x_test, y_test


def create_trainSet_testSet():
    """
    Create a binary classifier of '0' and '1' digits.
    :return:
    """
    x_train, y_train, x_test, y_test = create_data()
    train_images = np.logical_or((y_train == 0), (y_train == 1))
    test_images = np.logical_or((y_test == 0), (y_test == 1))
    x_train, y_train = x_train[train_images], y_train[train_images]
    x_test, y_test = x_test[test_images], y_test[test_images]
    return x_train, y_train, x_test, y_test


def Q12(x_train, y_train):
    """
    This function draws 3 images of samples labeled with '0' and 3 images of samples labeled with '1'.
    :param x_train: train set
    :param y_train: labels of train set
    :return:
    """
    images_0 = x_train[y_train == 0]
    images_1 = x_train[y_train == 1]
    for i in range(3):
        plt.imshow(images_0[i], cmap='Greys')
        plt.show()
    for i in range(3):
        plt.imshow(images_1[i], cmap='Greys')
        plt.show()


def rearrange_data(X):
    """
    This function rearrange the data that given a data as a tensor of size m x 28 x 28,
    returns a new matrix of size m x 784 with the same data.
    :param X: a tensor of size m x 28 x 28
    :return:a new matrix of size m x 784 with the same data
    """
    return X.reshape(X.shape[0], 28 * 28)


def draw_points(m, x_train, y_train):
    """
    Draw m training points by choosing uniformly and at random from the
    train set.
    :param m: number of samples
    :param x_train: training set
    :param y_train: training set labels
    :return: m samples of training set
    """
    rand_ind = np.random.choice(x_train.shape[0], size=m, replace=False)
    return x_train[rand_ind], y_train[rand_ind]


def create_set(m):
    """
    Draw m training points by choosing uniformly and at random from the train set.
    The training data should always have points from two classes. So
    if you draw a training set where no point has yi = 1 or no point has yi = -1 then
    just draw a new dataset instead, until you get points from both types.
    :param m:
    :return:
    """
    x_train, y_train, x_test, y_test = create_trainSet_testSet()
    x_train = rearrange_data(x_train)
    x_test = rearrange_data(x_test)
    x_train_m, y_train_m = draw_points(m, x_train, y_train)
    x_test_m, y_test_m = draw_points(m, x_test, y_test)
    while True:
        if 1 not in y_train_m or 0 not in y_train_m:
            x_train_m, y_train_m = draw_points(m, x_train, y_train)
        else:
            break
    return x_train_m, y_train_m, x_test_m, y_test_m


def calc_acc(testX, testY, y_pred):
    """
    Calculate the accuracy of classifiers (the fraction of test points that is classified
    correctly) on the test set.
    :param testX: test set
    :param testY: test set labels
    :return: The accuracy of the classifier
    """
    correct_classification = np.array(testY - y_pred == 0).sum()
    number_of_samples = testX.shape[0]
    acc = correct_classification / number_of_samples
    return acc


def Q14():
    """
    For each m [50; 100; 300; 500] , repeat the above procedure 50 times and save the
    elapsed running time and the accuracy of each classifier. Finally, plot the mean accuracy
    as function of m for each of the algorithms (SVM, Logistic regression, decision tree and
    nearest neighbors).
    :return:
    """
    svm = SOFT_SVM()
    logistic = Logistic()
    decision_tree = DecisionTree()
    knbrs = KNeighborsClassifier(n_neighbors=3, algorithm='auto')
    M = [50, 100, 300, 500]
    svm_acc = []
    logistic_acc = []
    decision_tree_acc = []
    knbrs_acc = []
    m_svm_acc = []
    m_logistic_acc = []
    m_decision_tree_acc = []
    m_knbrs_acc = []
    m_svm_time = []
    m_logistic_time = []
    m_decision_tree_time = []
    m_knbrs_time = []
    svm_time = []
    logistic_time = []
    decision_tree_time = []
    knbrs_time = []
    for m in M:
        x_train_m, y_train_m, x_test_m, y_test_m = create_set(m)
        for iter in range(50):
            start_svm = time.time()
            svm.fit(x_train_m, y_train_m)
            svm_pred = svm.predict(x_test_m)
            end_svm = time.time()
            start_logistic = time.time()
            logistic.fit(x_train_m, y_train_m)
            logistic_pred = logistic.predict(x_test_m)
            end_logistic = time.time()
            start_decision_tree = time.time()
            decision_tree.fit(x_train_m, y_train_m)
            decision_tree_pred = decision_tree.predict(x_test_m)
            end_decision_tree = time.time()
            start_knbrs = time.time()
            knbrs.fit(x_train_m, y_train_m)
            knbrs_pred = knbrs.predict(x_test_m)
            end_knbrs = time.time()
            svm_acc.append(calc_acc(x_test_m, y_test_m, svm_pred))
            logistic_acc.append(calc_acc(x_test_m, y_test_m, logistic_pred))
            decision_tree_acc.append(calc_acc(x_test_m, y_test_m, decision_tree_pred))
            knbrs_acc.append(calc_acc(x_test_m, y_test_m, knbrs_pred))

            svm_time.append(end_svm - start_svm)
            logistic_time.append(end_logistic - start_logistic)
            decision_tree_time.append(end_decision_tree - start_decision_tree)
            knbrs_time.append(end_knbrs - start_knbrs)

        mean_svm_acc = np.mean(svm_acc)
        mean_logistic_acc = np.mean(logistic_acc)
        mean_decision_acc = np.mean(decision_tree_acc)
        mean_knbrs_acc = np.mean(knbrs_acc)

        mean_svm_time = np.mean(svm_time)
        mean_logistic_time = np.mean(logistic_time)
        mean_decision_tree_time = np.mean(decision_tree_time)
        mean_knbrs_time = np.mean(knbrs_time)

        m_svm_acc.append(mean_svm_acc)
        m_logistic_acc.append(mean_logistic_acc)
        m_decision_tree_acc.append(mean_decision_acc)
        m_knbrs_acc.append(mean_knbrs_acc)

        m_svm_time.append(mean_svm_time)
        m_logistic_time.append(mean_logistic_time)
        m_decision_tree_time.append(mean_decision_tree_time)
        m_knbrs_time.append(mean_knbrs_time)
    return m_svm_acc, m_logistic_acc, m_decision_tree_acc, m_knbrs_acc, m_svm_time, m_logistic_time, m_decision_tree_time, m_knbrs_time


def plot_acc(m_svm_acc, m_logistic_acc, m_decision_tree_acc, m_knbrs_acc, M):
    """
    Plot the mean accuracy as function of m for each of the algorithms (SVM, Logistic regression, decision tree and
    nearest neighbors).
    :param m_svm_acc: Soft SVC classifier accuracy
    :param m_logistic_acc: Logistic classifier accuracy
    :param m_decision_tree_acc: DecisionTree classifier accuracy
    :param m_knbrs_acc: KNeighborsClassifier classifier accuracy
    :param M: a list of number of samples
    :return:
    """
    plt.title("The accuracy of each classifier as function of m")
    plt.ylabel("Classifier Accuracy")
    plt.xlabel("m")
    plt.plot(M, m_svm_acc, label="Soft SVC classifier accuracy")
    plt.plot(M, m_logistic_acc, label="Logistic classifier accuracy")
    plt.plot(M, m_decision_tree_acc, label="DecisionTree classifier accuracy")
    plt.plot(M, m_knbrs_acc, label="KNeighborsClassifier classifier accuracy")
    plt.legend()
    plt.show()


def plot_time(m_svm_time, m_logistic_time, m_decision_tree_time, m_knbrs_time, M):
    """
    Plot the mean time as function of m for each of the algorithms (SVM, Logistic regression, decision tree and
    nearest neighbors).
    :param m_svm_acc: Soft SVC classifier accuracy
    :param m_logistic_acc: Logistic classifier accuracy
    :param m_decision_tree_acc: DecisionTree classifier accuracy
    :param m_knbrs_acc: KNeighborsClassifier classifier accuracy
    :param M: a list of number of samples
    :return:
    """
    plt.title("The running time of each classifier as function of m")
    plt.ylabel("Classifier Running Time")
    plt.xlabel("m")
    plt.plot(M, m_svm_time, label="SVC classifier time")
    plt.plot(M, m_logistic_time, label="Logistic classifier time")
    plt.plot(M, m_decision_tree_time, label="DecisionTree classifier time")
    plt.plot(M, m_knbrs_time, label="KNeighborsClassifier classifier time")
    plt.legend()
    plt.show()


m_svm_acc, m_logistic_acc, m_decision_tree_acc, m_knbrs_acc, m_svm_time, m_logistic_time, m_decision_tree_time, m_knbrs_time = Q14()
M = [50, 100, 300, 500]
plot_acc(m_svm_acc, m_logistic_acc, m_decision_tree_acc, m_knbrs_acc, M)
plot_time(m_svm_time, m_logistic_time, m_decision_tree_time, m_knbrs_time, M)
