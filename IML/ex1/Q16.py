import numpy as np
import matplotlib.pyplot as plt

data = np.random.binomial(1, 0.25, (100000, 1000))
epsilon = [0.5, 0.25, 0.1, 0.01, 0.001]


def X_as_func_of_m(row):
    """

    :param row: The number of row to calculate estimated mean of all tosses up to m as a function of m
    :return: Estimated mean of all tosses up to m as a function of m
    """
    m = np.linspace(0, 1000, 1000)
    x = np.zeros_like(m)
    x[0] = np.mean(data[row][0])
    for i in range(1, len(m)):
        x[i] = np.mean(data[row, :int(m[i]) + 1])
    return m, x


def plot_result():
    """
    For the first 5 sequences of 1000 tosses (the first 5 rows in data"), plot the estimate mean of all tosses up to m
    as a function of m. 1 figure with 5 plots (each row in a different color).
    """
    for i in range(5):
        plt.title("Estimated mean of all tosses up to m as a function of m for row " + str(i))
        m, x = X_as_func_of_m(i)
        plt.plot(m, x, label="number of row " + str(i))
        plt.xlabel("Number of samples m")
        plt.ylabel("Estimate mean")
        plt.legend()
    plt.show()

    # Subplot
    # m0, x0 = X_as_func_of_m(0)
    # m1, x1 = X_as_func_of_m(1)
    # m2, x2 = X_as_func_of_m(2)
    # m3, x3 = X_as_func_of_m(3)
    # m4, x4 = X_as_func_of_m(4)

    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1)
    #
    # ax1.plot(m0, x0)
    # ax2.plot(m1, x1, 'tab:orange')
    # ax3.plot(m2, x2, 'tab:green')
    # ax4.plot(m3, x3, 'tab:red')
    # ax5.plot(m4, x4, 'tab:pink')
    #
    # ax1.set(xlabel='m', ylabel='X row 0')
    # ax2.set(xlabel='m', ylabel='X row 1')
    # ax3.set(xlabel='m', ylabel='X row 2')
    # ax4.set(xlabel='m', ylabel='X row 3')
    # ax5.set(xlabel='m', ylabel='X row 4')
    # plt.show()
    #
    # for ax in fig.get_axes():
    #     ax.label_outer()


def Chebyshev(epsilon, m):
    """
    Compute Chebyshev bound and for each epsilon, plot the upper bound as a function of m.
    :param epsilon:
    :param m: the number os samples (coin tosses)
    :return: upper bound
    """
    cheb = 1 / (4 * m * epsilon * epsilon)
    bound = np.where(cheb <= 1, cheb, 1)
    return bound


def Hoeffding(epsilon, m):
    """
    Compute Hoeffding bound and for each epsilon, plot the upper bound as a function of m
     :param epsilon:
    :param m: the number os samples (coin tosses)
    :return: upper bound
    """
    heof = 2 * np.exp(-2 * m * epsilon * epsilon)
    y = np.where(heof <= 1, heof, 1)
    return y


def percentage(eps):
    """
    :param eps: epsilon
    :return: percentage of sequences that satisfy the relevant equation as a function of m.
    """
    mean = 0.25
    num_of_seq = 100000
    per = np.zeros(1000)
    for c in range(1, 1001):
        x = np.mean(data[:, :c], axis=1)
        arr = np.where(np.abs(x - mean) >= eps)
        sum = len(arr[0])
        per[c-1] = sum / num_of_seq
    return per


def plot_bound():
    """
    For each bound (Chebyshev and Hoeding seen in class) and for each epsilon, plot the
    upper bound.
    """
    m = np.linspace(1, 1001, 1000)
    for i in range(len(epsilon)):
        plt.title("epsilon is " + str(epsilon[i]))
        y = Hoeffding(epsilon[i], m)
        bound = Chebyshev(epsilon[i], m)
        per = percentage(epsilon[i])
        plt.plot(m, y, label="Hoepding")
        plt.plot(m, bound, 'tab:pink', label="Chebyshev")
        plt.plot(m, per, 'tab:orange', label="Percentage")
        plt.xlabel("Number of samples m")
        plt.ylabel("Upper bound")
        plt.legend()
        plt.show()

plot_bound()
# plot_result()
