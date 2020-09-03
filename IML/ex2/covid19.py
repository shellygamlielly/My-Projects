import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def fit_linear_regression(X, y):
    """
    :param X: matrix `X` (numpy array with p rows and n columns)
    :param y:  respone vector `y` (numpy array with n rows)
    :return: two sets of values: the first is a numpy array of the coefcients vector `w`.
            And the second is a numpy array of the singular values of X.
    """
    s = np.linalg.svd(X, compute_uv=False)
    pseudo_inverse_X = np.linalg.pinv(X)
    w = np.dot(pseudo_inverse_X, y)
    return w, s

df = pd.read_csv('covid19_israel.csv')

# You should get three columns - day num, which count the days
# since first infection was identied in Israel, date, which represents the time of the event,
# and detected, which sums the number of detected cases up to this date.
date = df.iloc[:, 1].values
detected = df.iloc[:, 2].values
day_num = df.iloc[:, 0].values
# Create a new column in the DataFrame named "log detected" which holds the log of the
# "detected" column.
df_float = detected.astype(float)
# Declare a list that is to be converted into a column
log_d = np.log(df_float)
# Using 'log detected' as the column name  and equating it to the list
df.iloc[:, 2] = log_d
log_detected = df.iloc[:, 2].values

X = np.reshape(day_num, [len(day_num), 1])
y = np.reshape(log_detected, [len(log_detected), 1])
w,s=fit_linear_regression(X,y)

# Plot two graphs: the rst is the "log detected" as a function of "day num", and the second
# is "detected" as a function of "day num". On each graph, add the data as single points,
# and add the tted curve that you've estimated (think how to convert the linear result
# into an exponential result). Add those graphs to your answers PDF.

w = float(w)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("log_detected as a function of day num")
ax.set_xlabel("day num")
ax.set_ylabel("log_detected")
ax.plot(X, y, "r.", label='data')
ax.plot(X,w*X,"b", label='estimated curve ')
ax.legend()
plt.show()

w=np.exp(w)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("detected as a function of day num")
ax.set_xlabel("day num")
ax.set_ylabel("detected")
ax.plot(X,detected,"b.",label='data')
ax.plot(X,detected*w,"g",label='estimated curve ')
ax.legend()
plt.show()