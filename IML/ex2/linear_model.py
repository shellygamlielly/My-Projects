import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns


def fit_linear_regression(X, y):
    """
    :param X: matrix `X` (numpy array with p rows and n columns)
    :param y:  respone vector `y` (numpy array with n rows)
    :return: two sets of values: the first is a numpy array of the coefcients vector `w`.
            And the second is a numpy array of the singular values of X.
    """
    # Vector(s) with the singular values, within each vector sorted in descending order.
    s = np.linalg.svd(X, compute_uv=False)
    pseudo_inverse_X = np.linalg.pinv(X)
    w = np.dot(pseudo_inverse_X, y)
    return w, s


def predict(X, w):
    """
    :param X: a design matrix `X` (numpy array with p rows and m columns)
    :param w: coefcients vector `w`
    :return: a numpy array with the predicted value by the model.
    """
    y = np.dot(X, w)
    return y


def mse(response_vec, prediction_vec):
    """
    :param response_vec:response vector numpy arrays
    :param prediction_vec:prediction vector numpy arrays
    :return: the MSE over the received samples.
    """
    MSE = np.square(np.subtract(response_vec, prediction_vec)).mean()
    return MSE


def load_data(path):
    """
    Given a path to the csv file this function loads the dataset and performs all the needed preprocessing so to get a
    valid design matrix.
    :param path: a path to the csv file
    :return: the dataset after the preprocessing.
    """
    df = pd.read_csv(path)
    # drop the NaN
    df = df.dropna()
    # invalid values for features : price ,bathrooms and bedrooms .
    # house prices and number of rooms cannot be negative
    df = df.drop(df[df['price'] <= 0].index)
    df = df.drop(df[df['bathrooms'] <= 0].index)
    df = df.drop(df[df['bedrooms'] <= 0].index)
    # remove characters after the year and month
    df.update(df['date'].str[0:8])
    # is renovated?
    df.update(df['yr_renovated'].astype(bool))
    # change those categorical features into One Hot encoding (or "dummy variables")
    dummy = pd.get_dummies(df['zipcode'])
    # add to data frame
    df = pd.concat([df,dummy ], axis=1)
    preprocessed_df = df.drop(['zipcode', 'id', 'lat', 'long',
                                                'sqft_living15', 'sqft_lot15'], axis=1)
    # preprocessed_df = df.drop(['zipcode', 'id', 'lat', 'long','date',
    #                                             'sqft_living15', 'sqft_lot15'], axis=1)
    X = preprocessed_df.drop('price', axis=1)
    X = X.iloc[:, :-1].values
    y = preprocessed_df.iloc[:, 1].values
    return X, y,preprocessed_df


def create_dataset():
    """
    :return: feature matrix X transpose with homogeneous coordinate ,and the response vector y with new axis.
    """
    X, y,preprocessed_data = load_data('kc_house_data.csv')
    Y = y[:, np.newaxis]
    h, w = X.shape
    #add homogeneous coordinate
    X_T = np.ones((h, w + 1))
    X_T[:, 1:w + 1] = X.astype(np.float64)
    return X_T, Y


def create_model(X_T, Y):
    """
    This function splits the data into train- and test-sets randomly, such that the size of
    the test set is 1/4 of the total data.
    Next, over the training set performs the following: For every p fit a model based on the first p% of the training set.
    Then using the predict function test the performance of the fitted model on the test-set.
    It plots the MSE over the test set as a function of p%.
    :param X_T: feature matrix - of all relevant features that affects the price
    :param Y: the response vector - price.
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X_T, Y, test_size=0.25, random_state=0)
    size_train_X = X_train.shape[0]
    size_train_y = Y_train.shape[0]
    # size_test_X = X_test.shape[0]
    # size_test_Y = Y_test.shape[0]
    loss_test = []
    for p in range(1,101):
        size_X_train = int(size_train_X*p/100)
        size_Y_train = int(size_train_y*p/100)
        # size_X_test = int(size_test_X*p/100)
        # size_Y_test = int(size_test_Y*p/100)
        cut_train_set_X = X_train[:size_X_train]
        cut_train_set_y = Y_train[:size_Y_train]
        # cut_test_set_X = X_test[:size_X_test]
        # cut_test_set_y = Y_test[:size_Y_test]
        w_train = fit_linear_regression(cut_train_set_X,cut_train_set_y)[0]
        y_test_predicted = predict(X_test,w_train)
        loss_p_test = mse(Y_test,y_test_predicted)
        loss_test.append(loss_p_test)
    _mse = np.array(loss_test)
    x = np.linspace(1, 101, 100)
    plt.title("MSE as function of p% of the training set")
    plt.plot(x,_mse)
    plt.xlabel("p% of the training set")
    plt.ylabel("MSE")
    plt.show()


X_T, Y = create_dataset()
create_model(X_T, Y)

def plot_singular_values(singular_values):
    """
    This function receives a collection of singular values and plots them in descending order.
    :param singular_values: a collection of singular values
    """
    l = len(singular_values)
    x = np.linspace(1, l+1, l)
    singular_values = singular_values*x
    # singular_values_des = np.sort(singular_values)[::-1]
    plt.title("scree-plot")
    plt.plot(x,singular_values)
    plt.xlabel("Index number")
    plt.ylabel("singular values")
    plt.show()

s = fit_linear_regression(X_T,Y)[1]
print(s)
plot_singular_values(s)

def calculate_corr(feature_value,response_vector):
    """
    This function calculates the pearson correlation between the feature the the price.
    :param feature_value: feature value
    :param response_vector: response vector - price
    :return: pearson correlation
    """
    cov = np.cov(feature_value,response_vector)
    s1 = np.sqrt(np.var(feature_value))
    s2 = np.sqrt(np.var(response_vector))
    s= np.dot(s1,s2)
    pearson_corr = cov/s
    corr  = pearson_corr[0,1]
    return corr

def feature_evaluation(matrix,response_vector):
    """
    This function plots for every non-categorical feature, a graph (scatter plot) of
    the feature values and the response values.
    It then also computes and shows on the graph the Pearson Correlation between the feature and the response.
    :param matrix: data frame - the data after prepossessing.
    :param response_vector:response vector - price
    """
    feature_value_grade = matrix['grade']
    feature_value_bedrooms = matrix['bedrooms']
    feature_yr_built = matrix['yr_built']
    pearson_corr_grade = calculate_corr(feature_value_grade,response_vector)
    pearson_corr_bedrooms = calculate_corr(feature_value_bedrooms,response_vector)
    pearson_corr_waterfront = calculate_corr(feature_yr_built,response_vector)
    sns.catplot(x="grade", y="price", jitter=False, data=matrix)
    plt.text(2, 6, pearson_corr_grade, fontsize=10)
    plt.title('The feature is grade ')

    sns.catplot(x="bedrooms", y="price", jitter=False, data=matrix)
    plt.text(2, 6, pearson_corr_bedrooms, fontsize=10)
    plt.title('The feature is bedrooms ')

    sns.catplot(x="yr_built", y="price", jitter=False, data=matrix)
    plt.text(2, 6, pearson_corr_waterfront, fontsize=10)
    plt.title('The feature is yr_built ')

    plt.show()
X,y,preprocessed_data = load_data('kc_house_data.csv')
feature_evaluation(preprocessed_data,y)
