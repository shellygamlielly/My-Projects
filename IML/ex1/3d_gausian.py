import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import qr

mean = [0, 0, 0]
cov = np.eye(3)
x_y_z = np.random.multivariate_normal(mean, cov, 50000).T


def get_orthogonal_matrix(dim):
    H = np.random.randn(dim, dim)
    Q, R = qr(H)
    return Q


def plot_3d(x_y_z):
    '''
    plot points in 3D
    :param x_y_z: the points. numpy array with shape: 3 X num_samples (first dimension for x, y, z
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_y_z[0], x_y_z[1], x_y_z[2], s=1, marker='.', depthshade=False)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


def plot_2d(x_y):
    '''
    plot points in 2D
    :param x_y_z: the points. numpy array with shape: 2 X num_samples (first dimension for x, y
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_y[0], x_y[1], s=1, marker='.')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')


c_est = np.dot(x_y_z, x_y_z.T) / (50000 - 1)
print("Covariance matrix of data")
print(c_est)
# Q11 : Use the identity matrix as the covariance matrix to
# generate random points and than plot them
plot_3d(x_y_z)
plt.title("3d_covmat_identity")
# Q12 : Transform the data with the scaling matrix :
scaled_mat = np.diag(np.array([0.1, 0.5, 2]))
scaled_data = np.dot(scaled_mat, x_y_z)
scaled_cov = np.dot(scaled_mat, np.dot(cov, scaled_mat.T))
print("scaled covariance matrix")
print(scaled_cov)
scaled_est_cov = np.dot(scaled_mat, np.dot(c_est, scaled_mat.T))
print("scaled estimated covariance matrix")
print(scaled_est_cov)
# Plot the new points.
plot_3d(scaled_data)
plt.title("3d_covmat_scaled")

# Q13 : Multiply the scaled data by random orthogonal matrix.
orth_mat = get_orthogonal_matrix(3)
print("Orthogonal Matrix")
print(orth_mat)
orth_scaled_data = np.dot(orth_mat, scaled_data)
orth_scaled_mat = np.dot(orth_mat, np.dot(scaled_cov, orth_mat.T))
print("scaled orthogonal covariance matrix")
print(orth_scaled_mat)
orth_scaled_estimated_mat = np.dot(orth_mat, np.dot(scaled_est_cov, orth_mat.T))
print("scaled orthogonal estimated covariance matrix")
print(orth_scaled_estimated_mat)
# Plot the new points.
plot_3d(orth_scaled_data)
plt.title("3d_covmat_orth")

# plot_2d(x_y_z)

# Q14 : Plot the projection of the data to the x, y axes.
plot_2d(scaled_data)
plt.title("2D_covmat_scaled")
plot_2d(orth_scaled_data)
plt.title("2D_covmat_scaled_orth")

# Q15 : Only for points where 0.1>z>-0.4:
# Plot the projection of the points to the x, y axes.
cond = np.where((x_y_z[2] > -0.4) & (x_y_z[2] < 0.1))
new_x_y_z = x_y_z[0:2, cond]
plot_2d(new_x_y_z)
plt.title("for 0.1>z>-0.4 - the projection of the points to the x, y axes")

plt.show()
