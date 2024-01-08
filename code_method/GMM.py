# imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import scipy.linalg as la

# Load the dataset
data = np.load('data.npy')[:18, :2]
print("Shape of data:", data.shape)

# Choose a GMM with 2 components
m = np.zeros((2, 2))
m[0] = np.array([1, 2])
m[1] = np.array([5, 5])

S = np.zeros((2, 2, 2))
S[0] = np.array([[1, 0], [0, 1]])
S[1] = np.array([[1, 0], [0, 1]])

w = np.array([0.5, 0.5])

# Generate more data points per mixture component
N_split = 50  # number of data points per mixture component
N = N_split * 2  # total number of data points
x = []
y = []
for k in range(2):
    x_tmp, y_tmp = np.random.multivariate_normal(m[k], S[k], N_split).T
    x = np.hstack([x, x_tmp])
    y = np.hstack([y, y_tmp])

data = np.vstack([x, y])

# Visualize the dataset
plt.scatter(data[:, 0], data[:, 1], c='k', marker='o', alpha=0.3)
plt.title("Generated Dataset")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()

# Visualize the GMM components
X, Y = np.meshgrid(np.linspace(-2, 8, 100), np.linspace(-2, 8, 100))
pos = np.dstack((X, Y))

plt.title("Mixture components")
plt.scatter(data[:, 0], data[:, 1], c='k', marker='o', alpha=0.3)
plt.plot(m[:, 0], m[:, 1], 'or')

for k in range(2):
    mvn = multivariate_normal(m[k, :].ravel(), S[k, :, :])
    xx = mvn.pdf(pos)
    plt.contour(X, Y, xx, alpha=1.0, zorder=10)

plt.show()

# Build and visualize the GMM
plt.title("GMM")
plt.scatter(data[:, 0], data[:, 1], c='k', marker='o', alpha=0.3)

gmm = 0
for k in range(2):
    mix_comp = multivariate_normal(m[k, :].ravel(), S[k, :, :])
    gmm += w[k] * mix_comp.pdf(pos)

plt.contour(X, Y, gmm, alpha=1.0, zorder=10)
plt.show()



############
#EM
K = 2  # number of clusters

# Initialize means randomly
means = np.zeros((K, 2))
for k in range(K):
    means[k] = np.random.normal(size=(2,))

# Initialize covariance matrices to identity matrices
covs = np.zeros((K, 2, 2))
for k in range(K):
    covs[k] = np.eye(2)

# Initialize weights uniformly
weights = np.ones((K, 1)) / K

# Display the initial mean vectors
print("Initial mean vectors (one per row):\n" + str(means))

# EDIT THIS CELL
NLL = []  # log-likelihood of the GMM
gmm_nll = 0
NLL += [gmm_nll]  # <-- REPLACE THIS LINE

plt.figure()
plt.plot(x, y, 'ko', alpha=0.3)
plt.plot(means[:, 0], means[:, 1], 'oy', markersize=25)

for k in range(K):
    rv = multivariate_normal(means[k, :], covs[k, :, :])
    plt.contour(X, Y, rv.pdf(pos), alpha=1.0, zorder=10)

plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

# Set larger axis limits
plt.axis('equal')  # Set equal scaling
plt.xlim([-5, 15])  # Set X-axis limits
plt.ylim([-5, 15])  # Set Y-axis limits

# EDIT THIS CELL
r = np.zeros((K, N))  # will store the probabilities

for em_iter in range(100):
    means_old = means.copy()

    # E-step: update probabilities
    for k in range(K):
        rv = multivariate_normal(means[k, :], covs[k, :, :])
        r[k, :] = w[k] * rv.pdf(data.T)

    r /= np.sum(r, axis=0)

    # M-step
    N_k = np.sum(r, axis=1)

    for k in range(K):
        # update the means
        means[k, :] = np.sum((r[k, :, np.newaxis] * data.T), axis=0) / N_k[k]

        # update the covariances
        centered_data = data.T - means[k, :][np.newaxis, :]
        covs[k, :, :] = np.dot(r[k, :] * centered_data.T, centered_data) / N_k[k]

    # weights
    weights = N_k / N

    # log-likelihood
    gmm_nll = -np.sum(np.log(np.sum([w[k] * multivariate_normal(means[k, :], covs[k, :, :]).pdf(data.T)
                                     for k in range(K)], axis=0)))
    NLL += [gmm_nll]

    plt.figure()
    plt.plot(x, y, 'ko', alpha=0.3)
    plt.plot(means[:, 0], means[:, 1], 'oy', markersize=25)
    for k in range(K):
        rv = multivariate_normal(means[k, :], covs[k, :, :])
        plt.contour(X, Y, rv.pdf(pos), alpha=1.0, zorder=10)

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.text(x=3.5, y=8, s="EM iteration " + str(em_iter + 1))

    if la.norm(NLL[em_iter + 1] - NLL[em_iter]) < 1e-6:
        print("Converged after iteration ", em_iter + 1)
        break

# plot the final mixture model
plt.figure()
gmm = 0
for k in range(K):
    mix_comp = multivariate_normal(means[k, :].ravel(), covs[k, :, :])
    gmm += weights[k] * mix_comp.pdf(pos)

plt.plot(x, y, 'ko', alpha=0.3)
plt.contour(X, Y, gmm, alpha=1.0, zorder=10)
plt.xlim([-8, 8])
plt.ylim([-6, 6])

plt.figure()
plt.semilogy(np.arange(1, len(NLL) + 1), NLL)
plt.xlabel("EM iteration")
plt.ylabel("Negative log-likelihood")

idx = [0, 1, 9, em_iter]

for i in idx:
    plt.plot(i + 1, NLL[i], 'or')
