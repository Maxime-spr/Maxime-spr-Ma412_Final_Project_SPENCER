#PCA
import numpy as np
import matplotlib.pyplot as plt
import utils
from utils import draw_line
# Load the dataset from 'data.npy'
data = np.load('data.npy')

# Take the first two features for visualization 
X = data[:, :2]

# Visualize the 2D projection
plt.figure()
plt.scatter(X[:, 0], X[:, 1], facecolors='none', edgecolors='b')
plt.title('Original Dataset (2D Projection)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.xlim(-10, 80)
plt.ylim(-10, 60)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

def feature_normalize(X):
    # Compute mean and standard deviation for each feature
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    # Normalize the features
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma

def pca(X):
    # Compute the covariance matrix
    m, n = X.shape
    sigma = X.T.dot(X) / m

    # Perform SVD
    U, S, V = np.linalg.svd(sigma)

    return U, S, V


# Normalize the features
X_norm, mu, sigma = feature_normalize(X)

# Run PCA
U, S, V = pca(X_norm)

# Display the results
print("Principal Components (U):")
print(U)
print("\nSingular Values (S):")
print(S)
print("\nUnitary Matrices (V):")
print(V)

plt.figure()
utils.draw_line(mu, mu + 1.5 * S[0] * U[:, 0].T)
utils.draw_line(mu, mu + 1.5 * S[1] * U[:, 1].T)
plt.show()

print('Top eigenvector:')
print('U = ', U[:, 0])


#Dimension reduction

def project_data(X, U, K):

    Z = X @ U[:, :K]
    return Z

def recover_data(Z, U, K):

    X_rec = Z @ U[:, :K].T
    return X_rec

# Plot the normalized dataset 
plt.figure()
plt.scatter(X_norm[:, 0], X_norm[:, 1], facecolors='none', edgecolors='b')
plt.xlim(-4, 3)
plt.ylim(-4, 3)
plt.gca().set_aspect('equal', adjustable='box')

# Project the data onto K = 1 dimension
K = 1
Z = project_data(X_norm, U, K)
print('Projection of the first example:', Z[0, ])


X_rec = recover_data(Z, U, K)
print('Approximation of the first example:', X_rec[0, ])


# Draw lines connecting the projected points to the original points
plt.scatter(X_rec[:, 0], X_rec[:, 1], facecolors='none', edgecolors='r')
for i in range(X_norm.shape[0]):
    draw_line(X_norm[i, :], X_rec[i, :], dash=True)
axes = plt.gca()
axes.set_xlim([-3, 5])
axes.set_ylim([-3, 5])
axes.set_aspect('equal', adjustable='box')
plt.show()


# Visualize a subset of the data
plt.figure()
plt.scatter(data[:100, 0], data[:100, 1], facecolors='none', edgecolors='b')
plt.title('Original Dataset (Subset)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.xlim(-10, 80)
plt.ylim(-10, 60)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# Normalize the features
data_norm, mu_data, sigma_data = feature_normalize(data)

# Run PCA
U_data, S_data, V_data = pca(data_norm)

# Visualize the principal components
plt.figure()
plt.scatter(data_norm[:, 0], data_norm[:, 1], facecolors='none', edgecolors='b')
utils.draw_line(mu_data, mu_data + 1.5 * S_data[0] * U_data[:, 0].T)
utils.draw_line(mu_data, mu_data + 1.5 * S_data[1] * U_data[:, 1].T)
plt.title('PCA: Principal Components')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.xlim(-30, 70)
plt.ylim(-100, 20)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# Project the data onto K dimensions 
K_data = 2
Z_data = project_data(data_norm, U_data, K_data)

# Recover the data from the projected data
data_rec = recover_data(Z_data, U_data, K_data)

# Visualize the recovered data
plt.figure()
plt.scatter(data_rec[:, 0], data_rec[:, 1], facecolors='none', edgecolors='r')
plt.title('PCA: Recovered Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
