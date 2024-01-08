#kmeans

import numpy as np
from matplotlib import pyplot
import utils


def findClosestCentroids(X, centroids):
    # Set the number of clusters
    K = centroids.shape[0]
    # The centroids are in the following variable
    idx = np.zeros(X.shape[0], dtype=int)

    # ============================================================
    for i in range(X.shape[0]):
        # Calculate the Euclidean distance between the current data point and all centroids
        distances = np.linalg.norm(X[i] - centroids, axis=1)

        # Find the index of the centroid with the minimum distance
        idx[i] = np.argmin(distances)
    # ============================================================

    return idx

# Load the dataset
data = np.load('data.npy')[:18,:2]
print("Shape of data:", data.shape)

# Set the number of clusters (K=2)
K = 2

# Initialize centroids (you may choose your own initial values)
initial_centroids = np.array([[1, 2], [11, 2], [8, 10]])
print("Updated shape of initial_centroids:", initial_centroids.shape)

# Find closest centroids
idx = findClosestCentroids(data, initial_centroids)

# Display the result
print('Closest centroids for the first 3 examples:')
print(idx[:3])

def computeCentroids(X, idx, K):
    m, n = X.shape
    # The centroids are in the following variable
    centroids = np.zeros((K, n))

    # ============================================================
    for k in range(K):
        # Find indices of points assigned to the k-th centroid
        indices = np.where(idx == k)

        # Compute mean of the points assigned to the k-th centroid
        centroids[k, :] = np.mean(X[indices], axis=0)

    # ============================================================

    return centroids

centroids = computeCentroids(data, idx, K)

print('Centroids computed after initial finding of closest centroids:')
print(centroids)

max_iters = 10


initial_centroids =  np.array([[1, 2], [11, 2], [8, 10]])


centroids, idx, anim = utils.runkMeans(data, initial_centroids,
                                       findClosestCentroids, computeCentroids, max_iters, True)


anim.save('kmeans_animation.gif', writer='imagemagick', fps=2)

print("Animation saved as 'kmeans.gif'")

pyplot.show()

def kMeansInitCentroids(X, K):

    m, n = X.shape
    
    # You should return this value correctly
    centroids = np.zeros((K, n))

    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])
    
    # Take the first K examples as centroids
    centroids = X[randidx[:K], :]
    
    return centroids


####


# Initialize centroids 
initial_centroids = kMeansInitCentroids(data, K)

# Run K-Means algorithm
max_iters = 10
centroids, idx = utils.runkMeans(data, initial_centroids, findClosestCentroids, computeCentroids, max_iters)

# Display the result
print('Centroids computed after initial finding of closest centroids:')
print(centroids)

# Plot the data points and centroids
pyplot.scatter(data[:, 0], data[:, 1], c=idx, cmap='viridis', marker='o', s=50, label='Data Points')
pyplot.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
pyplot.title('K-means Clustering with K=2')
pyplot.xlabel('Feature 1')
pyplot.ylabel('Feature 2')
pyplot.legend()
pyplot.show()
