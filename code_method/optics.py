import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA

# Load the dataset
data = np.load('data.npy')

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
data_reduced = pca.fit_transform(data)

# Initialize and fit the OPTICS model
optics_model = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.1)
optics_model.fit(data_reduced)

# Obtain cluster labels
labels = optics_model.labels_

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(data_reduced[:, 0], data_reduced[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.title('OPTICS Clustering Results')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.show()

# Plot the reachability plot
reachability = optics_model.reachability_[optics_model.ordering_]
plt.figure(figsize=(10, 4))
plt.plot(reachability, marker='o')
plt.title('OPTICS Reachability Plot')
plt.xlabel('Data Points')
plt.ylabel('Reachability Distance')
plt.show()
