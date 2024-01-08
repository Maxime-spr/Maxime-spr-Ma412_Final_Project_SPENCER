# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 21:20:27 2024

@author: maxime_SPR
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = np.load('data.npy')

optimal_num_clusters = 2

kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(data)

data_with_clusters = np.column_stack((data, cluster_labels))

plt.figure(figsize=(10, 6))
plt.scatter(x=data_with_clusters[:, 0], y=data_with_clusters[:, 1], c=cluster_labels, cmap='viridis', s=50)
plt.title(f'Clustering Visualization (Optimal Clusters: {optimal_num_clusters})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(title='Cluster')
plt.show()
