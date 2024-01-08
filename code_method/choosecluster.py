# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 21:19:11 2024

@author: maxime_SPR
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data = np.load('data.npy')

cluster_range = range(2, 11)  

silhouette_scores = []

for num_clusters in cluster_range:
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data)

    silhouette_avg = silhouette_score(data, cluster_labels)
    silhouette_scores.append(silhouette_avg)

    print(f"Silhouette Score for {num_clusters} clusters: {silhouette_avg}")

optimal_num_clusters = cluster_range[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters: {optimal_num_clusters}")
