import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# Load the data
data = np.load('data.npy')

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply DBSCAN
dbscan = DBSCAN(eps=12.5, min_samples=4)
labels = dbscan.fit_predict(data_scaled)

# Assuming 'data' is a pandas DataFrame
df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(data.shape[1])])

# Add cluster labels to the DataFrame
df['Cluster'] = labels

# Display the DataFrame with cluster labels
print(df)

columns = [f'feature_{i}' for i in range(data.shape[1])]
df = pd.DataFrame(data, columns=columns)

# Perform DBSCAN clustering
X_train = df.values
clustering = DBSCAN(eps=12.5, min_samples=4).fit(X_train)
df['Cluster'] = clustering.labels_

# Display the cluster counts
cluster_counts = df['Cluster'].value_counts().to_frame()
print("Cluster Counts:")
print(cluster_counts)

# Visualize the clusters and outliers
outliers = df[df['Cluster'] == -1]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Scatter plot for feature_0 vs. feature_1
sns.scatterplot(x='feature_0', y='feature_1', data=df, hue='Cluster', palette='Set2', ax=axes[0], legend='full', s=200)
axes[0].scatter(outliers['feature_0'], outliers['feature_1'], s=10, label='Outliers', c="red")

# Scatter plot for feature_2 vs. feature_3
sns.scatterplot(x='feature_2', y='feature_3', data=df, hue='Cluster', palette='Set2', ax=axes[1], legend='full', s=200)
axes[1].scatter(outliers['feature_2'], outliers['feature_3'], s=10, label='Outliers', c="red")

axes[0].legend()
axes[1].legend()

plt.setp(axes[0].get_legend().get_texts(), fontsize='12')
plt.setp(axes[1].get_legend().get_texts(), fontsize='12')

plt.show()


# Initialize the Nearest Neighbors model
nbrs = NearestNeighbors(n_neighbors=5).fit(data)

# Find the k-neighbors of each point in the dataset
neigh_dist, neigh_ind = nbrs.kneighbors(data)

# Sort the neighbor distances in ascending order
sort_neigh_dist = np.sort(neigh_dist, axis=0)

k_dist = sort_neigh_dist[:, 4]

plt.plot(k_dist)
plt.ylabel("k-NN distance")
plt.xlabel("Sorted observations ")
plt.show()