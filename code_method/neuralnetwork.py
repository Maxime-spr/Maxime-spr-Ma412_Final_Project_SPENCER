#neural networks
import os
import numpy as np
import matplotlib.pyplot as plt
import utils 

# Load the dataset
data = np.load('data.npy')
X, y = data[:, :-1], data[:, -1].ravel()
X = X.T  
print(X.shape)  

y[y == 10] = 0

# Number of examples to display
num_examples = min(20, X.shape[0])  

# Randomly select examples to display
indices_to_display = np.random.choice(X.shape[0], num_examples, replace=False)
X_to_display = X[indices_to_display, :]

# Display a subset of features for each example
fig, axs = plt.subplots(num_examples, figsize=(6, 6))
for i in range(num_examples):
    axs[i].imshow(X_to_display[i].reshape(9, 2), cmap='gray')
    axs[i].axis('off')

plt.show()